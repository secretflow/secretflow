# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Callable, Dict, List

import numpy as np
import torch

from secretflow import reveal
from secretflow.device import PYU
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.core.torch import TorchModel
from secretflow_fl.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel


class FeatureInferenceAttack(AttackCallback):
    """
    Implemention of feature inference attack algorithm in paper Feature inference attack on model predictions in vertical federated learning: https://arxiv.org/abs/2010.10152.
    As this algorithm use the whole VFL model to calcualte loss(y_pred, y_groundtruth), FeatureInferenceAttack should hold SLModel.
    Attributes:
        victim_model_dict: victim's model dict, party -> (model_definition, model_save_path).
        base_model_list: list of party, which decide input order of fuse model(output of base model).
        attack_party: attack party.
        generator_model_wrapper: wrapper of feature_generator model define and its optimizer.
        victim_fea_dim: victim feature dimension.
        attacker_fea_dim: attacker feature dimension.
        generator_enable_attacker_fea: whether to use attacker's feature to inference victim's feature.
        enable_mean: whether to consider distance between victim feature infered and the mean value of real victim feature in the loss, if it is true, victim_mean_feature must not be None.
        enable_var: whether to consider variance of vitim feature infered in loss.
        mean_lambda: weight of mean loss.
        var_lambda: weight of variance loss.
        victim_mean_feature: mean value of victim feature.

    """

    def __init__(
        self,
        attack_party: PYU,
        victim_party: PYU,
        base_model_list: List[PYU],
        generator_model_wrapper: TorchModel,
        data_builder: Callable,
        victim_fea_dim: List[int],
        attacker_fea_dim: List[int],
        generator_enable_attacker_fea: bool = True,
        enable_mean: bool = False,
        enable_var: bool = False,
        mean_lambda: float = 1.2,
        var_lambda: float = 0.25,
        victim_mean_feature: np.ndarray = None,
        attack_epochs: int = 60,
        load_attacker_path: str = None,
        save_attacker_path: str = None,
        exec_device: str = "cpu",
        **params,
    ):
        super().__init__(
            **params,
        )
        self.attack_party = attack_party
        self.victim_party = victim_party

        # for attacker
        self.victim_model_dict = {}
        self.base_model_list = [p.party for p in base_model_list]
        self.generator_model_wrapper = generator_model_wrapper
        self.data_builder = data_builder
        self.victim_fea_dim = victim_fea_dim
        self.attacker_fea_dim = attacker_fea_dim
        self.generator_enable_attacker_fea = generator_enable_attacker_fea
        self.enable_mean = enable_mean
        self.enable_var = enable_var
        self.mean_lambda = mean_lambda
        self.var_lambda = var_lambda
        self.victim_mean_feature = victim_mean_feature
        self.attack_epochs = attack_epochs
        self.load_attacker_path = load_attacker_path
        self.save_attacker_path = save_attacker_path
        self.logs = {}
        self.exec_device = exec_device

        self.metrics = None

    def on_train_end(self, logs=None):
        def get_victim_model(victim_worker: SLBaseTorchModel):
            return victim_worker.model_base

        def feature_inference_attack(
            attack_worker, fia_attacker: FeatureInferenceAttacker
        ):
            fia_attacker.set_attacker_model(
                attacker_base_model=attack_worker.model_base,
                attacker_fuse_model=attack_worker.model_fuse,
            )
            ret = fia_attacker.attack()
            return ret

        victim_model = reveal(self._workers[self.victim_party].apply(get_victim_model))
        self.victim_model_dict[self.victim_party.party] = victim_model
        attacker = FeatureInferenceAttacker(
            self.victim_model_dict,
            self.base_model_list,
            self.attack_party.party,
            self.generator_model_wrapper,
            self.data_builder,
            self.victim_fea_dim,
            self.attacker_fea_dim,
            self.generator_enable_attacker_fea,
            self.enable_mean,
            self.enable_var,
            self.mean_lambda,
            self.var_lambda,
            self.victim_mean_feature,
            self.attack_epochs,
            self.exec_device,
        )
        self.metrics = reveal(
            self._workers[self.attack_party].apply(feature_inference_attack, attacker)
        )

    def get_attack_metrics(self):
        return self.metrics


class FeatureInferenceAttacker:
    def __init__(
        self,
        victim_model_dict: Dict[str, torch.nn.Module],
        base_model_list: List[str],
        attack_party: str,
        generator_model_wrapper: TorchModel,
        data_builder: Callable,
        victim_fea_dim: List[int],
        attacker_fea_dim: List[int],
        generator_enable_attacker_fea: bool = True,
        enable_mean: bool = False,
        enable_var: bool = False,
        mean_lambda: float = 1.2,
        var_lambda: float = 0.25,
        victim_mean_feature: np.ndarray | torch.Tensor = None,
        epochs: int = 60,
        exec_device: str = "cpu",
        load_model_path: str = None,
        save_model_path: str = None,
    ):
        super().__init__()

        # we get all parties' base_model
        # victim's base_model: victim's base model will be saved first, then we load it
        # worker's model does not need to tocpu or gpu
        self.attacker_base_model = None
        self.attacker_fuse_model = None
        self.base_models = {}
        self.victim_model_dict = victim_model_dict

        self.base_model_list = base_model_list
        self.attack_party = attack_party

        # build generator
        # reproducible, set seed here
        self.generator_model = generator_model_wrapper.model_fn(
            **generator_model_wrapper.kwargs
        ).to(exec_device)
        self.generator_optimizer = generator_model_wrapper.optim_fn(
            self.generator_model.parameters()
        )

        self.generator_enable_attacker_fea = generator_enable_attacker_fea
        assert len(attacker_fea_dim) == len(
            victim_fea_dim
        ), "attacker_fea_dim and victim_fea_dim should have same dimension"
        for i in range(len(attacker_fea_dim) - 1):
            assert (
                attacker_fea_dim[i] == victim_fea_dim[i]
            ), "attacker_fea_dim and victim_fea_dim should have same shape except last dim"
        self.attacker_fea_dim = attacker_fea_dim
        self.victim_fea_dim = victim_fea_dim

        # data builder
        self.data_builder = data_builder
        self.epochs = epochs

        # conf for loss
        self.enable_mean = enable_mean
        self.enable_var = enable_var
        self.mean_lambda = mean_lambda
        self.var_lambda = var_lambda
        self.victim_mean_feature = (
            torch.from_numpy(victim_mean_feature)
            if isinstance(victim_mean_feature, np.ndarray)
            else victim_mean_feature
        )
        if enable_mean:
            assert (
                victim_mean_feature is not None
            ), "if enable mean, victim_mean_feature should not be None"

        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.exec_device = exec_device

    def set_attacker_model(
        self,
        attacker_base_model: torch.nn.Module,
        attacker_fuse_model: torch.nn.Module,
    ):
        self.attacker_base_model = attacker_base_model
        self.attacker_fuse_model = attacker_fuse_model

    def attack(self):
        """Begin attack."""
        # load generator model
        if self.load_model_path is not None:
            self.load_model(self.load_model_path)

        for key in self.base_model_list:
            if key == self.attack_party:
                self.base_models[key] = self.attacker_base_model
            else:
                self.base_models[key] = self.victim_model_dict[key]
            self.base_models[key] = self.base_models[key].to(self.exec_device)

        # prepare data
        train_loaders, test_loader = self.data_builder()

        # attack
        res = self.train(train_loaders, test_loader, self.epochs)

        # save generator model
        if self.save_model_path is not None:
            self.save_model(self.save_model_path)
        return res

    def train(
        self,
        train_loaders: Dict[str, torch.utils.data.DataLoader],
        test_loader: Dict[str, torch.utils.data.DataLoader] = None,
        epochs: int = 60,
    ):
        """Train attacker's generator.
        Args:
            train_loaders: dict of dataloader, key is party, value is its dataloader; it must contains attacker and victim's dataloader
            test_loader: dataloaders for evaluation
            epochs: epoch number
        """
        res = None
        batch_num = -1
        for epoch in range(epochs):
            # prepare
            # freeze sl model
            train_data_iter = {}
            for key in train_loaders:
                if batch_num == -1:
                    batch_num = len(train_loaders[key])
                else:
                    assert batch_num == len(
                        train_loaders[key]
                    ), f"train_loaders length {len(train_loaders[key])} and batch_num {batch_num} should be same"
                train_data_iter[key] = iter(train_loaders[key])
                self.base_models[key].eval()
            self.attacker_fuse_model.eval()
            self.generator_model.train()

            losses = []
            for batch_idx in range(batch_num):
                self.generator_optimizer.zero_grad()
                attacker_fea = None

                # calcaulate logit_groundtruth and get attacker's feature
                hiddens = []
                for base_key in self.base_model_list:
                    if base_key == self.attack_party:
                        [attacker_fea] = next(train_data_iter[base_key])
                        attacker_fea = attacker_fea.to(self.exec_device)
                        hid = self.base_models[base_key](attacker_fea)
                        hiddens.append(hid)
                    else:
                        [fea] = next(train_data_iter[base_key])
                        fea = fea.to(self.exec_device)
                        hid = self.base_models[base_key](fea)
                        hiddens.append(hid)
                y_groundtruth = self.attacker_fuse_model(hiddens)

                # infer victim's feature
                if self.generator_enable_attacker_fea:
                    # reproducible: set seed here
                    rnd_shape = [attacker_fea.size(0)] + self.victim_fea_dim
                    noise = torch.randn(rnd_shape).to(self.exec_device)
                    generator_input = torch.cat((attacker_fea, noise), dim=-1)
                else:
                    # reproducible: set seed here
                    rnd_shape = (
                        [attacker_fea.size(0)]
                        + self.victim_fea_dim[:-1]
                        + [self.victim_fea_dim[-1] + self.attacker_fea_dim[-1]]
                    )
                    generator_input = torch.randn(rnd_shape)
                generator_output = self.generator_model(generator_input)

                # infer logit_pred
                hiddens = []
                for base_key in self.base_model_list:
                    if base_key == self.attack_party:
                        hid = self.base_models[base_key](attacker_fea)
                        hiddens.append(hid)
                    else:  # victim
                        hid = self.base_models[base_key](generator_output)
                        hiddens.append(hid)

                y_pred = self.attacker_fuse_model(hiddens)

                loss = ((y_pred - y_groundtruth.detach()) ** 2).sum()

                if self.enable_mean:
                    mean_loss = 0
                    for i in range(generator_output.size(1)):
                        mean_loss = (
                            mean_loss
                            + (
                                generator_output[:, i].mean()
                                - self.victim_mean_feature[i]
                            )
                            ** 2
                        )
                    loss += self.mean_lambda * mean_loss

                if self.enable_var:
                    unknown_var_loss = 0
                    for i in range(generator_output.size(1)):
                        unknown_var_loss = unknown_var_loss + (
                            generator_output[:, i].var()
                        )
                    loss += self.var_lambda * unknown_var_loss

                loss.backward()
                losses.append(loss.detach())
                self.generator_optimizer.step()

            # TODO: here to report
            logging.info(f"In epoch {epoch}, loss is {sum(losses) / len(losses)}")

            # evaluation
            if test_loader is not None:
                res = self.evaluate(test_loader)
        return res

    def evaluate(self, test_loaders: Dict[str, torch.utils.data.DataLoader]):
        """Evaluate generator.
        Args:
            test_loaders: dict of data_loader, key is party, value is its dataloader; it must contains groudtruth(real victim feature); if generator_enable_attacker_fea is true, test_loaders should contain attacker's dataloader
        """

        def loss_per_feature(input, target):
            res = []
            for i in range(input.size(1)):
                loss = ((input[:, i] - target[:, i]) ** 2).mean().item()
                res.append(loss)
            return np.array(res)

        self.generator_model.eval()

        mse = torch.nn.MSELoss(reduction="mean")
        generator_losses = []
        random_losses = []
        total_model_loss_pf = None
        total_random_loss_pf = None

        # prepare
        attacker_data_loader = None
        victim_data_loader = None
        batch_num = -1
        for key in test_loaders:
            if batch_num == -1:
                batch_num = len(test_loaders[key])
            else:
                assert batch_num == len(
                    test_loaders[key]
                ), f"length of all dataloaders should be same"

            if key == self.attack_party:
                attacker_data_loader = iter(test_loaders[key])
            else:
                victim_data_loader = iter(test_loaders[key])

        for batch_idx in range(batch_num):
            # groud truth
            [ground_truth] = next(victim_data_loader)
            ground_truth = ground_truth.to(self.exec_device)
            # infer victim feature
            if self.generator_enable_attacker_fea:
                noise = torch.randn([ground_truth.size(0)] + self.victim_fea_dim).to(
                    self.exec_device
                )
                [attacker_fea] = next(attacker_data_loader)
                attacker_fea = attacker_fea.to(self.exec_device)
                generator_input = torch.cat((attacker_fea, noise), dim=-1)
            else:
                rnd_shape = (
                    [ground_truth.size(0)]
                    + self.victim_fea_dim[:-1]
                    + [self.victim_fea_dim[-1] + self.attacker_fea_dim[-1]]
                )
                generator_input = torch.randn(rnd_shape).to(self.exec_device)

            generator_output = self.generator_model(generator_input)

            # random guess victim feature
            if self.enable_mean:
                randomguess = self.victim_mean_feature.repeat(
                    generator_output.size(0), 1
                )
                randomguess = randomguess + torch.normal(
                    0, 1 / 2, size=randomguess.size()
                )
                randomguess = randomguess.clamp(0, 1)
            else:
                randomguess = torch.rand_like(generator_output)

            model_loss = mse(ground_truth, generator_output).item()
            rand_loss = mse(ground_truth, randomguess).item()

            generator_losses.append(model_loss)
            random_losses.append(rand_loss)
            model_loss_pf = loss_per_feature(ground_truth, generator_output)
            random_loss_pf = loss_per_feature(ground_truth, randomguess)
            total_model_loss_pf = (
                model_loss_pf
                if total_model_loss_pf is None
                else total_model_loss_pf + model_loss_pf
            )
            total_random_loss_pf = (
                random_loss_pf
                if total_random_loss_pf is None
                else total_random_loss_pf + random_loss_pf
            )

        mean_model_loss = sum(generator_losses) / len(generator_losses)
        mean_guess_loss = sum(random_losses) / len(random_losses)
        mean_model_loss_pf = total_model_loss_pf / len(generator_losses)
        mean_guess_loss_pf = total_random_loss_pf / len(random_losses)
        logging.info(f"Mean generator loss: {mean_model_loss}")
        logging.info(f"Mean random guess loss: {mean_guess_loss}")
        logging.info(f"Mean generator loss Per Feature: {mean_model_loss_pf}")
        logging.info(f"Mean random guess loss Per Feature: {mean_guess_loss_pf}")

        res = {
            "mean_model_loss": mean_model_loss,
            "mean_guess_loss": mean_guess_loss,
            "mean_model_loss_pf": mean_model_loss_pf,
            "mean_guess_loss_pf": mean_guess_loss_pf,
        }

        return res

    def save_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        check_point = {
            "generator_model_state_dict": self.generator_model.state_dict(),
            "optimizer_state_dict": self.generator_optimizer.state_dict(),
        }
        torch.save(check_point, model_path)
        return 0

    def load_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        checkpoint = torch.load(model_path)
        self.generator_model.load_state_dict(checkpoint["generator_model_state_dict"])
        self.generator_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return 0
