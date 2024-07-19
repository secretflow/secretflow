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
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.core.torch import module
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.utils import TorchModel


class FeatureSpaceHijackingAttack(AttackCallback):
    """
    Implemention of feature space hijacking attack algorithm in paper Unleashing the Tiger: Inference Attacks on Split Learning: https://arxiv.org/abs/2012.02670.

    Attributes:
        victim_model_dict: victim's model dict, party -> (model_definition, model_save_path).
        base_model_list: list of party, which decide input order of fuse model(output of base model).
        attack_party: attack party.
        victim_party: victim party.
        pilot_model_wrapper: wrapper of the pilot model define and its optimizer.
        decoder_model_wrapper: wrapper of the decoder model define and its optimizer.
        discriminator_model_wrapper: wrapper of the discriminator model define and its optimizer.
        reconstruct_loss_builder: return a loss function for feature reconstruction.
        data_builder: build data loaders for target dataset and auxiliary dataset.
        victim_fea_dim: victim feature dimension.
        attacker_fea_dim: attacker feature dimension.
        gradient_penalty_weight: weight of gradient penalty loss.
        load_attacker_path: path to load attacker models.
        save_attacker_path: path to save attacker models.
    """

    def __init__(
        self,
        base_model_list: List[PYU],
        attack_party: PYU,
        victim_party: PYU,
        pilot_model_wrapper: TorchModel,
        decoder_model_wrapper: TorchModel,
        discriminator_model_wrapper: TorchModel,
        reconstruct_loss_builder: Callable,
        data_builder: Callable,
        victim_fea_dim: int,
        attacker_fea_dim: int,
        gradient_penalty_weight: float = 100,
        load_attacker_path: str = None,
        save_attacker_path: str = None,
        exec_device: str = 'cpu',
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
        self.pilot_model_wrapper = pilot_model_wrapper
        self.decoder_model_wrapper = decoder_model_wrapper
        self.discriminator_model_wrapper = discriminator_model_wrapper

        self.reconstruct_loss_builder = reconstruct_loss_builder
        self.data_builder = data_builder
        self.victim_fea_dim = victim_fea_dim
        self.attacker_fea_dim = attacker_fea_dim

        self.gradient_penalty_weight = gradient_penalty_weight
        self.load_attacker_path = load_attacker_path
        self.save_attacker_path = save_attacker_path

        self.logs = {}

        self.metrics = None
        self.exec_device = exec_device

    def on_train_begin(self, logs=None):
        def init_attacker(
            attack_worker: SLBaseTorchModel,
            fsha_attacker: FeatureSpaceHijackingAttacker,
        ):
            attack_worker.attacker = fsha_attacker

        attacker = FeatureSpaceHijackingAttacker(
            base_model_list=self.base_model_list,
            attack_party=self.attack_party.party,
            victim_party=self.victim_party.party,
            pilot_model_wrapper=self.pilot_model_wrapper,
            decoder_model_wrapper=self.decoder_model_wrapper,
            discriminator_model_wrapper=self.discriminator_model_wrapper,
            reconstruct_loss_builder=self.reconstruct_loss_builder,
            data_builder=self.data_builder,
            victim_fea_dim=self.victim_fea_dim,
            attacker_fea_dim=self.attacker_fea_dim,
            gradient_penalty_weight=self.gradient_penalty_weight,
            load_model_path=self.load_attacker_path,
            save_model_path=self.save_attacker_path,
            exec_device=self.exec_device,
        )
        self._workers[self.attack_party].apply(init_attacker, attacker)

    def on_epoch_begin(self, epoch=None, logs=None):
        """Initialize an iterator of auxiliary dataset before each epoch."""

        def init_auxiliary_dataloader(attack_worker):
            attack_worker.attacker.aux_data = iter(
                attack_worker.attacker.aux_data_loader
            )

        self._workers[self.attack_party].apply(init_auxiliary_dataloader)

    def on_fuse_backward_end(self):
        """Replace victim gradients with malicious gradients."""

        victim_idx = self.base_model_list.index(self.victim_party.party)

        def get_victim_hidden(worker):
            # in debug mode, we cannot return _h directly
            h = worker._h.detach().clone()
            h.requires_grad = True
            if worker._h.grad:
                h.grad = worker._h.grad.clone()
            return h

        victim_hidden = self._workers[self.victim_party].apply(get_victim_hidden)
        victim_hidden = victim_hidden.to(self.attack_party)

        def replace_gradients(attack_worker, victim_hidden, victim_idx):
            grad = attack_worker.attacker.train_step(victim_hidden)
            attack_worker._gradient[victim_idx] = grad

        self._workers[self.attack_party].apply(
            replace_gradients, victim_hidden, victim_idx
        )

    def on_train_end(self, logs=None):
        """Save the victim model and run reconstruction attacks."""

        def feature_reconstruct_attack(attack_worker, victim_model_dict: Dict):
            ret = attack_worker.attacker.attack(victim_model_dict)
            return ret

        victim_model = reveal(
            self._workers[self.victim_party].apply(lambda worker: worker.model_base)
        )
        self.victim_model_dict[self.victim_party.party] = victim_model
        self.metrics = reveal(
            self._workers[self.attack_party].apply(
                feature_reconstruct_attack, self.victim_model_dict
            )
        )

    def get_attack_metrics(self):
        return self.metrics


class FeatureSpaceHijackingAttacker:
    def __init__(
        self,
        base_model_list: List[str],
        attack_party: str,
        victim_party: str,
        pilot_model_wrapper: TorchModel,
        decoder_model_wrapper: TorchModel,
        discriminator_model_wrapper: TorchModel,
        reconstruct_loss_builder: Callable,
        data_builder: Callable,
        victim_fea_dim: int,
        attacker_fea_dim: int,
        gradient_penalty_weight: float = 100,
        load_model_path: str = None,
        save_model_path: str = None,
        exec_device: str = 'cpu',
    ):
        super().__init__()

        self.base_models = {}

        self.base_model_list = base_model_list
        self.attack_party = attack_party
        self.victim_party = victim_party
        self.victim_base_model = None
        # build pilots
        self.pilot_model = module.build(pilot_model_wrapper, device=exec_device)
        self.pilot_optimizer = pilot_model_wrapper.optim_fn(
            self.pilot_model.parameters()
        )
        # build decoder
        self.decoder_model = module.build(decoder_model_wrapper, device=exec_device)
        self.decoder_optimizer = decoder_model_wrapper.optim_fn(
            self.decoder_model.parameters()
        )
        # build discriminator
        self.discriminator_model = module.build(
            discriminator_model_wrapper, device=exec_device
        )
        self.discriminator_optimizer = discriminator_model_wrapper.optim_fn(
            self.discriminator_model.parameters()
        )

        self.reconstruct_loss = reconstruct_loss_builder()

        self.attacker_fea_dim = attacker_fea_dim
        self.victim_fea_dim = victim_fea_dim

        # data builder
        # self.data_builder = data_builder
        self.target_data_loader, self.aux_data_loader = data_builder()
        self.gp_weight = gradient_penalty_weight

        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.aux_data = None
        self.exec_device = exec_device

    def attack(self, victim_model_dict: Dict):
        """Begin attack."""

        # load attack model
        if self.load_model_path is not None:
            self.load_model(self.load_model_path)

        self.victim_base_model = victim_model_dict[self.victim_party]
        # attack
        res = self.evaluate(self.target_data_loader)

        # save attack model
        if self.save_model_path is not None:
            self.save_model(self.save_model_path)
        return res

    def train_step(self, victim_h):
        """Update attack models and return malicious gradients."""
        victim_h.retain_grad()
        [x_pub] = next(self.aux_data)
        x_pub = x_pub.to(self.exec_device)

        self.pilot_model.train()
        self.decoder_model.train()
        self.discriminator_model.train()

        self.pilot_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        # train base_model(generator)
        adv_priv_logits = self.discriminator_model(victim_h)
        priv_d_loss = torch.mean(adv_priv_logits)  # base_model, discriminator
        priv_d_loss.backward(retain_graph=True)
        victim_grads = (
            victim_h.grad.detach()
        )  # dsc_loss.backward() changes victim_h.grad

        # train pilot and decoder
        z_pub = self.pilot_model(x_pub)
        rec_x_pub = self.decoder_model(z_pub)
        pub_rec_loss = self.reconstruct_loss(x_pub, rec_x_pub)  # pilot, decoder
        pub_rec_loss.backward(retain_graph=True)
        # store pilot grad first as dsc_loss generate pilot grad also
        pilot_grad = [p.grad.detach() for p in self.pilot_model.parameters()]
        self.decoder_optimizer.step()

        # train discriminator
        # priv_d_loss should not impact self.discriminator_model.param, clear grad first
        # p.grad.zero_() will get same grad with torch.autograd.grad(dsc_loss, self.discriminator_model.parameters())
        # get different grad using self.discriminator_optimizer.zero_grad() or p.grad = None
        # maybe because it removes edge between discriminator_model.param and adv_priv_logits?
        for name, p in self.discriminator_model.named_parameters():
            p.grad.zero_()
        adv_pub_logits = self.discriminator_model(z_pub)
        dsc_loss = torch.mean(adv_pub_logits) - torch.mean(adv_priv_logits)
        gp_loss = self._gradient_penalty(victim_h, z_pub)
        (dsc_loss + self.gp_weight * gp_loss).backward()
        self.discriminator_optimizer.step()

        # update pilot
        for p, g in zip(self.pilot_model.parameters(), pilot_grad):
            p.grad = g
        self.pilot_optimizer.step()

        return victim_grads

    def evaluate(self, test_loader: torch.utils.data.DataLoader):
        """Evaluate attack models.
        Args:
            test_loader: victim data loader;
        """

        def loss_per_feature(input, target):
            res = []
            for i in range(input.size(1)):
                loss = ((input[:, i] - target[:, i]) ** 2).mean().item()
                res.append(loss)
            return np.array(res)

        self.decoder_model.eval()

        mse = torch.nn.MSELoss(reduction='mean')
        model_losses = []
        random_losses = []
        total_model_loss_pf = None
        total_random_loss_pf = None

        # prepare
        victim_data_loader = None
        batch_num = len(test_loader)
        victim_data_loader = iter(test_loader)

        for batch_idx in range(batch_num):
            # groud truth
            [victim_fea] = next(victim_data_loader)
            victim_fea = victim_fea.to(self.exec_device)
            victim_h = self.victim_base_model(victim_fea)
            decoder_output = self.decoder_model(victim_h)

            # random guess victim feature
            randomguess = torch.rand_like(decoder_output)
            model_loss = mse(victim_fea, decoder_output).item()
            rand_loss = mse(victim_fea, randomguess).item()

            model_losses.append(model_loss)
            random_losses.append(rand_loss)
            model_loss_pf = loss_per_feature(victim_fea, decoder_output)
            random_loss_pf = loss_per_feature(victim_fea, randomguess)
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

        mean_model_loss = sum(model_losses) / len(model_losses)
        mean_guess_loss = sum(random_losses) / len(random_losses)
        mean_model_loss_pf = total_model_loss_pf / len(model_losses)
        mean_guess_loss_pf = total_random_loss_pf / len(random_losses)
        logging.info(f"Mean model loss: {mean_model_loss}")
        logging.info(f"Mean random guess loss: {mean_guess_loss}")
        logging.info(f"Mean model loss Per Feature: {mean_model_loss_pf}")
        logging.info(f"Mean random guess loss Per Feature: {mean_guess_loss_pf}")

        res = {
            'mean_model_loss': mean_model_loss,
            'mean_guess_loss': mean_guess_loss,
            'mean_model_loss_pf': mean_model_loss_pf,
            'mean_guess_loss_pf': mean_guess_loss_pf,
        }

        return res

    def save_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        check_point = {
            'pilot_model_state_dict': self.pilot_model.state_dict(),
            'pilot_optimizer_state_dict': self.pilot_optimizer.state_dict(),
            'decoder_model_state_dict': self.decoder_model.state_dict(),
            'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
            'discriminator_model_state_dict': self.discriminator_model.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        }
        torch.save(check_point, model_path)
        return 0

    def load_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        checkpoint = torch.load(model_path)
        self.pilot_model.load_state_dict(checkpoint['pilot_model_state_dict'])
        self.pilot_optimizer.load_state_dict(checkpoint['pilot_optimizer_state_dict'])
        self.decoder_model.load_state_dict(checkpoint['decoder_model_state_dict'])
        self.decoder_optimizer.load_state_dict(
            checkpoint['decoder_optimizer_state_dict']
        )
        self.discriminator_model.load_state_dict(
            checkpoint['discriminator_model_state_dict']
        )
        self.discriminator_optimizer.load_state_dict(
            checkpoint['discriminator_optimizer_state_dict']
        )
        return 0

    def _gradient_penalty(self, x, x_gen):
        """Gradient regularization."""

        bs = x.size(0)
        size = [bs] + (len(x.shape) - 1) * [1]
        epsilon = torch.rand(*size).to(self.exec_device)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        x_hat.requires_grad_(True)
        d_hat = self.discriminator_model(x_hat)

        ones = torch.ones(d_hat.size(), requires_grad=False).to(self.exec_device)
        gradients = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(bs, -1)
        ddx = torch.sqrt(torch.sum(gradients**2, axis=1))

        d_regularizer = torch.mean((ddx - 1.0) ** 2)

        return d_regularizer
