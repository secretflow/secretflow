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

"""
This file references code of paper SDAR: Passive Inference Attacks on Split Learning via Adversarial Regularization (https://arxiv.org/abs/2310.10483)
"""

import logging
from typing import Callable, Dict, List


import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision

from secretflow.device import PYU, reveal, wait
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.core.torch import BuilderType, module

import torch.nn.functional as F
from itertools import chain


from secretflow import reveal
from secretflow.device import PYU
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.core.torch import module
from secretflow_fl.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow_fl.ml.nn.utils import TorchModel


class SDARAttack(AttackCallback):
    def __init__(
        self,
        attack_party: PYU,
        victim_party: PYU,
        base_model_list: List[PYU],
        e_model_wrapper: TorchModel,
        decoder_model_wrapper: TorchModel,
        simulator_d_model_wrapper: TorchModel,
        decoder_d_model_wrapper: TorchModel,
        reconstruct_loss_builder: Callable,
        data_builder: Callable,
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

        # model wrapper
        self.e_model_wrapper = e_model_wrapper
        self.decoder_model_wrapper = decoder_model_wrapper
        self.simulator_d_model_wrapper = simulator_d_model_wrapper
        self.decoder_d_model_wrapper = decoder_d_model_wrapper

        self.reconstruct_loss_builder = reconstruct_loss_builder
        self.data_builder = data_builder

        self.load_attacker_path = load_attacker_path
        self.save_attacker_path = save_attacker_path

        self.logs = {}

        self.metrics = None
        self.exec_device = exec_device

    def on_train_begin(self, logs=None):
        def init_attacker(
            attack_worker: SLBaseTorchModel,
            sdar_attacker: SDARAttacker,
        ):
            attack_worker.attacker = sdar_attacker

        attacker = SDARAttacker(
            base_model_list=self.base_model_list,
            attack_party=self.attack_party,
            victim_party=self.victim_party,
            e_model_wrapper=self.e_model_wrapper,
            decoder_model_wrapper=self.decoder_model_wrapper,
            simulator_d_model_wrapper=self.simulator_d_model_wrapper,
            decoder_d_model_wrapper=self.decoder_d_model_wrapper,
            reconstruct_loss_builder=self.reconstruct_loss_builder,
            data_builder=self.data_builder,
            load_model_path=self.load_attacker_path,
            save_model_path=self.save_attacker_path,
            exec_device=self.exec_device,
        )
        self._workers[self.attack_party].apply(init_attacker, attacker)

    def on_epoch_begin(self, epoch=None, logs=None):

        def init_auxiliary_dataloader(attack_worker):
            attack_worker.attacker.aux_data = iter(
                attack_worker.attacker.aux_data_loader
            )

        self._workers[self.attack_party].apply(init_auxiliary_dataloader)

    def on_fuse_backward_end(self):

        def get_victim_hidden(worker):
            # in debug mode, we cannot return _h directly
            h = worker._h.detach().clone()
            h.requires_grad = True
            if worker._h.grad:
                h.grad = worker._h.grad.clone()
            return h

        victim_hidden = self._workers[self.victim_party].apply(get_victim_hidden)
        victim_hidden = victim_hidden.to(self.attack_party)

        def get_y(worker):
            return worker.train_y

        y = self._workers[self.attack_party].apply(get_y)

        def get_g(worker):
            return worker.model_fuse

        g_model = self._workers[self.attack_party].apply(get_g)

        def sdar_step(attack_worker, victim_hidden, y, g_model):
            attack_worker.attacker.train_step(victim_hidden, y, g_model)

        self._workers[self.attack_party].apply(sdar_step, victim_hidden, y, g_model)

    def on_train_end(self, logs=None):
        """Save the victim model and run reconstruction attacks."""

        def feature_reconstruct_attack(attack_worker, victim_model_dict: Dict):
            ret = attack_worker.attacker.attack(victim_model_dict)
            return ret

        victim_model = reveal(
            self._workers[self.victim_party].apply(lambda worker: worker.model_base)
        )
        print(self.victim_party)
        self.victim_model_dict[self.victim_party] = victim_model
        self.metrics = reveal(
            self._workers[self.attack_party].apply(
                feature_reconstruct_attack, self.victim_model_dict
            )
        )

    def get_attack_metrics(self, *args, **kwargs):
        return self.metrics


class SDARAttacker:
    def __init__(
        self,
        base_model_list: List[str],
        attack_party: str,
        victim_party: str,
        e_model_wrapper: TorchModel,
        decoder_model_wrapper: TorchModel,
        simulator_d_model_wrapper: TorchModel,
        decoder_d_model_wrapper: TorchModel,
        reconstruct_loss_builder: Callable,
        data_builder: Callable,
        load_model_path: str = None,
        save_model_path: str = None,
        exec_device: str = "cpu",
    ):
        super().__init__()

        self.base_models = {}

        self.base_model_list = base_model_list
        self.attack_party = attack_party
        self.victim_party = victim_party

        self.e_model = module.build(e_model_wrapper, device=exec_device)
        self.e_optimizer = e_model_wrapper.optim_fn(self.e_model.parameters())
        self.decoder_model = module.build(decoder_model_wrapper, device=exec_device)
        self.decoder_optimizer = decoder_model_wrapper.optim_fn(
            self.decoder_model.parameters()
        )
        self.simulator_d_model = module.build(
            simulator_d_model_wrapper, device=exec_device
        )
        self.simulator_d_optimizer = simulator_d_model_wrapper.optim_fn(
            self.simulator_d_model.parameters()
        )
        self.decoder_d_model = module.build(decoder_d_model_wrapper, device=exec_device)
        self.decoder_d_optimizer = decoder_d_model_wrapper.optim_fn(
            self.decoder_d_model.parameters()
        )

        self.reconstruct_loss = reconstruct_loss_builder()

        self.aux_data_loader, self.target_data_loader = data_builder()
        self.aux_data = None

        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.exec_device = exec_device

    def train_step(self, victim_h, y, g_model):
        x_pub, y_pub = next(self.aux_data)
        x_pub, y_pub = x_pub.to(self.exec_device), y_pub.to(self.exec_device)
        self.e_optimizer.zero_grad()
        self.simulator_d_optimizer.zero_grad()
        self.e_model.train()
        z_pub = self.e_model(x_pub)

        g_model.eval()
        y_pred_simulator = g_model(z_pub, True)

        eg_loss = F.cross_entropy(y_pred_simulator, y_pub.view(-1))

        e_dis_real_loss = 0.0
        e_dis_fake_loss = 0.0
        e_gen_loss = 0.0
        e_dis_loss = 0.0

        self.simulator_d_model.train()
        e_dis_fake_output = self.simulator_d_model(z_pub, y_pub)
        e_dis_real_output = self.simulator_d_model(victim_h, y[0])
        e_dis_real_loss = F.binary_cross_entropy_with_logits(
            e_dis_real_output, torch.ones_like(e_dis_real_output)
        )
        e_dis_fake_loss = F.binary_cross_entropy_with_logits(
            e_dis_fake_output, torch.zeros_like(e_dis_fake_output)
        )
        e_dis_loss = e_dis_real_loss + e_dis_fake_loss
        e_gen_loss = F.binary_cross_entropy_with_logits(
            e_dis_fake_output, torch.ones_like(e_dis_fake_output)
        )
        eg_loss = (
            eg_loss + e_gen_loss * 0.02
        )  # This value defined by the original code of paper
        torch.autograd.backward(
            eg_loss, inputs=list(self.e_model.parameters()), create_graph=True
        )
        self.e_optimizer.step()

        torch.autograd.backward(
            e_dis_loss,
            inputs=list(self.simulator_d_model.parameters()),
            create_graph=True,
        )
        self.simulator_d_optimizer.step()

        with torch.no_grad():
            self.e_model.train()
            zs = self.e_model(x_pub)

        self.decoder_model.train()
        decoded_x_pub = self.decoder_model(zs, y_pub.view(-1))
        d_mse_loss = torch.mean(torch.square(x_pub - decoded_x_pub))
        d_dis_loss = 0.0
        d_gen_loss = 0.0

        decoded_x = self.decoder_model(z_pub, y_pub.view(-1))
        self.decoder_d_model.train()

        d_dis_fake_output = self.decoder_d_model(decoded_x, y_pub.view(-1))
        d_dis_real_output = self.decoder_d_model(x_pub, y_pub.view(-1))
        d_dis_fake_loss = F.binary_cross_entropy_with_logits(
            d_dis_fake_output, torch.zeros_like(d_dis_fake_output)
        )
        d_dis_real_loss = F.binary_cross_entropy_with_logits(
            d_dis_real_output, torch.ones_like(d_dis_real_output)
        )

        d_dis_loss = d_dis_real_loss + d_dis_fake_loss
        d_gen_loss = F.binary_cross_entropy_with_logits(
            d_dis_fake_output, torch.ones_like(d_dis_fake_output)
        )

        d_loss = (
            d_mse_loss + d_gen_loss * 1e-5
        )  # This value defined by the original code of paper

        self.decoder_optimizer.zero_grad()
        self.decoder_d_optimizer.zero_grad()
        torch.autograd.backward(
            d_loss, inputs=list(self.decoder_model.parameters()), create_graph=True
        )
        self.decoder_optimizer.step()

        torch.autograd.backward(
            d_dis_loss,
            inputs=list(self.decoder_d_model.parameters()),
            create_graph=True,
        )
        self.decoder_d_optimizer.step()

    def evaluate(self, test_loader: torch.utils.data.DataLoader):
        with torch.no_grad():
            total_mse = 0.0
            count = 0
            mse_criterion = torch.nn.MSELoss(reduction='mean')
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(self.exec_device), y.to(self.exec_device)
                self.victim_base_model.train()
                z = self.victim_base_model(x)
                self.decoder_model.eval()
                x_recon = self.decoder_model(z, y)
                batch_mse = mse_criterion(x, x_recon)
                total_mse += batch_mse.item() * len(x)
                count += len(x)
            # uncomment this the line after to check the atteck effect
            # self.plot_attack_results(x.cpu().detach().numpy(), x_recon.cpu().detach().numpy(), "sdar_attack.png")
        logging.info(f"Average MSE over all client's images: {total_mse / count}.")
        res = {"mse": total_mse / count}
        return res

    def attack(self, victim_model_dict: Dict):
        if self.load_model_path is not None:
            self.load_model(self.load_model_path)

        self.victim_base_model = victim_model_dict[self.victim_party]
        # attack
        res = self.evaluate(self.target_data_loader)

        # save attack model
        if self.save_model_path is not None:
            self.save_model(self.save_model_path)

        return res

    def plot_attack_results(self, X, X_recon, file_name):
        import numpy as np
        import matplotlib.pyplot as plt

        X = np.transpose(X, (0, 2, 3, 1))
        X_recon = np.transpose(X_recon, (0, 2, 3, 1))
        n = len(X)
        fig, ax = plt.subplots(2, n, figsize=(n * 1.3, 3))
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(n):
            ax[0, i].imshow(X[i])
            ax[1, i].imshow(X_recon[i])
            ax[0, i].set(xticks=[], yticks=[])
            ax[1, i].set(xticks=[], yticks=[])
        # plt.show()
        plt.savefig(file_name, dpi=fig.dpi, bbox_inches='tight')
        return fig
