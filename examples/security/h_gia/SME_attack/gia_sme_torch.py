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
from torchvision.transforms import ToPILImage
import os
import copy
import json
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from secretflow import reveal
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel

def require_grad(net, flag):
    for p in net.parameters():
        p.require_grad = flag

def prior_boundary(data, low, high):
    with torch.no_grad():
        data.data = torch.clamp(data, low, high)

def compute_norm(inputs):
    squared_sum = sum([p.square().sum() for p in inputs])
    norm = squared_sum.sqrt()
    return norm

def total_variation(x):
    dh = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean()
    dw = (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    return (dh + dw) / 2

def psnr(data, rec, sort=False):
    assert data.max().item() <= 1.0001 and data.min().item() >= -0.0001
    assert rec.max().item() <= 1.0001 and rec.min().item() >= -0.0001
    cost_matrix = []
    if sort:
        for x_ in rec:
            cost_matrix.append(
                [(x_ - d).square().mean().item() for d in data]
            )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert np.all(row_ind == np.arange(len(row_ind)))
        data = data[col_ind]
    psnr_list = [10 * np.log10(1 / (d - r).square().mean().item()) for d, r in zip(data, rec)]
    return np.mean(psnr_list)

def save_figs(tensors, path, subdir=None, dataset=None):
    def save(imgs, path):
        for name, im in imgs:
            plt.figure()
            plt.imshow(im, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(path, f'{name}.png'), bbox_inches='tight')
            plt.close()
    tensor2image = ToPILImage()
    path = os.path.join(path, subdir)
    os.makedirs(path, exist_ok=True)
    if dataset == "FEMNIST":
        tensors = 1 - tensors
    imgs = [
        [i, tensor2image(tensors[i].detach().cpu().squeeze())] for i in range(len(tensors))
    ]
    save(imgs, path)


class GiadentInversionAttackSME(AttackCallback):

    def __init__(self, attack_party, victim_party, attack_configs, **kwargs):
        super().__init__(**kwargs)

        self.attack_party = attack_party
        self.victim_party = victim_party
        self.attack_configs = attack_configs
        self.metrics = None

    def on_train_begin(self, logs=None):
        def get_victim_model(victim_worker: BaseTorchModel):
            print('get victim model before training')
            return victim_worker.model
        
        #get victim model before training
        self.victim_model1 = copy.deepcopy(reveal(self._workers[self.victim_party].apply(get_victim_model)))

        return 

    def on_train_end(self, logs=None):
        def get_victim_model(victim_worker: BaseTorchModel):
            print('get victim model after training')
            return victim_worker.model
        
        def get_victim_trainloader(victim_worker: BaseTorchModel):
            return victim_worker.train_set
        
        victim_trainloader = reveal(self._workers[self.victim_party].apply(get_victim_trainloader))

        self.victim_model2 = copy.deepcopy(reveal(self._workers[self.victim_party].apply(get_victim_model)))
        

        setup = {"device": "cuda", "dtype": torch.float32}
        path_to_res="./res"
        attacker = SME(
            trainloader=victim_trainloader,
            setup=setup,
            alpha=self.attack_configs['alpha'],
            test_steps=self.attack_configs['test_steps'],
            path_to_res=self.attack_configs['path_to_res'],
            lamb=self.attack_configs['lamb'],
            mean_std=(0., 1.),
        )

        attacker.net0 = self.victim_model1
        attacker.net1 = self.victim_model2

        # Reconstruction
        def gia_sme_attack(sme_attacker: SME):
            stats = sme_attacker.reconstruction(
                eta=self.attack_configs['eta'],
                beta=self.attack_configs['beta'],
                iters=self.attack_configs['iters'],
                lr_decay=self.attack_configs['lr_decay'],
                save_figure=self.attack_configs['save_figure'],
            )
            with open(os.path.join(path_to_res, "res.json"), "w") as f:
                json.dump(stats, f, indent=4)
            
            return stats
        
        smetrics = gia_sme_attack(attacker)
    
    def get_attack_metrics(self):
        return self.metrics


class SME:
    def __init__(
            self,
            trainloader,
            setup,
            alpha,
            test_steps,
            path_to_res,
            mean_std,
            lamb,
            dataset=None,
    ):
        self.alpha = torch.tensor(alpha, requires_grad=True, **setup)
        self.rec_alpha = 0 < self.alpha < 1
        self.setup = setup
        self.net0 = None
        self.net1 = None
        self.test_steps = test_steps
        os.makedirs(path_to_res, exist_ok=True)
        self.path = path_to_res
        self.lamb = lamb
        self.dataset = dataset
        data, labels = [], []
        for img, l in trainloader:
            labels.append(l)
            data.append(img)
        self.data = torch.cat(data).to(**setup)

        # We assume that labels have been restored separately, for details please refer to the paper.
        self.y = torch.cat(labels).to(device=setup["device"])
        # Dummy input.
        self.x = torch.normal(0, 1, size=self.data.shape, requires_grad=True, **setup)

        self.mean = torch.tensor(mean_std[0]).to(**setup).reshape(1, -1, 1, 1)
        self.std = torch.tensor(mean_std[1]).to(**setup).reshape(1, -1, 1, 1)
        # This is a trick (a sort of prior information) adopted from IG.
        prior_boundary(self.x, -self.mean / self.std, (1 - self.mean) / self.std)

    def reconstruction(self, eta, beta, iters, lr_decay, signed_grad=False, save_figure=True):
        self.net0.to(device=self.setup['device'])
        self.net1.to(device=self.setup['device'])
        # when taking the SME strategy, alpha is set within (0, 1).
        if 0 < self.alpha < 1:
            self.alpha.grad = torch.tensor(0.).to(**self.setup)
        optimizer = torch.optim.Adam(params=[self.x], lr=eta)
        alpha_opti = torch.optim.Adam(params=[self.alpha], lr=beta)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[iters // 2.667,
                                                                     iters // 1.6,
                                                                     iters // 1.142],
                                                         gamma=0.1)
        alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(alpha_opti,
                                                               milestones=[iters // 2.667,
                                                                           iters // 1.6,
                                                                           iters // 1.142],
                                                               gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        # Direction of the weight update.
        w1_w0 = []
        for p0, p1 in zip(self.net0.parameters(), self.net1.parameters()):
            w1_w0.append(p0.data - p1.data)
        norm = compute_norm(w1_w0)
        w1_w0 = [p / norm for p in w1_w0]

        # Construct the model for gradient inversion attack.
        require_grad(self.net0, False)
        require_grad(self.net1, False)
        with torch.no_grad():
            _net = copy.deepcopy(self.net0)
            _net.to(device=self.setup['device'])
            for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                z.data = (1 - self.alpha) * p + self.alpha * q

        # Reconstruction
        _net.eval()
        stats = []
        for i in range(iters):
            optimizer.zero_grad()
            alpha_opti.zero_grad(set_to_none=False)
            _net.zero_grad()

            if self.rec_alpha:
                # Update the surrogate model.
                with torch.no_grad():
                    for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                        z.data = (1 - self.alpha) * p + self.alpha * q
            pred = _net(self.x)
            loss = criterion(input=pred, target=self.y)
            grad = torch.autograd.grad(loss, _net.parameters(), create_graph=True)
            norm = compute_norm(grad)
            grad = [p / norm for p in grad]

            # Compute x's grad.
            cos_loss = 1 - sum([
                p.mul(q).sum() for p, q in zip(w1_w0, grad)
            ])
            loss = cos_loss + self.lamb * total_variation(self.x)
            loss.backward()
            if signed_grad:
                self.x.grad.sign_()

            # Compute alpha's grad.
            if self.rec_alpha:
                with torch.no_grad():
                    for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                        self.alpha.grad += z.grad.mul(
                            q.data - p.data
                        ).sum()
                if signed_grad:
                    self.alpha.grad.sign_()

            # Update x and alpha.
            optimizer.step()
            alpha_opti.step()
            prior_boundary(self.x, -self.mean / self.std, (1 - self.mean) / self.std)
            prior_boundary(self.alpha, 0, 1)
            if lr_decay:
                scheduler.step()
                alpha_scheduler.step()
            if i % self.test_steps == 0 or i == iters - 1:
                with torch.no_grad():
                    _x = self.x * self.std + self.mean
                    _data = self.data * self.std + self.mean
                measurement = psnr(_data, _x, sort=True)
                print(f"iter: {i}| alpha: {self.alpha.item():.2f}| (1 - cos): {cos_loss.item():.3f}| "
                      f"psnr: {measurement:.3f}")
                stats.append({
                    "iter": i,
                    "alpha": self.alpha.item(),
                    "cos_loss": cos_loss.item(),
                    "psnr": measurement,
                })
                if save_figure:
                    save_figs(tensors=_x, path=self.path, subdir=str(i), dataset=self.dataset)
        if save_figure:
            save_figs(tensors=self.data * self.std + self.mean,
                      path=self.path, subdir="original", dataset=self.dataset)
        return stats
