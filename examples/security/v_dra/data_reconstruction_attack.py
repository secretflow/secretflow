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
This file references code of paper GAN You See Me? Enhanced Data Reconstruction Attacks against Split Inference
"""

import logging

import torch
import torch.optim as optim
from torchvision.transforms import Normalize
from model_stylegan2 import Generator

from secretflow.ml.nn.sl.backend.torch.callback import Callback


class DeNormalize(torch.nn.Module):
    """
    DeNormalize a tensor image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def _denormalize(self, tensor):
        tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(
                'std evaluated to zero after conversion to {}, leading to division by zero.'.format(
                    dtype
                )
            )
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)

        return tensor

    def forward(self, x):
        return self._denormalize(x)


class FeatureLoss(object):
    """
    Loss function for Feature Space Distance
    """

    def __call__(self, x, y):
        return ((x - y) ** 2).mean()


def tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


class TVLoss(object):
    """
    Loss function for Total Variation
    """

    def __call__(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = tensor_size(x[:, :, 1:, :])
        count_w = tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : w_x - 1], 2).sum()

        return (h_tv / count_h + w_tv / count_w) / batch_size


class L2Loss(object):
    """
    Loss function for L2-Norm / Mean Square Error
    """

    def __call__(self, x):
        return (x**2).mean()


class KLDLoss(object):
    """
    Loss function for Kullback–Leibler Divergence
    """

    def __call__(self, x):
        return -0.5 * torch.sum(
            1
            + torch.log(torch.std(x.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10)
            - torch.mean(x.squeeze(), axis=-1).pow(2)
            - torch.std(x.squeeze(), unbiased=False, axis=-1).pow(2)
        )


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise


class DataReconstructionAttacker(Callback):
    """
    Implementation of GAN-based LAtent Space Search (GLASS) aglorithm in paper GAN You See Me? Enhanced Data Reconstruction Attacks against Split Inference (under review).
    Attributes:
        device:
        trg_model: target model, it should has a client model samed as model_base in SLModel, and a server model samed as model_fuse in SLModel.
        att_model: attack model, it is based on StyleGAN2.
        lr: learning rate for optimizer.
        restarts: number of candidate z.
        iter_z: number of iterations for optimizing z.
        iter_w: number of iterations for optimizing w+.
        lambda_TV: weight of TV loss.
        lambda_L2: weight of L2 loss.
        lambda_KLD: weight of KLD loss.
    """

    def __init__(
        self,
        device: torch.device,
        trg_model: torch.nn.Module,
        att_model: Generator,
        lr: float = 1e-2,
        restarts: int = 100,
        iter_z: int = 100,
        iter_w: int = 20000,
        lambda_TV: float = 1e-2,
        lambda_L2: float = 1e-2,
        lambda_KLD: float = 1,
    ):
        super().__init__()

        self.device = device

        self.trg_model = trg_model
        self.att_model = att_model

        self.lr = lr
        self.restarts = restarts
        self.iter_z = iter_z
        self.iter_w = iter_w

        self.lambda_TV = lambda_TV
        self.lambda_L2 = lambda_L2
        self.lambda_KLD = lambda_KLD

        self.train_criterion = FeatureLoss()
        self.tv = TVLoss()
        self.l2 = L2Loss()
        self.kld = KLDLoss()

        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.denorm = DeNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def attack(self, trg_feat):
        """
        Attack process of GLASS algorithm
        Args:
            trg_feat: The target intermediate features exploited by attacker
        """
        self.att_model.eval()

        # find the mean and std of w latent space, n_latent set to 10000
        with torch.no_grad():
            n_w_latent = 10000
            noise_sample = torch.randn(
                n_w_latent, self.att_model.style_dim, device=self.device
            )
            latent_out = self.att_model.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_w_latent) ** 0.5

        sample_z = [None for _ in range(self.restarts)]
        optimizer = [None for _ in range(self.restarts)]
        scores = torch.zeros(self.restarts)

        # initialize candidate z
        for re in range(self.restarts):
            sample_z[re] = torch.randn(1, self.att_model.style_dim, device=self.device)
            sample_z[re].requires_grad_()

            optimizer[re] = optim.Adam(
                params=[sample_z[re]], lr=self.lr, eps=1e-3, amsgrad=True
            )

        # restart to optimize the next candidate z
        for z_index in range(self.restarts):
            # z space search
            for i in range(self.iter_z):
                sample, _ = self.att_model([sample_z[z_index]], truncation=1)

                # convert sample to the distribution of normal images
                Min = -1
                Max = 1
                sample.clamp_(min=Min, max=Max)
                sample.add_(-Min).div_(Max - Min + 1e-5)

                sample = self.norm(sample)
                sample_feat = self.trg_model.client_model(sample)

                L_feat = self.train_criterion(sample_feat, trg_feat)
                L_tv = self.tv(sample)
                L_l2 = self.l2(sample)
                L_kld = self.kld(sample_z[z_index])

                optimizer[z_index].zero_grad()
                loss = (
                    L_feat
                    + self.lambda_TV * L_tv
                    + self.lambda_L2 * L_l2
                    + self.lambda_KLD * L_kld
                )
                loss.backward(retain_graph=True)
                optimizer[z_index].step()

                logging.info(
                    f"z_index:{z_index}, Iter optimize z: {i}, FeatureLoss: {L_feat.item()}, TVLoss: {L_tv.item()}, L2Loss: {L_l2.item()}, KLDLoss: {L_kld.item()}"
                )

            scores[z_index] = L_feat.detach()

        # obtain the optimal z
        tok1_of_restarts_indices = torch.topk(scores, k=1, largest=False)
        latent_init_z = sample_z[tok1_of_restarts_indices.indices[0]].data
        latent_init_w = self.att_model.style(latent_init_z)
        latent_in = latent_init_w.detach().clone()

        # initialize w+ latent space
        latent_in = latent_in.unsqueeze(1).repeat(1, self.att_model.n_latent, 1)
        latent_in.requires_grad = True

        optimizer = optim.Adam(params=[latent_in], lr=self.lr, eps=1e-3, amsgrad=True)

        noise_step = 10000
        noise_level = 0.05
        noise_ramp = 0.75

        # w+ space search
        for j in range(self.iter_w):
            # embed noise into w+ latent space
            t = j / noise_step
            noise_strength = latent_std * noise_level * max(0, 1 - t / noise_ramp) ** 2
            latent_in_with_noise = latent_noise(latent_in, noise_strength.item())
            sample, _ = self.att_model(
                [latent_in_with_noise], input_is_latent=True, truncation=1
            )

            # convert sample to the distribution of normal images
            Min = -1
            Max = 1
            sample.clamp_(min=Min, max=Max)
            sample.add_(-Min).div_(Max - Min + 1e-5)

            sample = self.norm(sample)
            sample_feat = self.trg_model.client_model(sample)

            L_feat = self.train_criterion(sample_feat, trg_feat)
            L_tv = self.tv(sample)
            L_l2 = self.l2(sample)

            optimizer.zero_grad()
            totalLoss = L_feat + self.lambda_TV * L_tv + self.lambda_L2 * L_l2
            totalLoss.backward(retain_graph=True)
            optimizer.step()

            logging.info(
                f"Iter optimize w+: {j}, Feature loss: {L_feat.item()}, TVLoss: {L_tv.item()}, l2Loss: {L_l2.item()}"
            )

        attack_result = self.denorm(sample.detach())
        return attack_result
