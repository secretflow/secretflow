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


import torch
from torch import nn as nn

from secretflow.security.privacy.accounting.budget_accountant import BudgetAccountant


class GaussianEmbeddingDP(nn.Module, BudgetAccountant):
    def __init__(
        self,
        noise_multiplier: float,
        batch_size: int,
        num_samples: int,
        l2_norm_clip: float = 1.0,
        delta: float = None,
        is_secure_generator: bool = False,
    ) -> None:
        super().__init__()
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.delta = delta if delta is not None else min(1 / num_samples**2, 1e-5)
        self.is_secure_generator = is_secure_generator

    def forward(self, input):
        norm_vec = torch.norm(input, p=2, dim=-1)
        ones = torch.ones(size=norm_vec.shape)
        max_v = torch.diag(1.0 / torch.maximum(norm_vec / self.l2_norm_clip, ones))
        embed_clipped = torch.matmul(max_v, input)

        # add noise
        if self.is_secure_generator:
            import secretflow.security.privacy._lib.random as random

            noise = random.secure_normal_real(
                0, self.noise_multiplier * self.l2_norm_clip, size=input.shape
            )
        else:
            noise = torch.normal(
                0.0,
                self.noise_multiplier * self.l2_norm_clip,
                size=input.shape,
            )
        dp_h = torch.add(embed_clipped, noise)
        return dp_h
