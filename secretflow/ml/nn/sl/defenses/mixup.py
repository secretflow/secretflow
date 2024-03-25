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
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss as BaseTorchLoss

from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel


def _get_perm_index(batch_size, seed):
    torch.manual_seed(seed)
    index = torch.randperm(batch_size)
    return index


class Mixuplayer(nn.Module):
    """Implementation of data mixing process in mixup :
        @article{
            zhang2018mixup,
            title={mixup: Beyond Empirical Risk Minimization},
            author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
            journal={International Conference on Learning Representations},
            year={2018},
            url={https://openreview.net/forum?id=r1Ddp1-Rb},
        }
    We empirically found that using mixup during training can defend against gradient-based LIA. It could also be used as a method to improve model accuracy as mentioned in the paper.
    Args:
        param lam: lambda value for mixup. If set to None, it will be sampled from beta distribution.
        alpha: parameter for beta distribution.
        perm_seed: random seed for permutation.
    """

    def __init__(self, lam=0.5, perm_seed=42) -> None:
        super().__init__()
        self.lam = lam
        self.perm_seed = perm_seed

        assert self.lam >= 0 and self.lam <= 1

    def forward(self, x):
        if self.training:
            lam = self.lam
            index = _get_perm_index(batch_size=x.size()[0], seed=self.perm_seed).to(
                device=x.device
            )
            mixed_x = lam * x + (1 - lam) * x[index, :]
            return mixed_x
        else:
            return x


class Mixuploss(BaseTorchLoss):
    """Implementation of loss in mixup:
        @article{
            zhang2018mixup,
            title={mixup: Beyond Empirical Risk Minimization},
            author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
            journal={International Conference on Learning Representations},
            year={2018},
            url={https://openreview.net/forum?id=r1Ddp1-Rb},
        }
    We empirically found that using mixup during training can defend against gradient-based LIA.
    It could also be used as a method to improve model accuracy as mentioned in the paper.
    Args:
        loss_fn: loss function to be wrapped.
        param lam: lambda value for mixup. If set to None, it will be sampled from beta distribution.
        perm_seed: random seed for permutation.
    """

    def __init__(self, loss_fn, lam=None, perm_seed=42) -> None:
        super().__init__()
        self.lam = lam
        self.perm_seed = perm_seed
        self.loss_fn = loss_fn()

        if self.lam is not None:
            assert self.lam >= 0 and self.lam <= 1

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        lam = self.lam
        index = _get_perm_index(batch_size=input.size()[0], seed=self.perm_seed).to(
            device=input.device
        )
        loss = lam * self.loss_fn(input, target) + (1 - lam) * self.loss_fn(
            input, target[index]
        )
        return loss


class MixupDefense(Callback):
    def __init__(self, lam=0.5, perm_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.perm_seed = perm_seed

    @staticmethod
    def inject_mixup(worker: SLBaseTorchModel, lam, perm_seed):
        worker.model_base = (
            worker.builder_base.model_fn(
                **worker.builder_base.kwargs,
                preprocess_layer=Mixuplayer(lam, perm_seed)
            )
            if worker.builder_base and worker.builder_base.model_fn
            else None
        )
        worker.loss_fuse = (
            Mixuploss(worker.builder_fuse.loss_fn, lam, perm_seed)
            if worker.builder_fuse and worker.builder_fuse.loss_fn
            else None
        )

    def on_train_begin(self, logs=None):
        for worker in self._workers.values():
            worker.apply(self.inject_mixup, self.lam, self.perm_seed)
