# Copyright 2024 Ant Group Co., Ltd.
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


""" Pipeline split learning strategy

"""
from typing import List, Optional, Union

import torch

from secretflow.ml.nn.core.torch import BuilderType, module
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.communicate import ForwardData


class PipelineTorchModel(SLBaseTorchModel):
    def __init__(
        self,
        builder_base: BuilderType,
        builder_fuse: BuilderType,
        dp_strategy: DPStrategy,
        random_seed: int = None,
        pipeline_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            builder_base,
            builder_fuse,
            dp_strategy,
            random_seed,
            **kwargs,
        )
        self.pipeline_size = pipeline_size

        # for backward to get gradient
        self.model_base_copy = module.build(builder_base, self.exec_device)

        self.pre_param_list = []
        self.hidden_list = []
        self.base_loss_list = []
        self.pre_train_y = []
        self.pre_sample_weight = []

    def get_batch_data(self, stage="train", epoch=1):
        super().get_batch_data(stage, epoch)
        if stage == 'train' and self.model_fuse is not None:
            self.pre_train_y.append(self.train_y)
            self.pre_sample_weight.append(self.train_sample_weight)

    def reset_data_iter(self, stage):
        super().reset_data_iter(stage)
        if stage == 'train':
            assert (
                self.hidden_list == []
            ), f'hidden_list is not empty {len(self.hidden_list)}'
            assert (
                self.base_loss_list == []
            ), f'base_loss_list is not empty {len(self.base_loss_list)}'
            assert (
                self.pre_train_y == []
            ), f'pre_train_y is not empty {len(self.pre_train_y)}'
            assert (
                self.pre_param_list == []
            ), f'pre_param_list is not empty {len(self.pre_param_list)}'
            assert (
                self.pre_sample_weight == []
            ), f'pre_sample_weight is not empty {len(self.pre_sample_weight)}'

    def base_forward(self) -> Optional[ForwardData]:
        """compute hidden embedding"""

        if not self.model_base:
            return None
        # TODO(caibei): dp not support now

        if self._training:
            # copy model's param -> pre_param
            pre_param = {
                name: torch.nn.Parameter(
                    param.clone(), requires_grad=param.requires_grad
                )
                for name, param in self.model_base.named_parameters()
            }
            self.pre_param_list.append(pre_param)

            # set model_base_copy's param, param._version = 0
            for name, param in pre_param.items():
                layer_name = name.split('.')[:-1]
                param_name = name.split('.')[-1]
                layer = getattr(self.model_base_copy, layer_name[0])
                for ll in layer_name[1:]:
                    layer = getattr(layer, ll)
                setattr(layer, param_name, param)

            self._h = self.base_forward_internal(
                self._data_x, model=self.model_base_copy
            )
            self.hidden_list.append(self._h)
            self.base_loss_list.append(self._base_losses)
        else:
            self._h = self.base_forward_internal(self._data_x)

    def base_backward(self):
        """backward on fusenet"""

        hiddens = self.hidden_list.pop(0)
        base_losses = self.base_loss_list.pop(0)
        return_hiddens = self.base_backward_hidden_internal(
            hiddens, base_losses=base_losses
        )

        # apply gradients for base net
        optimizer = self.model_base.optimizers()
        optimizer.zero_grad()
        for rh in return_hiddens:
            rh.backward(retain_graph=True)

        pre_param = self.pre_param_list.pop(0)
        # copy gradient from pre_param -> model's param
        for name, param in self.model_base.named_parameters():
            param.grad = pre_param[name].grad

        optimizer.step()

        self._h = None

    def fuse_net(
        self,
        forward_data: Union[List[ForwardData], ForwardData],
        _num_returns=2,
    ):
        """Fuses the hidden layer and calculates the reverse gradient
        only on the side with the label

        Args:
            hidden_features: A list of hidden layers for each party to compute
        Returns:
            gradient of hiddens
        """

        assert (
            self.model_fuse is not None
        ), "Fuse model cannot be none, please give model define"

        hiddens = self.unpack_forward_data(forward_data)

        train_y = self.pre_train_y.pop(0)
        train_y = train_y[0] if len(train_y) == 1 else train_y
        train_sample_weight = self.pre_sample_weight.pop(0)

        logs = {}
        gradient = self.fuse_net_internal(
            hiddens,
            train_y,
            train_sample_weight,
            logs,
        )
        for m in self.model_fuse.metrics:
            logs['train_' + m.__class__.__name__] = m.compute().cpu().numpy()
        self.logs = logs
        self._gradient = gradient


@register_strategy(strategy_name='pipeline', backend='torch')
class PYUPipelineTorchModel(PipelineTorchModel):
    pass
