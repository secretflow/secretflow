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
from typing import Callable, List, Optional, Union

import torch

from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.ml.nn.utils import TorchModel
from secretflow.security.privacy import DPStrategy
from secretflow.utils.communicate import ForwardData


class PipelineTorchModel(SLBaseTorchModel):
    def __init__(
        self,
        builder_base: Callable[[], TorchModel],
        builder_fuse: Callable[[], TorchModel],
        dp_strategy: DPStrategy,
        random_seed: int = None,
        pipeline_size: int = 1,
        *args,
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
        self.model_base_copy = (
            builder_base.model_fn(**builder_base.kwargs)
            if builder_base and builder_base.model_fn
            else None
        )

        self.pre_param_list = []
        self.hidden_list = []
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

            self._h = self.model_base_copy(self._data_x)
            self.hidden_list.append(self._h)
        else:
            self._h = self.model_base(self._data_x)

    def base_backward(self):
        """backward on fusenet"""

        return_hiddens = []

        hid = self.hidden_list.pop(0)
        pre_param = self.pre_param_list.pop(0)
        if len(self._gradient) == len(hid):
            for i in range(len(self._gradient)):
                return_hiddens.append(self.fuse_op.apply(hid[i], self._gradient[i]))
        else:
            self._gradient = (
                self._gradient[0]
                if isinstance(self._gradient[0], torch.Tensor)
                else torch.tensor(self._gradient[0])
            )
            return_hiddens.append(self.fuse_op.apply(hid, self._gradient))

        # apply gradients for base net
        self.optim_base.zero_grad()
        for rh in return_hiddens:
            if rh.requires_grad:
                rh.sum().backward(retain_graph=True)

        # copy gradient from pre_param -> model's param
        for name, param in self.model_base.named_parameters():
            param.grad = pre_param[name].grad

        self.optim_base.step()

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
        if isinstance(forward_data, ForwardData):
            forward_data = [forward_data]
        forward_data[:] = (h for h in forward_data if h is not None)
        for i, h in enumerate(forward_data):
            assert h.hidden is not None, f"hidden cannot be found in forward_data[{i}]"
            if isinstance(h.losses, List) and h.losses[0] is None:
                h.losses = None

        hidden_features = [h.hidden for h in forward_data]

        hiddens = []
        for h in hidden_features:
            # h will be list, if basenet is multi output
            if isinstance(h, List):
                for i in range(len(h)):
                    hiddens.append(torch.tensor(h[i]))
            else:
                hiddens.append(torch.tensor(h))

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
        for m in self.metrics_fuse:
            logs['train_' + m.__class__.__name__] = m.compute().numpy()
        self.logs = logs

        return gradient


@register_strategy(strategy_name='pipeline', backend='torch')
class PYUPipelineTorchModel(PipelineTorchModel):
    pass
