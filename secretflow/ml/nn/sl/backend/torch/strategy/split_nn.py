#!/usr/bin/env python3
# *_* coding: utf-8 *_*

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


""" Async split learning strategy

"""
import copy
from typing import List, Union, Optional

import torch

from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.utils.communicate import ForwardData


class SLTorchModel(SLBaseTorchModel):
    def base_forward(self, stage="train", step=0, **kwargs) -> Optional[ForwardData]:
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
        Returns: hidden embedding
        """

        if step == 0:
            self._reset_data_iter(stage=stage)
        data_x = self.get_batch_data(stage=stage)
        if not self.model_base:
            return None
        self.h = self.base_forward_internal(
            data_x,
        )
        forward_data = ForwardData()

        # The compressor can only recognize np type but not tensor.
        forward_data.hidden = (
            self.h.detach().numpy() if isinstance(self.h, torch.Tensor) else self.h
        )
        # The compressor in forward can only recognize np type but not tensor.
        return forward_data

    def base_backward(self, gradient):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
        """

        return_hiddens = []

        if len(gradient) == len(self.h):
            for i in range(len(gradient)):
                return_hiddens.append(self.fuse_op.apply(self.h[i], gradient[i]))
        else:
            gradient = (
                gradient[0]
                if isinstance(gradient[0], torch.Tensor)
                else torch.tensor(gradient[0])
            )
            return_hiddens.append(self.fuse_op.apply(self.h, gradient))

        # apply gradients for base net
        self.optim_base.zero_grad()
        for rh in return_hiddens:
            if rh.requires_grad:
                rh.sum().backward(retain_graph=True)
        self.optim_base.step()

        # clear intermediate results
        self.tape = None
        self.h = None
        self.kwargs = {}

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
            gradient Of hiddens
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

        train_y = self.train_y[0] if len(self.train_y) == 1 else self.train_y

        logs = {}
        gradient = self.fuse_net_internal(
            hiddens,
            train_y,
            self.train_sample_weight,
            logs,
        )
        for m in self.metrics_fuse:
            logs['train_' + m.__class__.__name__] = m.compute().numpy()
        self.logs = copy.deepcopy(logs)

        return gradient


@register_strategy(strategy_name='split_nn', backend='torch')
@proxy(PYUObject)
class PYUSLTorchModel(SLTorchModel):
    pass
