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
from typing import List

import torch

from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy


class SLTorchModel(SLBaseTorchModel):
    def base_forward(self, stage="train"):
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
        Returns: hidden embedding
        """

        assert (
            self.model_base is not None
        ), "Base model cannot be none, please give model define or load a trained model"

        data_x = self.get_batch_data(stage=stage)
        self.h = self.base_forward_internal(
            data_x,
        )
        # The compressor in forward can only recognize np type but not tensor.
        return self.h.detach().numpy() if isinstance(self.h, torch.Tensor) else self.h

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

    def fuse_net(self, hidden_features, _num_returns=2):
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

        hiddens = []
        for h in hidden_features:
            # h will be list, if basenet is multi output
            if isinstance(h, List):
                for i in range(len(h)):
                    hiddens.append(torch.tensor(h[i]))
            else:
                hiddens.append(torch.tensor(h))

        logs = {}
        gradient = self.fuse_net_internal(
            hiddens,
            self.train_y,
            self.train_sample_weight,
            logs,
        )
        for m in self.metrics_fuse:
            logs['train_' + m.__class__.__name__] = m.compute()
        self.logs = copy.deepcopy(logs)

        return gradient


@register_strategy(strategy_name='split_nn', backend='torch')
@proxy(PYUObject)
class PYUSLTorchModel(SLTorchModel):
    pass
