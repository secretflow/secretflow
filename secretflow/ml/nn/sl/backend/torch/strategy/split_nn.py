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
from typing import List, Union

from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.utils.communicate import ForwardData


class SLTorchModel(SLBaseTorchModel):
    def base_forward(self, stage: str = 'train', **kwargs):
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
        Returns: hidden embedding
        """
        if not self.model_base:
            return None
        self._h = self.base_forward_internal(
            self._data_x,
        )

    def base_backward(self):
        """backward on fusenet

        Args:
            self.gradient: gradient of fusenet hidden layer
        """
        return_hiddens = self.base_backward_hidden_internal(self._h)

        # apply gradients for base net
        self.model_base.backward_step(return_hiddens)

        # clear intermediate results
        self._h = None
        self._base_losses = None
        self.kwargs = {}

    def fuse_net(
        self,
        forward_data: Union[List[ForwardData], ForwardData],
        _num_returns=2,
    ):
        """Fuses the hidden layer and calculates the reverse gradient
        only on the side with the label

        Args:
            forward_data: A list of hidden layers for each party to compute.
            _num_returns: the return nums.
        Returns:
            gradient Of hiddens
        """
        assert (
            self.model_fuse is not None
        ), "Fuse model cannot be none, please give model define"

        hiddens = self.unpack_forward_data(forward_data)
        train_y = self.train_y[0] if len(self.train_y) == 1 else self.train_y

        logs = {}
        gradient = self.fuse_net_internal(
            hiddens,
            train_y,
            self.train_sample_weight,
            logs,
        )
        for m in self.model_fuse.metrics:
            logs['train_' + m.__class__.__name__] = m.compute().cpu().numpy()
        self.logs = copy.deepcopy(logs)
        self._gradient = gradient


@register_strategy(strategy_name='split_nn', backend='torch')
class PYUSLTorchModel(SLTorchModel):
    pass
