# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod

import torch


class BaseClient(torch.nn.Module):
    """Abstract class foe the client of collaborative learning.

    Args:
        model (torch.nn.Module): a local model
        user_id (int, optional): id of this client. Defaults to 0.
    """

    def __init__(self, model, user_id=0):
        """Initialize BaseClient"""
        super(BaseClient, self).__init__()
        self.model = model
        self.user_id = user_id

    def forward(self, x):
        return self.model(x)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def backward(self, loss):
        """Execute backward mode automatic differentiation with the give loss.

        Args:
            loss (torch.Tensor): the value of calculated loss.
        """
        loss.backward()

    @abstractmethod
    def upload(self):
        """Upload the locally learned informatino to the server."""
        pass

    @abstractmethod
    def download(self):
        """Download the global model from the server."""
        pass

    @abstractmethod
    def local_train(self):
        pass
