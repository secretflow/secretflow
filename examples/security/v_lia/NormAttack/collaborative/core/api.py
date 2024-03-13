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

import copy
import os
import sys
from abc import abstractmethod

from utils import accuracy_torch_dataloader


class BaseFedAPI:
    """Abstract class for Federated Learning API"""

    @abstractmethod
    def local_train(self):
        pass

    @abstractmethod
    def run(self):
        pass


class BaseFLKnowledgeDistillationAPI:
    """Abstract class for API of federated learning with knowledge distillation.

    Args:
        server (aijack.collaborative.core.BaseServer): the server
        clients (List[aijack.collaborative.core.BaseClient]): a list of the clients
        public_dataloader (torch.utils.data.DataLoader): a dataloader for the public dataset
        local_dataloaders (List[torch.utils.data.DataLoader]): a list of local dataloaders
        validation_dataloader (torch.utils.data.DataLoader): a dataloader for the validation dataset
        criterion (function): a function to calculate the loss
        num_communication (int): the number of communication
        device (str): device type
    """

    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        validation_dataloader,
        criterion,
        num_communication,
        device,
    ):
        """Initialize BaseFLKnowledgeDistillationAPI"""
        self.server = server
        self.clients = clients
        self.public_dataloader = public_dataloader
        self.local_dataloaders = local_dataloaders
        self.validation_dataloader = validation_dataloader
        self.criterion = criterion
        self.num_communication = num_communication
        self.device = device

        self.client_num = len(clients)

    def train_client(self, epoch=1, public=True):
        """Train local models with the local datasets or the public dataset.

        Args:
            public (bool, optional): Train with the public dataset or the local datasets.
                                     Defaults to True.

        Returns:
            List[float]: a list of average loss of each clients.
        """
        loss_on_local_dataest = []
        for client_idx in range(self.client_num):
            if public:
                trainloader = self.public_dataloader
            else:
                trainloader = self.local_dataloaders[client_idx]

            running_loss = self.clients[client_idx].local_train(
                epoch, self.criterion, trainloader, self.client_optimizers[client_idx]
            )

            loss_on_local_dataest.append(copy.deepcopy(running_loss / len(trainloader)))

        return loss_on_local_dataest

    @abstractmethod
    def run(self):
        pass

    def score(self, dataloader, only_local=False):
        """Returns the performance on the given dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): a dataloader
            only_local (bool): show only the local results

        Returns:
            Dict[str, int]: performance of global model and local models
        """

        clients_score = [
            accuracy_torch_dataloader(client, dataloader, device=self.device)
            for client in self.clients
        ]

        if only_local:
            return {"clients_score": clients_score}
        else:
            server_score = accuracy_torch_dataloader(
                self.server, dataloader, device=self.device
            )
            return {"server_score": server_score, "clients_score": clients_score}

    def local_score(self):
        """Returns the local performance of each clients.

        Returns:
            Dict[str, int]: performance of global model and local models
        """
        local_score_list = []
        for client, local_dataloader in zip(self.clients, self.local_dataloaders):
            temp_score = accuracy_torch_dataloader(
                client, local_dataloader, device=self.device
            )
            local_score_list.append(temp_score)

        return {"clients_score": local_score_list}
