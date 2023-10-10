from abc import abstractmethod

import torch


class BaseServer(torch.nn.Module):
    """Abstract class for the server of the collaborative learning.

    Args:
        clients (List[BaseClient]): a list of clients
        server_model (torch.nn.Module): a global model
        server_id (int, optional): the id of this server. Defaults to 0.
    """

    def __init__(self, clients, server_model, server_id=0):
        """Initialie BaseServer"""
        super(BaseServer, self).__init__()
        self.clients = clients
        self.server_id = server_id
        self.server_model = server_model
        self.num_clients = len(clients)

    def forward(self, x):
        return self.server_model(x)

    def train(self):
        self.server_model.train()

    def eval(self):
        self.server_model.eval()

    @abstractmethod
    def action(self):
        """Execute thr routine of each communication."""
        pass

    @abstractmethod
    def update(self):
        """Update the global model."""
        pass

    @abstractmethod
    def distribute(self):
        """Distribute the global model to each client."""
        pass
