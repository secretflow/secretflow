
from ..model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(Model):
    def __init__(self, config):
        dim = config['dim']
        num_cls = config['num_classes']
        W = None
        self.normalize = config['normalize']
        self.return_embedding = config['return_embedding']
        super().__init__(config)
        self.fc1 = nn.Linear(2, dim * 8)
        self.fc2 = nn.Linear(dim * 8, dim * 4)
        self.fc3 = nn.Linear(dim * 4, dim * 2)
        self.fc4 = nn.Linear(dim * 2, dim)
        if W is None:
            temp = nn.Linear(dim, num_cls, bias=False).state_dict()['weight']
            self.prototype = nn.Parameter(temp)
        else:
            self.prototype = W

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        feature_embedding = self.fc4(x)
        if self.normalize:
            feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
            logits = torch.matmul(feature_embedding, normalized_prototype.T)
        else:
            logits = torch.matmul(feature_embedding, self.prototype.T)

        if self.return_embedding:
            return feature_embedding, logits
        return logits


class MLPNH(Model):
    def __init__(self, config):
        dim = config['dim']
        num_cls = config['num_classes']
        W = None
        self.normalize = config['normalize']
        self.return_embedding = config['return_embedding']
        super().__init__(config)
        self.fc1 = nn.Linear(2, dim * 8)
        self.fc2 = nn.Linear(dim * 8, dim * 4)
        self.fc3 = nn.Linear(dim * 4, dim * 2)
        self.fc4 = nn.Linear(dim * 2, dim)
        if W is None:
            temp = nn.Linear(dim, num_cls, bias=False).state_dict()['weight']
            self.prototype = nn.Parameter(temp)
        else:
            self.prototype = W

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        feature_embedding = self.fc4(x)
        if self.normalize:
            feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
            logits = torch.matmul(feature_embedding, normalized_prototype.T)
        else:
            logits = torch.matmul(feature_embedding, self.prototype.T)

        if self.return_embedding:
            return feature_embedding, logits
        return logits


"""
Ref: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""

# class MLP(Model):
#     def __init__(self, config):
#         super().__init__(config, None, None)
#         dim = config['dim']
#         self.fc1 = nn.Linear(2, 64)
#         self.fc2 = nn.Linear(64, dim * 4)
#         self.fc3 = nn.Linear(dim * 4, dim * 2)
#         self.feature_embedding = nn.Linear(dim * 2, dim)

#     def forward(self, x, normalize=False):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         embedding = self.feature_embedding(x)
#         if normalize:
#             normalized_embedding = embedding / torch.norm(embedding, dim=1).view(-1,1)
#             return normalized_embedding
#         else:
#             return embedding


# class MLPCE(Model):
#     def __init__(self, config):
#         super().__init__(config, None, None)
#         dim = config['dim']
#         self.fc1 = nn.Linear(2, 64)
#         self.fc2 = nn.Linear(64, dim * 4)
#         self.fc3 = nn.Linear(dim * 4, dim * 2)
#         self.W = nn.Linear(dim * 2, 2)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         logits = self.W(x)
#         return logits

# class MLP(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.fc1 = nn.Linear(2, 64)
#         self.fc2 = nn.Linear(64, dim * 4)
#         self.fc3 = nn.Linear(dim * 4, dim * 2)
#         self.feature_embedding = nn.Linear(dim * 2, dim)

#     def forward(self, x, normalize=False):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         embedding = self.feature_embedding(x)
#         if normalize:
#             normalized_embedding = embedding / torch.norm(embedding, dim=1).view(-1,1)
#             return normalized_embedding
#         else:
#             return embedding


# class MLPCE(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.fc1 = nn.Linear(2, 64)
#         self.fc2 = nn.Linear(64, dim * 4)
#         self.fc3 = nn.Linear(dim * 4, dim * 2)
#         self.W = nn.Linear(dim * 2, 2)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         logits = self.W(x)
#         return logits
