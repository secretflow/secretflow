from ..model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from .ResNet import ResNet18, ResNet18NoNorm


class Conv2Cifar(Model):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 5 * 5, 384)
        self.linear2 = nn.Linear(384, 192)
        # intentionally remove the bias term for the last linear layer for fair comparison
        self.prototype = nn.Linear(192, config['num_classes'], bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return x, logits


class Conv2CifarNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 5 * 5, 384)
        self.linear2 = nn.Linear(384, 192)
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear1(x))
        feature_embedding = F.relu(self.linear2(x))
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


class ResNetMod(Model):
    def __init__(self, config):
        super().__init__(config)
        if config['no_norm']:
            self.backbone = ResNet18NoNorm(num_classes=config['num_classes'])
        else:
            self.backbone = ResNet18(num_classes=config['num_classes'])
        self.prototype = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False)
        self.backbone.linear = None

    def forward(self, x):
        # Convolution layers
        feature_embedding = self.backbone(x)
        logits = self.prototype(feature_embedding)
        return logits

    def get_embedding(self, x):
        feature_embedding = self.backbone(x)
        logits = self.prototype(feature_embedding)
        return feature_embedding, logits


class ResNetModNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        if config['no_norm']:
            self.backbone = ResNet18NoNorm(num_classes=config['num_classes'])
        else:
            self.backbone = ResNet18(num_classes=config['num_classes'])
        temp = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.backbone.linear = None
        self.scaling = torch.nn.Parameter(torch.tensor([20.0]))
        self.activation = None

    def forward(self, x):
        feature_embedding = self.backbone(x)
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits
        self.activation = self.backbone.activation
        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits
