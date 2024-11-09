"""
客户端定义
"""
import os

from model.cifar10 import resnet18 as CifarResNet

import logging
import torch.utils
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from secretflow import PYUObject, proxy

#获取当前脚本的绝对目录
current_dir = os.path.dirname(os.path.abspath(__file__))

@proxy(PYUObject)
class Client:
    def __init__(self, client_id, dev, train_index, test_index, batch_size, inner_epoch):
        self.client_id = client_id
        self.device = dev
        self.batch_size = batch_size
        self.inner_epoch = inner_epoch
        self.model = CifarResNet().to(dev)
        self.criterion = nn.CrossEntropyLoss().to(dev)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        self.train_index = train_index
        self.test_index = test_index
        self.train_loader = None
        self.test_loader = None
        self.train_subset = None
        self.test_subset = None
        self.train_accuracy = 0
        self.test_accuracy = 0

    def load_dataset(self):
        logging.warning("start loading dataset")
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
        train_dataset = datasets.CIFAR10(current_dir+ '/../dataset', train=True, download=True,
                                        transform=transform)
        test_dataset = datasets.CIFAR10( current_dir+ '/../dataset', train=False, download=True,
                                        transform=transform)
        self.train_subset = torch.utils.data.Subset(train_dataset, indices=self.train_index)
        self.test_subset = torch.utils.data.Subset(test_dataset, indices=self.test_index)
        self.train_loader = torch.utils.data.DataLoader(self.train_subset, batch_size=self.batch_size)
        self.test_loader = torch.utils.data.DataLoader(self.test_subset, batch_size=self.batch_size)
        logging.warning(f"Client {self.client_id} dataset loaded")

    def train(self):
        logging.warning(f"Client {self.client_id} start training")
        self.model.train()
        for i in range(self.inner_epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                logit, _ = self.model(data, mode='local')
                loss = self.criterion(logit, target)
                loss.backward()
                self.optimizer.step()

        for i in range(self.inner_epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                for n, p in self.model.named_parameters():
                    if 'local' in n or 'global' in n:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                self.optimizer.zero_grad()
                logit, prob = self.model(data, mode='personalized')
                loss = self.criterion(logit, target) + 0.001 * (torch.mean(-torch.log(prob[:, 1, :] ** 2 + 1e-6)))
                loss.backward()
                self.optimizer.step()

                for n, p in self.model.named_parameters():
                    if 'local' in n or 'prob' in n:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                self.optimizer.zero_grad()
                logit, _ = self.model(data, mode='global')
                loss = self.criterion(logit, target)
                loss.backward()
                self.optimizer.step()


    # 由于该算法的特殊性，不能在训练阶段计算出准确率，所以测试集的准确率和训练集的准确率一起进行计算
    def test(self):
        train_correct, train_total, train_loss = 0, 0, 0.0
        test_correct, test_total, test_loss = 0, 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                logit, _ = self.model(data, mode='personalized', hard_decision=True)
                train_loss += self.criterion(logit, target).item()
                _, predicted = torch.max(logit, 1)
                train_total += target.shape[0]
                train_correct += (predicted == target).sum().item()
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                logit, _ = self.model(data, mode='personalized', hard_decision=True)
                test_loss += self.criterion(logit, target).item()
                _, predicted = torch.max(logit, 1)
                test_total += target.shape[0]
                test_correct += (predicted == target).sum().item()
        self.train_accuracy = train_correct / train_total
        self.test_accuracy = test_correct / test_total


    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_train_accuracy(self):
        return self.train_accuracy

    def get_test_accuracy(self):
        return self.test_accuracy