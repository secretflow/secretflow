import unittest
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall
from secretflow.ml.nn.fl.backend.torch.strategy.scaffold import Scaffold
from secretflow.ml.nn.utils import BaseModule

class My_Model(BaseModule):
    def __init__(self):
        super(My_Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.head = nn.Linear(200, 10)
        self.cg = []
        self.c =[]
        for param in self.parameters():
            self.cg.append(torch.zeros_like(param))
            self.c.append(torch.zeros_like(param))
        self.eta_l = 0.01
    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

class TestFedPer(unittest.TestCase):
    def test_fed_per_local_step(self):
        # Initialize FedPer strategy with ConvNet model
        class ConvNetBuilder:
            def __init__(self):
                # 初始化时，定义一个空的指标列表
                self.metrics = [
                    lambda: Accuracy(task="multiclass", num_classes=10, average='macro')
                ]

            def model_fn(self):
                # 返回 ConvNet 模型的实例
                return My_Model()

            def loss_fn(self):
                # 返回损失函数
                return CrossEntropyLoss()

            def optim_fn(self, parameters):
                # 返回优化器
                return optim.Adam(parameters)

        # 使用 ConvNetBuilder 初始化 FedPer
        conv_net_builder = ConvNetBuilder()
        fed_per_worker = Scaffold(builder_base=conv_net_builder)

        # Prepare dataset
        x_test = torch.rand(128, 1, 28, 28)  # Randomly generated data
        y_test = torch.randint(0, 10, (128,))  # Randomly generated labels for a 10-class task
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=True)
        fed_per_worker.train_set = iter(test_loader)
        fed_per_worker.train_iter = iter(fed_per_worker.train_set)

        # Perform a training step
        gradients = None
        gradients, num_sample = fed_per_worker.train_step(
            gradients, cur_steps=0, train_steps=1
        )

        # Apply weights update
        fed_per_worker.apply_weights(gradients)

        # Assert the sample number and length of gradients
        self.assertEqual(num_sample, 32)  # Batch size
        self.assertEqual(len(gradients), len(list(fed_per_worker.model.parameters())))  # Number of model parameters

        # Perform another training step to test cumulative behavior
        _, num_sample = fed_per_worker.train_step(gradients, cur_steps=1, train_steps=2)
        self.assertEqual(num_sample, 64)  # Cumulative batch size over two steps

if __name__ == '__main__':
    unittest.main()
