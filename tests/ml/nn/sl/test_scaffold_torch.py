import torch
import torch.optim as optim
from secretflow.ml.nn.fl.backend.torch.strategy.scaffold import Scaffold
from secretflow.ml.nn.utils import BaseModule
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall


class My_Model(BaseModule):
    def __init__(self):
        super(My_Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
        )
        self.head = nn.Linear(200, 10)
        self.cg = []
        self.c = []
        for param in self.parameters():
            self.cg.append(torch.zeros_like(param))
            self.c.append(torch.zeros_like(param))
        self.eta_l = 0.01

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


class TestScaffold:
    def test_scaffold_local_step(self, sf_simulation_setup_devices):
        # Initialize Scaffold strategy with ConvNet model
        class ConvNetBuilder:
            def __init__(self):
                # Initialize with an empty indicator list
                self.metrics = [
                    lambda: Accuracy(task="multiclass", num_classes=10, average="macro")
                ]

            def model_fn(self):
                # Return an instance of the ConvNet model
                return My_Model()

            def loss_fn(self):
                # Return the loss
                return CrossEntropyLoss()

            def optim_fn(self, parameters):
                # Return optim
                return optim.Adam(parameters)

        # Initialize Scaffold strategy with ConvNet model
        conv_net_builder = ConvNetBuilder()
        scaffold_worker = Scaffold(builder_base=conv_net_builder)

        # Prepare dataset
        x_test = torch.rand(128, 1, 28, 28)  # Randomly generated data
        y_test = torch.randint(
            0, 10, (128,)
        )  # Randomly generated labels for a 10-class task
        test_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=32, shuffle=True
        )
        scaffold_worker.train_set = iter(test_loader)
        scaffold_worker.train_iter = iter(scaffold_worker.train_set)

        # Perform a training step
        gradients = None
        gradients, num_sample = scaffold_worker.train_step(
            gradients, cur_steps=0, train_steps=1
        )

        # Apply weights update
        scaffold_worker.apply_weights(gradients)

        # Assert the sample number and length of gradients
        assert num_sample == 32  # Batch size
        assert len(gradients) == len(list(scaffold_worker.model.parameters()))  # Number of model parameters

        # Perform another training step to test cumulative behavior
        _, num_sample = scaffold_worker.train_step(
            gradients, cur_steps=1, train_steps=2
        )
        assert num_sample == 64  # Cumulative batch size over two steps

