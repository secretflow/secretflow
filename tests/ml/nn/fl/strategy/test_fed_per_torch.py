import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall
from secretflow.ml.nn.fl.backend.torch.strategy.fed_per import FedPer
from tests.ml.nn.fl.model_def import ConvNet


class TestFedPer:
    def test_fed_per_local_step(self, sf_simulation_setup_devices):
        class ConvNetBuilder:
            def __init__(self):
                self.metrics = [
                    lambda: Accuracy(task="multiclass", num_classes=10, average='macro')
                ]

            def model_fn(self):
                return ConvNet()

            def loss_fn(self):
                return CrossEntropyLoss()

            def optim_fn(self, parameters):
                return optim.Adam(parameters)

        # Initialize FedPer strategy with ConvNet model
        conv_net_builder = ConvNetBuilder()
        fed_per_worker = FedPer(builder_base=conv_net_builder)

        # Prepare dataset
        x_test = torch.rand(128, 1, 28, 28)  # Randomly generated data
        y_test = torch.randint(
            0, 10, (128,)
        )  # Randomly generated labels for a 10-class task
        test_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=32, shuffle=True
        )
        fed_per_worker.train_set = iter(test_loader)
        fed_per_worker.train_iter = iter(fed_per_worker.train_set)

        # Perform a training step
        weights = None
        weights, num_sample = fed_per_worker.train_step(
            weights, cur_steps=0, train_steps=1
        )

        # Apply weights update
        fed_per_worker.apply_weights(weights)

        # Assert the sample number and length of weights
        assert num_sample == 32  # Batch size
        assert len(weights) == len(
            list(fed_per_worker.model.parameters())
        )  # Number of model parameters

        # Perform another training step to test cumulative behavior
        _, num_sample = fed_per_worker.train_step(weights, cur_steps=1, train_steps=2)
        assert num_sample == 64  # Cumulative batch size over two steps
