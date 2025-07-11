# Copyright xuxiaoyang, ywenrou123@163.com
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

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision

from secretflow import reveal
from secretflow_fl.ml.nn import FLModel
from secretflow_fl.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from secretflow_fl.ml.nn.fl.backend.torch.strategy.fed_gen import (
    FedGen,
    FedGenActor,
    FedGenGeneratorModel,
)
from secretflow_fl.security.aggregation.stateful_fedgen_aggregator import (
    StatefulFedGenAggregator,
)
from secretflow_fl.utils.simulation.datasets_fl import load_mnist


class ConvNet(BaseModule):
    """Small ConvNet for MNIST."""

    def __init__(self, kl_div_loss, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)
        self.kl_div_loss = kl_div_loss
        self.num_classes = num_classes

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx == -1:
            x = self.fc(x)
            return x
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)
        x = self.fc(x)
        return x


class DiversityLoss(nn.Module):
    """
    Custom diversity loss function, designed to encourage diversity in model predictions.
    """

    def __init__(self, metric):
        """
        Initializes the DiversityLoss class.

        Parameters:
        metric (str): The metric for computing distances, can be 'l1', 'l2', or 'cosine'.
        """
        super(
            DiversityLoss, self
        ).__init__()  # Call the parent class's initialization method
        self.metric = metric  # Save the metric
        self.cosine = nn.CosineSimilarity(
            dim=2
        )  # Initialize the cosine similarity computation object, dim=2 indicates similarity is computed along the 2nd dimension

    def compute_distance(self, tensor1, tensor2):
        """
        Computes the distance between two tensors.

        Parameters:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.

        Returns:
        torch.Tensor: The distance between the two tensors.
        """
        if self.metric == 'l1':
            # If the metric is L1 norm, compute the mean of the absolute differences between tensor elements
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif self.metric == 'l2':
            # If the metric is L2 norm, compute the mean of the squared differences between tensor elements
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif self.metric == 'cosine':
            # If the metric is cosine similarity, use cosine similarity to compute the distance between the two tensors
            # Cosine similarity values range from -1 to 1, here we convert it to a distance by 1 - cosine_similarity
            return 1 - self.cosine(tensor1, tensor2)
        else:
            # If the metric is not one of the above three, raise a ValueError exception
            raise ValueError("Unsupported metric: {}".format(self.metric))


class TestFedGen:

    def test_fed_gen_local_step_without_generator_raises_error(
        self, sf_simulation_setup_devices
    ):
        # Set the number of classes
        num_classes = 10

        # Define the loss function and optimizer
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
        kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        diversity_loss = DiversityLoss(metric='l1')

        # Build the model
        builder = TorchModel(
            model_fn=ConvNet,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            kl_div_loss=kl_div_loss,
            num_classes=num_classes,
        )
        fed_gen_worker = FedGen(builder_base=builder)

        # Prepare the test dataset
        x_test = torch.rand(128, 1, 28, 28)  # Randomly generated input data
        y_test = torch.randint(0, 10, (128,))  # Randomly generated labels

        # Create a data loader
        test_loader = DataLoader(
            TensorDataset(x_test, y_test), batch_size=32, shuffle=True
        )
        fed_gen_worker.train_set = iter(test_loader)  # Set the training set
        fed_gen_worker.train_iter = iter(
            fed_gen_worker.train_set
        )  # Initialize the training iterator

        # Execute a training step
        gradients = None  # Gradients of the model parameters
        generator_params = None  # Parameters of the generator
        weights = {"model_params": gradients, "generator_params": generator_params}

        # Attempt to perform a training step; expect to raise AssertionError
        try:
            fed_gen_worker.train_step(weights, cur_steps=0, train_steps=1)
        except AssertionError as e:
            # Assert the error message to ensure the correct error is raised
            assert str(e) == "Generator cannot be none, please define the generator."
        else:
            # If no error is raised, the test fails
            assert False, "ValueError not raised"

    def test_fed_gen_local_step(self, sf_simulation_setup_devices):
        devices = sf_simulation_setup_devices
        # Initialize the FedGen strategy with a ConvNet model
        num_classes = 10
        loss_fn = nn.CrossEntropyLoss  # Loss function for classification
        optim_fn = optim_wrapper(optim.Adam, lr=1e-2)  # Optimizer with learning rate
        kl_div_loss = nn.KLDivLoss(reduction="batchmean")  # KL Divergence loss
        diversity_loss = DiversityLoss(
            metric='l1'
        )  # Diversity loss to promote varied outputs

        # Create a generator model for FedGen
        generator = FedGenGeneratorModel(
            hidden_dimension=256,
            latent_dimension=192,
            noise_dim=64,
            num_classes=num_classes,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            diversity_loss=diversity_loss,
        )

        # Define the main model to be used
        model_def = TorchModel(
            model_fn=ConvNet,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
            ],
            kl_div_loss=kl_div_loss,
            num_classes=num_classes,
        )

        # Set up the FedGen server actor
        server_actor = FedGenActor(device=devices.carol, generator=generator)
        device_list = [devices.alice, devices.bob]

        # Initialize the aggregator for the federated learning process
        aggregator = StatefulFedGenAggregator(devices.carol, device_list, server_actor)

        # Create the federated learning model
        fl_model = FLModel(
            server=devices.carol,
            device_list=device_list,
            model=model_def,
            strategy="fed_gen",
            backend="torch",
            aggregator=aggregator,
            generator=generator,
        )

        # Prepare the dataset
        (_, _), (data, label) = load_mnist(
            parts={devices.alice: 0.4, devices.bob: 0.6},
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )

        # Train the model with the prepared data
        history = fl_model.fit(
            data,
            label,
            validation_data=(data, label),
            epochs=1,
            batch_size=32,
            aggregate_freq=1,
        )

        # Make predictions using the trained model
        result = fl_model.predict(data, batch_size=32)
        assert (
            len(reveal(result[device_list[0]])) == 4000
        )  # Check the number of predictions

        global_metric, _ = fl_model.evaluate(
            data, label, batch_size=32, random_seed=1234
        )
        print(history, global_metric)

        # Assert that the final accuracy matches the recorded history
        assert (
            global_metric[0].result().numpy()
            == history["global_history"]['val_multiclassaccuracy'][-1]
        )
        assert global_metric[0].result().numpy() > 0.1
