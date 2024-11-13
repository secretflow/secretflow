# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Callable, List, Union

import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torchmetrics import Metric
from typing_extensions import TypeAlias, override

from .mixins import ParametersMixin


class BaseModule(ParametersMixin, nn.Module):
    """Lightning style base class for your torch neural network models.

    You can define your model by subclassing this class and define your forward pass:

    .. code-block:: python

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(BaseModule):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    And if you have a complex training step, you can write it yourself:

    .. code-block:: python

        class AutoEncoder(BaseModule):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
                self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

            def forward(self, x):
                embedding = self.encoder(x)
                return embedding

            def training_step(self, batch, batch_idx):
                x, y = batch
                x = x.view(x.size(0), -1)
                z = self.encoder(x)
                x_hat = self.decoder(z)
                loss = F.mse_loss(x_hat, x)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters())

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.automatic_optimization = True
        self.loss = None
        self.metrics: List[Metric] = []
        self.logs = {}

    @override
    def forward(self, *input: Any, **kwargs: Any) -> Any:
        """Same as :meth:`torch.nn.Module.forward`.

        Args:
            *input: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output

        """
        return super().forward(*input, **kwargs)

    def configure_optimizers(self) -> Union[optim.Optimizer, List[optim.Optimizer]]:
        """Choose what optimizers to use in your optimization. Normally you'd need one.
        But in the case of GANs or similar you might have multiple.
        Optimization with multiple optimizers only works in the manual optimization mode.

        Return:
            - **Single optimizer**.
            - **List or Tuple** of optimizers.
        """
        pass

    def configure_metrics(self) -> List[Metric]:
        """Choose what metrics to use in your training_step or validation_step.

        Returns:
            List[Metric]: List of Metric.
        """
        pass

    def configure_loss(self) -> _Loss:
        """Choose what loss to use in your training_step or validation_step.

        Returns:
            _Loss: torch builtin Losses or custom Loss.
        """
        pass

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0, **kwargs
    ):
        """Here you compute and return the training loss.

        Args:
            batch: The batch data for training.
            batch_idx: The index of this batch. not supported in SLModel for now.
            dataloader_idx: The index of the dataloader that produced this batch. not supported for now.
            sample_weight: Sample weight of the dataset.

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``None`` - In automatic optimization, this will skip to the next batch.
                For manual optimization, this has no special meaning, as returning
                the loss is not required.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                x, y = batch
                out = self.encoder(x)
                loss = self.loss(out, y)
                return loss

        To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

        .. code-block:: python

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()

        """
        _, loss = self.forward_step(batch, batch_idx, dataloader_idx)

        if self.automatic_optimization:
            return loss
        else:
            return None

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0, **kwargs
    ):
        """Operates on a single batch of data from the validation set.
        Same as training_step by default in automatic optimization.

        Args:
            batch: The batch data for training.
            batch_idx: The index of this batch. not supported in SLModel for now.
            dataloader_idx: The index of the dataloader that produced this batch. not supported for now.
            sample_weight: Sample weight of the dataset.

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``None`` - Skip to the next batch.
        """
        with torch.no_grad():
            if self.automatic_optimization:
                return self.training_step(batch, batch_idx, dataloader_idx, **kwargs)
            _, loss = self.forward_step(batch, batch_idx, dataloader_idx)
            return loss

    def forward_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        y_pred = self(x)

        self.update_metrics(y_pred, y)

        if self.loss:
            loss = self.loss(y_pred, y)
            return y_pred, loss
        else:
            return y_pred, None

    def backward_step(self, loss: Union[torch.Tensor, List[torch.Tensor]]):
        optimizer = self.optimizers()
        assert isinstance(
            optimizer, optim.Optimizer
        ), "Only one optimizer is allowed in automatic optimization"

        optimizer.zero_grad()
        self.backward(loss)
        optimizer.step()

    def backward(
        self, loss: Union[torch.Tensor, List[torch.Tensor]], *args: Any, **kwargs: Any
    ) -> None:
        if isinstance(loss, torch.Tensor):
            loss.backward()
        elif isinstance(loss, List):
            for idx, l in enumerate(loss):
                if idx < len(loss) - 1:
                    l.backward(retain_graph=True)
                else:
                    l.backward()
        else:
            raise TypeError(f"unsupported loss type {type(loss)}")

    def update_metrics(self, y_pred, y_true):
        for m in self.metrics:
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                m.update(y_pred, y_true.int().argmax(-1))
            else:
                m.update(y_pred, y_true.int())


class TorchModel:
    def __init__(
        self,
        model_fn: Callable[..., Union[BaseModule, nn.Module]] = None,
        loss_fn: Callable[..., _Loss] = None,
        optim_fn: Callable[..., optim.Optimizer] = None,
        metrics: List[Callable[..., Metric]] = [],
        **kwargs,
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.metrics = metrics
        self.kwargs = kwargs

    def build(self, device: torch.device = None) -> BaseModule:
        if self.model_fn is None:
            return None
        model = self.model_fn(**self.kwargs)
        return self.build_from_model(model, device)

    def _configure_warning(self, used_conf, no_use_conf):
        logging.warning(
            f"Both Module.{no_use_conf}() and TorchModel.{used_conf} are defined. Only TorchModel.{used_conf} will be used."
        )

    def build_from_model(
        self, model: nn.Module, device: torch.device = None
    ) -> BaseModule:
        assert isinstance(model, nn.Module)

        if device is None:
            device = torch.device("cpu")

        optimizer = self.optim_fn(model.parameters()) if self.optim_fn else None
        loss = self.loss_fn() if self.loss_fn else None
        metrics = [m().to(device) for m in self.metrics] if self.metrics else []

        # use object.__setattr__ instead of the typical model.a = a to avoid Module.__setattr__ overhead.
        if not hasattr(model, "_optimizers"):
            object.__setattr__(model, "_optimizers", [])
        if not hasattr(model, "loss"):
            object.__setattr__(model, "loss", None)
        if not hasattr(model, "metrics"):
            object.__setattr__(model, "metrics", [])

        if optimizer:
            object.__setattr__(model, "_optimizers", [optimizer])
        if loss:
            object.__setattr__(model, "loss", loss)
        if metrics:
            object.__setattr__(model, "metrics", metrics)
        object.__setattr__(model, "logs", {})

        if isinstance(model, BaseModule):
            configured_optimizers = model.configure_optimizers()
            if configured_optimizers:
                if optimizer:
                    self._configure_warning("optim_fn", "configure_optimizers")
                else:
                    if isinstance(configured_optimizers, List):
                        object.__setattr__(model, "_optimizers", configured_optimizers)
                    elif isinstance(configured_optimizers, optim.Optimizer):
                        object.__setattr__(
                            model, "_optimizers", [configured_optimizers]
                        )
            configured_metrics = model.configure_metrics()
            if configured_metrics:
                if metrics:
                    self._configure_warning("metrics", "configure_metrics")
                else:
                    configured_metrics = [m for m in configured_metrics]
                    object.__setattr__(model, "metrics", configured_metrics)
            configured_loss = model.configure_loss()
            if configured_loss:
                if loss:
                    self._configure_warning("loss_fn", "configure_loss")
                else:
                    object.__setattr__(model, "loss", configured_loss)
        else:
            # mix Module in user defined nn.Module
            object.__setattr__(model, "automatic_optimization", True)
            model.__class__ = type(
                'GeneratedTorchModule', (BaseModule, model.__class__), {}
            )

        model.to(device)
        for m in model.metrics:
            m.to(device)

        return model


BuilderType: TypeAlias = Union[TorchModel, Callable[[], BaseModule]]


def build(builder: BuilderType, device: torch.device = None):
    if builder is None:
        return None

    if not isinstance(builder, TorchModel):
        builder = TorchModel(model_fn=builder)

    return builder.build(device)
