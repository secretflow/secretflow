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

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from secretflow import PYU
from secretflow.ml.nn.callbacks import Callback
from secretflow.ml.nn.core.torch import BaseModule, module


class MIDModel(BaseModule):
    """Implementation of defense method in Mutual Information Regularization for Vertical Federated Learning: https://arxiv.org/abs/2301.01142.
    The MID model structure is mainly borrowed from https://github.com/FLAIR-THU/VFLAIR/blob/main/src/models/mid_model_rapper.py

    This defense model can be used in both base model and fuse model. It's a FIA defense method when used in base model and
    a LIA defense method when used in fuse model.

    You can use MIDModel directly in your model, or use MIDefense callback to mix MIDModel into your model automatically.

    Args:
        input_dim: input dim of the mid model, usually the hidden size of model.
        output_dim: output dim of the mid model, usually same as input_dim.
        mid_lambda: loss weight of mid model, range: (0, 1).
        encode_scale: scale of encoder layer, layer size will be encode_scale * input_dim, default: 1.
        decode_scale: scale of decoder layer, layer size will be decode_scale * input_dim, default: 5.
        std_shift: shift of encoder std.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mid_lambda: float,
        encode_scale: int = 1,
        decode_scale: int = 5,
        std_shift: float = 0.5,
    ):
        super(MIDModel, self).__init__()

        self.encode_scale = encode_scale
        self.decode_scale = decode_scale
        self.input_dim = input_dim
        self.mid_lambda = mid_lambda
        self.std_shift = std_shift
        self.encoder_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim * 2 * encode_scale),
            nn.ReLU(inplace=True),
        )
        self.decoder_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * encode_scale, input_dim * decode_scale),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * decode_scale, output_dim),
            nn.ReLU(inplace=True),
        )

    def epsilon(self, x):
        epsilon = torch.empty((x.shape[0], x.shape[1] * self.encode_scale)).to(x.device)
        torch.nn.init.normal_(epsilon, mean=0, std=1)
        return epsilon

    def encoder(self, x):
        x_double = self.encoder_layer(x)
        mu, std = (
            x_double[:, : self.input_dim * self.encode_scale],
            x_double[:, self.input_dim * self.encode_scale :],
        )
        std = F.softplus(std - self.std_shift)

        return mu, std

    def forward(self, x):
        mu, std = self.encoder(x)
        z = mu + std * self.epsilon(x)
        z = self.decoder_layer(z)
        return z

    def forward_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, _ = batch
        mu, std = self.encoder(x)
        z = mu + std * self.epsilon(x)
        z = self.decoder_layer(z)

        mid_loss = self.mid_lambda * torch.mean(
            torch.sum((-0.5) * (1 + 2 * torch.log(std) - mu**2 - std**2), 1)
        )

        return z, mid_loss


class MIDBaseModelWrapper(BaseModule):
    def __init__(
        self,
        model: BaseModule,
        input_dim: int,
        output_dim: int,
        mid_lambda: float,
        encode_scale: int = 1,
        decode_scale: int = 5,
        std_shift: float = 0.5,
    ):
        super().__init__()

        assert (
            model.automatic_optimization
        ), "models use MID must use automatic optimization."

        self.model = model
        self.mid_model = MIDModel(
            input_dim, output_dim, mid_lambda, encode_scale, decode_scale, std_shift
        )

        self._optimizers = self.model._optimizers
        self.metrics = self.model.metrics
        self.logs = self.model.logs

    def forward(self, x):
        h = self.model(x)
        z = self.mid_model(h)
        return z

    def forward_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        h, model_loss = self.model.forward_step(batch, batch_idx, dataloader_idx)

        z, mid_loss = self.mid_model.forward_step((h, None), batch_idx)

        total_loss = mid_loss
        if model_loss is not None:
            total_loss += model_loss

        return z, total_loss


class MidFuseModelWrapper(BaseModule):
    def __init__(
        self,
        model: BaseModule,
        mid_params: List[Dict[str, Any]],
    ):
        super().__init__()
        assert (
            model.automatic_optimization
        ), "models use MID must use automatic optimization."

        self.model = model
        mid_model_list = []
        for params in mid_params:
            if params is None:
                mid_model_list.append(None)
                continue

            mid_model_list.append(MIDModel(**params))
        self.mid_model_list = nn.ModuleList(mid_model_list)
        self._optimizers = self.model._optimizers
        self.metrics = self.model.metrics
        self.logs = self.model.logs

    def forward(self, hiddens):
        mid_hiddens = []
        for mid_model, h in zip(self.mid_model_list, hiddens):
            if mid_model is None:
                mid_hiddens.append(h)
            else:
                mh = mid_model(h)
                mid_hiddens.append(mh)

        return self.model(mid_hiddens)

    def forward_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        hiddens, y = batch
        mid_hiddens = []
        mid_loss = 0
        for mid_model, h in zip(self.mid_model_list, hiddens):
            if mid_model is None:
                mid_hiddens.append(h)
            else:
                mh, loss = mid_model.forward_step((h, None), batch_idx)
                mid_loss += loss
                mid_hiddens.append(mh)

        y_pred, model_loss = self.model.forward_step(
            (mid_hiddens, y), batch_idx, dataloader_idx
        )
        total_loss = model_loss + mid_loss
        return y_pred, total_loss


class MIDefense(Callback):
    """Callback for MID defense method.

    For each PYU in base_params, a mid model will surround the base model output.
    For each PYU in fuse_params, a mid model will surround the hidden layer from that PYU.

    Args:
        base_params: base model params dict, key is PYU, value is args of MIDModel, can be empty.
        fuse_params: fuse model params dict, key is PYU, value is args of MIDModel, can be empty.
    """

    def __init__(
        self,
        base_params: Dict[PYU, Dict],
        fuse_params: Dict[PYU, Dict],
        exec_device: torch.device | str = 'cpu',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.base_params = base_params
        self.fuse_params = fuse_params
        self.exec_device = exec_device

    @staticmethod
    def apply_base_model(worker, params: Dict, exec_device: torch.device):
        worker.use_base_loss = True
        worker.model_base = MIDBaseModelWrapper(worker.model_base, **params).to(
            exec_device
        )

    @staticmethod
    def apply_fuse_model(
        worker, fuse_params_list: List[Dict], exec_device: torch.device
    ):
        worker.model_fuse = MidFuseModelWrapper(
            worker.model_fuse, mid_params=fuse_params_list
        ).to(exec_device)

    def on_train_begin(self, logs=None):
        for pyu, params in self.base_params.items():
            if params is None:
                continue
            worker = self._workers[pyu]
            worker.apply(self.apply_base_model, params, self.exec_device)

        fuse_params_list = []
        for pyu in self._workers.keys():
            fuse_params_list.append(self.fuse_params.get(pyu, None))

        if any(fuse_params_list):
            self._workers[self.device_y].apply(
                self.apply_fuse_model, fuse_params_list, self.exec_device
            )

    def create_model_builder(
        self, orig_base_model_dict, orig_model_fuse, exec_device='cpu'
    ):
        def create_base_builder(params, model_builder):
            def builder():
                model = module.build(model_builder)
                wrapper = MIDBaseModelWrapper(model, **params).to(exec_device)
                return wrapper

            return builder

        def create_fuse_builder(mid_params, model_builder):
            def builder():
                model = module.build(model_builder)
                wrapper = MidFuseModelWrapper(model, mid_params).to(exec_device)
                return wrapper

            return builder

        base_model_dict = {}
        for pyu, builder in orig_base_model_dict.items():
            if pyu not in self.base_params:
                base_model_dict[pyu] = builder
            else:
                base_model_dict[pyu] = create_base_builder(
                    self.base_params[pyu], builder
                )

        fuse_params_list = []
        for pyu in orig_base_model_dict.keys():
            fuse_params_list.append(self.fuse_params.get(pyu, None))

        model_fuse = create_fuse_builder(fuse_params_list, orig_model_fuse)

        return base_model_dict, model_fuse
