# Copyright 2024 Ant Group Co., Ltd.
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

import numpy as np
import torch
from torch import nn
from torchmetrics import Accuracy

import secretflow as sf
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow.ml.nn.sl.defenses.cafe_fake_gradients import (
    CAFEFakeGradientsMultiClient,
)
from secretflow.preprocessing import StandardScaler
from secretflow.utils.simulation.datasets import load_creditcard_small


def test_fake_gradients_multi_client(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    carol = sf_simulation_setup_devices.carol

    data = load_creditcard_small({alice: (0, 25), carol: (25, 29)}, num_sample=500)
    label = load_creditcard_small({bob: (29, 30)}, num_sample=500).astype(np.float32)
    scaler = StandardScaler()
    data = scaler.fit_transform(data).astype(np.float32)
    random_state = 1234
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.8, random_state=random_state
    )
    base_model = TorchModel(
        model_fn=DnnBase,
        loss_fn=nn.BCELoss,
        optim_fn=optim_wrapper(torch.optim.Adam),
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
        ],
        input_dims=[25],
        dnn_units_size=[16],
    )
    base_model2 = TorchModel(
        model_fn=DnnBase,
        loss_fn=nn.BCELoss,
        optim_fn=optim_wrapper(torch.optim.Adam),
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
        ],
        input_dims=[4],
        dnn_units_size=[16],
    )
    fuse_model = TorchModel(
        model_fn=DnnFuse,
        loss_fn=nn.BCELoss,
        optim_fn=optim_wrapper(torch.optim.Adam),
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
        ],
        input_dims=[32],
        dnn_units_size=[1],
    )
    base_model_dict = {
        alice: base_model,
        carol: base_model2,
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=bob,
        model_fuse=fuse_model,
        simulation=True,
        random_seed=1234,
        strategy="split_nn",
        backend="torch",
        # agg_method=Concat,
    )
    fake_gradients = CAFEFakeGradientsMultiClient(backend="torch")

    history = sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=1,
        batch_size=128,
        shuffle=False,
        random_seed=1234,
        callbacks=[fake_gradients],
    )
    print(history)
    # assert history["val_BinaryAccuracy"][-1] > 0.6
