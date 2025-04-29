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
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC

from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.attacks.grad_lia_attack_torch import (
    GradientClusterLabelInferenceAttack,
)


def test_grad_lia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    data = np.random.randn(400, 10).astype(np.float32)
    label = np.random.randint(2, size=(400,)).astype(np.float32)
    df = pd.DataFrame(data)
    label = pd.DataFrame(label)
    vdata = VDataFrame(
        partitions={
            alice: partition(lambda df: df.iloc[:, :5], device=alice, df=df),
            bob: partition(lambda df: df.iloc[:, 5:], device=bob, df=df),
        }
    )
    vlabel = VDataFrame(
        partitions={alice: partition(lambda l: l, device=alice, l=label)}
    )
    base_model = TorchModel(
        model_fn=DnnBase,
        loss_fn=nn.BCELoss,
        optim_fn=optim_wrapper(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4),
        metrics=[metric_wrapper(AUROC, task="binary")],
        input_dims=[5],
        dnn_units_size=[64, 16],
    )
    fuse_model = TorchModel(
        model_fn=DnnFuse,
        loss_fn=nn.BCELoss,
        optim_fn=optim_wrapper(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4),
        metrics=[metric_wrapper(AUROC, task="binary")],
        input_dims=[32],
        dnn_units_size=[64, 1],
    )
    sl_model = SLModel(
        base_model_dict={alice: base_model, bob: base_model},
        device_y=alice,
        model_fuse=fuse_model,
        backend="torch",
        strategy="split_nn",
    )
    grad_lia = GradientClusterLabelInferenceAttack(
        attack_party=bob, label_party=alice, num_classes=2
    )
    sl_model.fit(
        vdata,
        vlabel,
        validation_data=(vdata, vlabel),
        epochs=1,
        batch_size=64,
        random_seed=1234,
        callbacks=[grad_lia],
    )
    attack_metrics = grad_lia.get_attack_metrics()
    assert "attack_acc" in attack_metrics
