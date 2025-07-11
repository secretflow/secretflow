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
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision

import secretflow as sf
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.defenses.confusional_autoencoder import CAEDefense

from ..attack.model_def import BottomModelForCifar10, TopModelForCifar10


def test_sl_and_defense(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    device_y = bob

    bs = 5
    sample_size = 20
    train_data = np.random.rand(sample_size, 3, 32, 32).astype(np.float32)
    train_label = np.random.randint(0, 10, size=(sample_size))

    # put into FedNdarray
    fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :, :, 0:16])(train_data),
            bob: bob(lambda x: x[:, :, :, 16:32])(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label = bob(lambda x: x)(train_label)

    # model configure
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=1e-3)
    base_model = TorchModel(
        model_fn=BottomModelForCifar10,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average="micro"
            ),
        ],
    )

    fuse_model = TorchModel(
        model_fn=TopModelForCifar10,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average="micro"
            ),
        ],
    )

    base_model_dict = {
        alice: base_model,
        bob: base_model,
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=fuse_model,
        dp_strategy_dict=None,
        compressor=None,
        simulation=True,
        random_seed=1234,
        backend="torch",
        strategy="split_nn",
    )

    cae_cb = CAEDefense(
        defense_party=device_y,
        exec_device="cpu",
        autoencoder_epochs=1,
        train_sample_size=10000,
        test_sample_size=2000,
    )

    history = sl_model.fit(
        fed_data,
        label,
        validation_data=(fed_data, label),
        epochs=1,
        batch_size=bs,
        shuffle=False,
        random_seed=1234,
        callbacks=[cae_cb],
    )
    print("history: ", history)

    res = sf.reveal(sl_model.predict(fed_data, callbacks=[cae_cb]))
    assert res[0].size()[0] == sample_size
