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
from torchmetrics import Accuracy
import logging
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.attacks.sim_lia_torch import SimilarityLabelInferenceAttack


def do_test_sl_and_sim_lia(alice, bob, config):

    class BaseNet(nn.Module):
        def __init__(self):
            super(BaseNet, self).__init__()
            self.fc1 = nn.Linear(100, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 100)
            self.ReLU = nn.ReLU()

        def forward(self, x):

            x = self.fc1(x)
            x = self.ReLU(x)
            x = self.fc2(x)
            x = self.ReLU(x)
            x = self.fc3(x)
            return x

        def output_num(self):
            return 1

    class FuseNet(nn.Module):
        def __init__(self):
            super(FuseNet, self).__init__()
            self.fc1 = nn.Linear(100, 100)
            self.fc2 = nn.Linear(100, 10)
            self.ReLU = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.ReLU(x)
            x = self.fc2(x)
            return x

    train_data = np.random.rand(1000, 100).astype(np.float32)
    train_label = np.random.randint(0, 10, size=(1000,)).astype(np.int64)

    # put into FedNdarray
    fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(train_data),
            # bob: bob(lambda x: x)(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )

    label = bob(lambda x: x)(train_label)

    # model configure
    loss_fn = nn.CrossEntropyLoss

    optim_fn = optim_wrapper(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4)

    _base_model, _fuse_model = BaseNet, FuseNet

    base_model = TorchModel(
        model_fn=_base_model,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
        ],
    )

    fuse_model = TorchModel(
        model_fn=_fuse_model,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
        ],
    )

    base_model_dict = {
        alice: base_model,
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=bob,
        model_fuse=fuse_model,
        simulation=True,
        backend="torch",
        strategy="split_nn",
    )

    attack_method, data_type, distance_metric = config.split(",")
    logging.info(
        (
            f"attack_method: {attack_method}, distance_metric: {distance_metric}, data_type: {data_type}"
        )
    )

    sim_lia_callback = SimilarityLabelInferenceAttack(
        attack_party=alice,
        label_party=bob,
        data_type=data_type,
        attack_method=attack_method,
        known_num=10,
        distance_metric=distance_metric,
        exec_device="cpu",
    )

    history = sl_model.fit(
        fed_data,
        label,
        validation_data=(fed_data, label),
        epochs=2,
        batch_size=128,
        random_seed=1234,
        callbacks=[sim_lia_callback],
    )
    print(history)

    pred_bs = 128
    result = sl_model.predict(fed_data, batch_size=pred_bs, verbose=1)

    return sim_lia_callback.get_attack_metrics()


def test_sl_and_lia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    config = [
        "k-means,feature,cosine",
        "k-means,grad,cosine",
        "distance,feature,cosine",
        "distance,grad,cosine",
        "distance,feature,euclidean",
        "distance,grad,euclidean",
    ]
    for i in config:
        do_test_sl_and_sim_lia(alice, bob, i)
