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

import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from secretflow_fl.ml.nn.sl.attacks.fsha_torch import FeatureSpaceHijackingAttack


class AliceSLBaseNet(BaseModule):
    def __init__(self):
        super(AliceSLBaseNet, self).__init__()
        self.linear = nn.Linear(28, 64)

    def forward(self, x):
        y = self.linear(x)
        return y

    def output_num(self):
        return 1


class BobSLBaseNet(BaseModule):
    def __init__(self):
        super(BobSLBaseNet, self).__init__()
        self.linear = nn.Linear(20, 64)

    def forward(self, x):
        y = self.linear(x)
        return y

    def output_num(self):
        return 1


class SLFuseModel(BaseModule):
    def __init__(self, input_dim=128, output_dim=11):
        super(SLFuseModel, self).__init__()
        torch.manual_seed(1234)
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim),
        )

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return self.dense(x)


class Pilot(nn.Module):
    def __init__(self, input_dim=20, target_dim=64):
        super().__init__()
        self.net = nn.Linear(input_dim, target_dim)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, target_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, target_dim),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, target_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, target_dim),
        )

    def forward(self, x):
        return self.net(x)


def data_builder(target_data, aux_data, batch_size, train_size):
    target_data = target_data[:, 28:]
    aux_data = aux_data[:, 28:]
    idx = np.arange(aux_data.shape[0])
    sample_idx = np.random.choice(idx, size=train_size, replace=True)
    aux_data = aux_data[sample_idx]

    def prepare_data():
        target_dataset = TensorDataset(torch.tensor(target_data))
        target_dataloader = DataLoader(
            dataset=target_dataset,
            shuffle=False,
            batch_size=batch_size,
        )
        aux_dataset = TensorDataset(torch.tensor(aux_data))
        aux_dataloader = DataLoader(
            dataset=aux_dataset,
            shuffle=False,
            batch_size=batch_size,
        )
        return target_dataloader, aux_dataloader

    return prepare_data


def do_test_sl_and_fsha(config: dict, alice, bob):
    device_y = alice

    tmp_dir = tempfile.TemporaryDirectory()
    fsha_path = tmp_dir.name

    # example data: 2 * 48
    a = [
        [
            0.37313217,
            0.7419293,
            0.4503198,
            0.0,
            0.15831026,
            0.45544708,
            0.0,
            0.05980535,
            0.07682485,
            0.16307054,
            0.11160016,
            0.12957136,
            0.7419293,
            0.11375341,
            0.9142711,
            0.44000423,
            1.0,
            0.22890328,
            0.22547188,
            0.16455701,
            0.64451,
            0.46011323,
            0.9142711,
            0.7419293,
            0.4828744,
            0.22630581,
            0.18375066,
            0.55607855,
            0.44052622,
            0.91091925,
            0.33469912,
            0.05109914,
            0.0,
            0.16758855,
            0.10267376,
            1.0,
            0.0,
            0.41330767,
            0.9142711,
            0.03903092,
            0.44000423,
            0.06001657,
            0.16131945,
            0.43294185,
            0.0,
            0.556717,
            0.06283119,
            0.5551476,
        ],
        [
            0.79601526,
            1.0,
            0.27199343,
            1.0,
            0.0,
            0.24018548,
            1.0,
            0.63244516,
            0.35403085,
            0.34976155,
            1.0,
            1.0,
            1.0,
            0.79320437,
            0.7714237,
            1.0,
            0.4448759,
            0.25663236,
            0.27058935,
            0.31781185,
            0.9895554,
            1.0,
            0.7714237,
            1.0,
            0.4329526,
            0.27190614,
            1.0,
            0.35267952,
            1.0,
            0.72828853,
            0.2890689,
            0.0,
            0.46661487,
            0.42133534,
            1.0,
            0.21951422,
            1.0,
            0.8043404,
            0.7714237,
            1.0,
            1.0,
            0.6310099,
            0.3497153,
            0.27636313,
            1.0,
            0.35258734,
            0.6463324,
            0.35425386,
        ],
    ]
    b = [0, 0]

    train_fea = np.array(a).astype(np.float32)
    train_label = np.array(b).astype(np.int64)
    test_fea = np.array(a).astype(np.float32)
    test_label = np.array(b).astype(np.int64)

    target_data = train_fea
    aux_data = train_fea
    train_size = train_fea.shape[0]

    fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :28])(train_fea),
            bob: bob(lambda x: x[:, 28:])(train_fea),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    test_fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :28])(test_fea),
            bob: bob(lambda x: x[:, 28:])(test_fea),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    test_data_label = device_y(lambda x: x)(test_label)

    label = device_y(lambda x: x)(train_label)

    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(torch.optim.Adam)

    alice_base_model = TorchModel(
        model_fn=AliceSLBaseNet,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=11, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=11, average="micro"
            ),
        ],
    )

    bob_base_model = TorchModel(
        model_fn=BobSLBaseNet,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=11, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=11, average="micro"
            ),
        ],
    )

    fuse_model = TorchModel(
        model_fn=SLFuseModel,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=11, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=11, average="micro"
            ),
        ],
    )

    base_model_dict = {
        alice: alice_base_model,
        bob: bob_base_model,
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

    batch_size = 64
    victim_model_save_path = fsha_path + "/sl_model_victim"
    victim_model_dict = {
        "bob": [BobSLBaseNet, victim_model_save_path],
    }
    pilot_optim_fn = optim_wrapper(optim.Adam, lr=0.0001)
    pilot_model = TorchModel(
        model_fn=Pilot,
        loss_fn=None,
        optim_fn=pilot_optim_fn,
        metrics=None,
    )

    decoder_optim_fn = optim_wrapper(optim.Adam, lr=0.0001)
    decoder_model = TorchModel(
        model_fn=Decoder,
        loss_fn=None,
        optim_fn=decoder_optim_fn,
        metrics=None,
    )

    discriminator_optim_fn = optim_wrapper(optim.Adam, lr=0.0001)
    discriminator_model = TorchModel(
        model_fn=Discriminator,
        loss_fn=None,
        optim_fn=discriminator_optim_fn,
        metrics=None,
    )
    data_buil = data_builder(target_data, aux_data, batch_size, train_size)

    fsha_callback = FeatureSpaceHijackingAttack(
        attack_party=alice,
        victim_party=bob,
        base_model_list=[alice, bob],
        pilot_model_wrapper=pilot_model,
        decoder_model_wrapper=decoder_model,
        discriminator_model_wrapper=discriminator_model,
        reconstruct_loss_builder=torch.nn.MSELoss,
        data_builder=data_buil,
        victim_fea_dim=20,
        attacker_fea_dim=28,
        gradient_penalty_weight=500,
    )

    sl_model.fit(
        fed_data,
        label,
        validation_data=(test_fed_data, test_data_label),
        epochs=1,
        batch_size=batch_size,
        shuffle=False,
        random_seed=1234,
        dataset_builder=None,
        callbacks=[fsha_callback],
    )
    metrics = fsha_callback.get_attack_metrics()
    return metrics


def test_sl_and_fsha(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    do_test_sl_and_fsha({"optim_lr": 0.0001}, alice, bob)
