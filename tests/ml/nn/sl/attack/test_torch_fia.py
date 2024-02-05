import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn.sl.attacks.fia_torch import (
    FeatureInferenceAttack,
)
from secretflow.ml.nn.utils import BaseModule, TorchModel


class SLBaseNet(BaseModule):
    def __init__(self):
        super(SLBaseNet, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        y = x
        return y

    def output_num(self):
        return 1


class SLFuseModel(BaseModule):
    def __init__(self, input_dim=48, output_dim=11):
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


class Generator(nn.Module):
    def __init__(self, attack_dim, victim_dim):
        super().__init__()
        self.attack_dim = attack_dim
        self.victim_dim = victim_dim
        self.reshape = len(attack_dim) > 1

        input_shape = 1
        for aa in attack_dim:
            input_shape *= aa

        output_shape = 1
        for vv in victim_dim:
            output_shape *= vv

        input_shape += output_shape

        self.net = nn.Sequential(
            nn.Linear(input_shape, 600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(600, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, output_shape),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        if self.reshape:  # pic input
            bs = x.size(0)
            x = x.reshape(bs, -1)
            r = self.net(x)
            r = r.reshape([bs] + self.victim_dim)
            return r
        else:
            return self.net(x)


def data_builder(data, label, batch_size):
    def prepare_data():
        print('prepare_data num: ', data.shape)
        alice_data = data[:, :28]
        bob_data = data[:, 28:]

        alice_dataset = TensorDataset(torch.tensor(alice_data))
        alice_dataloader = DataLoader(
            dataset=alice_dataset,
            shuffle=False,
            batch_size=batch_size,
        )

        bob_dataset = TensorDataset(torch.tensor(bob_data))
        bob_dataloader = DataLoader(
            dataset=bob_dataset,
            shuffle=False,
            batch_size=batch_size,
        )

        dataloader_dict = {'alice': alice_dataloader, 'bob': bob_dataloader}
        return dataloader_dict, dataloader_dict

    return prepare_data


def do_test_sl_and_fia(config: dict, alice, bob):
    enable_mean = config['enable_mean'] if 'enale_mean' in config else True
    attack_epochs = config['attack_epochs'] if 'attack_epochs' in config else 60
    optim_lr = config['optim_lr'] if 'optim_lr' in config else 0.0001
    device_y = alice

    tmp_dir = tempfile.TemporaryDirectory()
    fia_path = tmp_dir.name

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
    pred_fea = np.array(a).astype(np.float32)
    pred_label = np.array(b).astype(np.int64)

    data_fea = np.concatenate((train_fea, test_fea, pred_fea), axis=0)
    bob_mean = data_fea.mean(axis=0)
    bob_mean = bob_mean[28:]

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
    base_model = TorchModel(
        model_fn=SLBaseNet,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=11, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=11, average='micro'
            ),
        ],
    )

    fuse_model = TorchModel(
        model_fn=SLFuseModel,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=11, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=11, average='micro'
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
        backend='torch',
        strategy='split_nn',
    )

    batch_size = 64
    victim_model_save_path = fia_path + '/sl_model_victim'
    victim_model_dict = {
        bob: [TorchModel(model_fn=SLBaseNet), victim_model_save_path],
    }
    optim_fn = optim_wrapper(optim.Adam, lr=optim_lr)
    generator_model = TorchModel(
        model_fn=Generator,
        loss_fn=None,
        optim_fn=optim_fn,
        metrics=None,
        attack_dim=[28],
        victim_dim=[20],
    )

    data_buil = data_builder(pred_fea, pred_label, batch_size)
    generator_save_path = fia_path + '/generator'

    fia_callback = FeatureInferenceAttack(
        victim_model_path=victim_model_save_path,
        attack_party=alice,
        victim_party=bob,
        victim_model_dict=victim_model_dict,
        base_model_list=[alice, bob],
        generator_model_wrapper=generator_model,
        data_builder=data_buil,
        victim_fea_dim=[20],
        attacker_fea_dim=[28],
        enable_mean=enable_mean,
        enable_var=True,
        mean_lambda=1.2,
        var_lambda=0.25,
        attack_epochs=attack_epochs,
        victim_mean_feature=bob_mean,
        save_attacker_path=generator_save_path,
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
        callbacks=[fia_callback],
    )
    metrics = fia_callback.get_attack_metrics()
    return metrics


def test_sl_and_fia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    do_test_sl_and_fia(
        {'enable_mean': False, 'attack_epochs': 2, "optim_lr": 0.0001}, alice, bob
    )
