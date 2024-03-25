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

import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import secretflow as sf
import secretflow.distributed as sfd
from secretflow.data import FedNdarray, PartitionWay
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.ml.linear.hess_sgd import HESSLogisticRegression
from tests.cluster import cluster, set_self_party
from tests.conftest import heu_config, semi2k_cluster


@dataclass
class DeviceInventory:
    alice: sf.PYU = None
    bob: sf.PYU = None
    carol: sf.PYU = None
    davy: sf.PYU = None
    spu: sf.SPU = None
    heu_x: sf.HEU = None
    heu_y: sf.HEU = None


@pytest.fixture(scope="module")
def env(request, sf_party_for_4pc):
    devices = DeviceInventory()
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.PRODUCTION)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address='local',
        num_cpus=8,
        log_to_driver=True,
        cluster_config=cluster(),
        enable_waiting_for_other_parties_ready=False,
    )

    devices.alice = sf.PYU('alice')
    devices.bob = sf.PYU('bob')
    devices.carol = sf.PYU('carol')
    devices.davy = sf.PYU('davy')

    cluster_def = sf.reveal(devices.alice(semi2k_cluster)())

    devices.spu = sf.SPU(
        cluster_def,
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
        },
    )

    config_x = heu_config
    config_x['encoding'] = {
        'cleartext_type': 'DT_I32',
        'encoder': "IntegerEncoder",
        'encoder_args': {"scale": 1},
    }
    devices.heu_x = sf.HEU(config_x, devices.spu.cluster_def['runtime_config']['field'])

    config_y = copy.deepcopy(config_x)
    sk_keeper = config_y["sk_keeper"]
    evaluator = config_y["evaluators"][0]
    config_y["sk_keeper"] = evaluator
    config_y["evaluators"][0] = sk_keeper

    devices.heu_y = sf.HEU(config_y, devices.spu.cluster_def['runtime_config']['field'])

    yield devices
    del devices
    sf.shutdown()


def _load_dataset(env):
    def _load_dataset(return_label=False) -> Tuple[np.ndarray, np.ndarray]:
        features, label = load_breast_cancer(return_X_y=True)

        if return_label:
            return features[:, 15:], label
        else:
            return features[:, :15], None

    def _transform(data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    x1, _ = env.alice(_load_dataset)(return_label=False)
    x2, y = env.bob(_load_dataset)(return_label=True)

    x1 = env.alice(_transform)(x1)
    x2 = env.bob(_transform)(x2)

    x = FedNdarray(
        partitions={x1.device: x1, x2.device: x2},
        partition_way=PartitionWay.VERTICAL,
    )
    y = FedNdarray(partitions={y.device: y}, partition_way=PartitionWay.VERTICAL)

    return x, y


def test_model(env):
    x, y = _load_dataset(env)

    model = HESSLogisticRegression(env.spu, env.heu_x, env.heu_y)
    model.fit(x, y, epochs=4, batch_size=64)

    print(f"w {sf.reveal(model._w)}")

    label = sf.reveal(y.partitions[env.bob])
    yhat = sf.reveal(model.predict(x))

    auc = roc_auc_score(label, yhat)

    print(f'auc={auc}')
    assert auc > 0.99

    model.fit(x, y, epochs=4, batch_size=64, learning_rate=0.1)
    yhat = sf.reveal(model.predict(x))
    auc = roc_auc_score(label, yhat)
    print(f'auc={auc}')
    assert auc > 0.98
