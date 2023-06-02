from dataclasses import dataclass
from typing import List
from sys import platform

import numpy as np
import pytest
import spu
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

import secretflow as sf
import secretflow.distributed as sfd
from secretflow.data.base import Partition
from secretflow.data.mix import MixDataFrame
from secretflow.data.split import train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.ml.linear.fl_lr_mix import FlLogisticRegressionMix
from secretflow.preprocessing.scaler import StandardScaler
from secretflow.security.aggregation import SecureAggregator
from tests.cluster import cluster, set_self_party


@dataclass
class DeviceInventory:
    alice: sf.PYU = None
    bob: sf.PYU = None
    carol: sf.PYU = None
    davy: sf.PYU = None
    heu0: sf.HEU = None
    heu1: sf.HEU = None


@pytest.fixture(scope="module")
def env(request, sf_party_for_4pc):
    devices = DeviceInventory()
    sfd.set_production(True)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address='local',
        num_cpus=8,
        log_to_driver=True,
        cluster_config=cluster(),
        exit_on_failure_cross_silo_sending=True,
        enable_waiting_for_other_parties_ready=False,
    )

    devices.alice = sf.PYU('alice')
    devices.bob = sf.PYU('bob')
    devices.carol = sf.PYU('carol')
    devices.davy = sf.PYU('davy')

    def heu_config(sk_keeper: str, evaluators: List[str]):
        return {
            'sk_keeper': {'party': sk_keeper},
            'evaluators': [{'party': evaluator} for evaluator in evaluators],
            'mode': 'PHEU',
            'he_parameters': {
                'schema': 'paillier',
                'key_pair': {'generate': {'bit_size': 2048}},
            },
        }

    devices.heu0 = sf.HEU(heu_config('alice', ['bob', 'carol']), spu.spu_pb2.FM128)
    devices.heu1 = sf.HEU(heu_config('alice', ['bob', 'davy']), spu.spu_pb2.FM128)

    features, label = load_breast_cancer(return_X_y=True, as_frame=True)
    label = label.to_frame()
    feat_list = [
        features.iloc[:, :10],
        features.iloc[:, 10:20],
        features.iloc[:, 20:],
    ]
    x = VDataFrame(
        partitions={
            devices.alice: Partition(devices.alice(lambda: feat_list[0])()),
            devices.bob: Partition(devices.bob(lambda: feat_list[1])()),
            devices.carol: Partition(devices.carol(lambda: feat_list[2])()),
        }
    )
    x = StandardScaler().fit_transform(x)
    y = VDataFrame(
        partitions={devices.alice: Partition(devices.alice(lambda: label)())}
    )
    x1, x2 = train_test_split(x, train_size=0.5, shuffle=False)
    y1, y2 = train_test_split(y, train_size=0.5, shuffle=False)

    # davy holds same x
    x2_davy = x2.partitions[devices.carol].data.to(devices.davy)
    del x2.partitions[devices.carol]
    x2.partitions[devices.davy] = Partition(x2_davy)

    yield devices, {
        'x': MixDataFrame(partitions=[x1, x2]),
        'y': MixDataFrame(partitions=[y1, y2]),
        'y_arr': label.values,
    }
    del devices
    sf.shutdown()

@pytest.mark.skipif(platform == 'darwin', reason="macOS has accuracy issue")
def test_model_should_ok_when_fit_dataframe(env):
    devices, data = env

    # GIVEN
    aggregator0 = SecureAggregator(
        devices.alice, [devices.alice, devices.bob, devices.carol]
    )
    aggregator1 = SecureAggregator(
        devices.alice, [devices.alice, devices.bob, devices.davy]
    )
    # aggregator2 = SecureAggregator(self.alice, [self.alice, self.bob, self.eric])

    model = FlLogisticRegressionMix()

    # WHEN
    model.fit(
        data['x'],
        data['y'],
        epochs=3,
        batch_size=64,
        learning_rate=0.1,
        aggregators=[aggregator0, aggregator1],
        heus=[devices.heu0, devices.heu1],
        # aggr_hooks=[RouterLrAggrHook(devices.alice)]
    )

    y_pred = np.concatenate(sf.reveal(model.predict(data['x'])))

    auc = roc_auc_score(data['y_arr'], y_pred)
    acc = np.mean((y_pred > 0.5) == data['y_arr'])

    # THEN
    auc = sf.reveal(auc)
    acc = sf.reveal(acc)
    print(f'auc={auc}, acc={acc}')

    assert auc > 0.98
    assert acc > 0.93
