from dataclasses import dataclass

import numpy as np
import pytest
import spu
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

import secretflow as sf
import secretflow.distributed as sfd
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.ml.linear.fl_lr_v import FlLogisticRegressionVertical
from secretflow.preprocessing import StandardScaler
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
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
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.PRODUCTION)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address='local',
        num_cpus=8,
        log_to_driver=True,
        cluster_config=cluster(),
        cross_silo_comm_backend='brpc_link',
        cross_silo_comm_options={
            'exit_on_sending_failure': True,
            'http_max_payload_size': 5 * 1024 * 1024,
            'recv_timeout_ms': 1800 * 1000,
        },
        enable_waiting_for_other_parties_ready=False,
    )

    devices.alice = sf.PYU('alice')
    devices.bob = sf.PYU('bob')
    devices.carol = sf.PYU('carol')
    devices.davy = sf.PYU('davy')

    heu_config = {
        'sk_keeper': {'party': 'alice'},
        'evaluators': [{'party': 'bob'}, {'party': 'carol'}],
        'mode': 'PHEU',
        'he_parameters': {
            'schema': 'paillier',
            'key_pair': {'generate': {'bit_size': 2048}},
        },
    }

    devices.heu = sf.HEU(heu_config, spu.spu_pb2.FM128)

    features, label = load_breast_cancer(return_X_y=True, as_frame=True)
    label = label.to_frame()
    feat_list = [
        features.iloc[:, :10],
        features.iloc[:, 10:20],
        features.iloc[:, 20:],
    ]
    x = VDataFrame(
        partitions={
            devices.alice: partition(devices.alice(lambda: feat_list[0])()),
            devices.bob: partition(devices.bob(lambda: feat_list[1])()),
            devices.carol: partition(devices.carol(lambda: feat_list[2])()),
        }
    )
    x = StandardScaler().fit_transform(x)
    y = VDataFrame(
        partitions={devices.alice: partition(devices.alice(lambda: label)())}
    )

    yield devices, {
        'x': x,
        'y': y,
    }
    del devices
    sf.shutdown()


def test_model_should_ok_when_fit_dataframe(env):
    devices, data = env
    # GIVEN
    aggregator = PlainAggregator(devices.alice)

    model = FlLogisticRegressionVertical(
        [devices.alice, devices.bob, devices.carol], aggregator, devices.heu
    )

    # WHEN
    model.fit(data['x'], data['y'], epochs=3, batch_size=64)

    y_pred = model.predict(data['x'])

    y = data['y'].values.partitions[devices.alice]
    auc = devices.alice(roc_auc_score)(y, y_pred)
    acc = devices.alice(lambda y_true, y_pred: np.mean((y_pred > 0.5) == y_true))(
        y, y_pred
    )

    # THEN
    auc = sf.reveal(auc)
    acc = sf.reveal(acc)
    print(f'auc={auc}, acc={acc}')

    assert auc > 0.99
    assert acc > 0.94


def test_model_should_ok_when_fit_ndarray(env):
    devices, data = env
    # GIVEN
    aggregator = PlainAggregator(devices.alice)

    model = FlLogisticRegressionVertical(
        [devices.alice, devices.bob, devices.carol], aggregator, devices.heu
    )
    x = data['x'].values
    y = data['y'].values

    # WHEN
    model.fit(x, y, epochs=3, batch_size=64)

    y_pred = model.predict(x)

    y = y.partitions[devices.alice]
    auc = devices.alice(roc_auc_score)(y, y_pred)
    acc = devices.alice(lambda y_true, y_pred: np.mean((y_pred > 0.5) == y_true))(
        y, y_pred
    )

    # THEN
    auc = sf.reveal(auc)
    acc = sf.reveal(acc)
    print(f'auc={auc}, acc={acc}')

    assert auc > 0.99
    assert acc > 0.94


def test_fit_should_error_when_mismatch_heu_sk_keeper(env):
    devices, data = env
    # GIVEN
    aggregator = PlainAggregator(devices.alice)

    model = FlLogisticRegressionVertical(
        [devices.alice, devices.bob, devices.carol], aggregator, devices.heu
    )
    x = data['x'].values
    y = VDataFrame(
        partitions={devices.bob: partition(devices.bob(lambda: [1, 2, 3])())}
    )

    # WHEN
    with pytest.raises(
        AssertionError, match='Y party shoule be same with heu sk keeper'
    ):
        model.fit(x, y, epochs=3, batch_size=64)
