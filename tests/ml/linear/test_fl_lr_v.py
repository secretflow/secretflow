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

from dataclasses import dataclass

import numpy as np
import pytest
import spu
from sklearn.metrics import roc_auc_score

import secretflow as sf
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.ml.linear.fl_lr_v import FlLogisticRegressionVertical
from secretflow.security.aggregation.plain_aggregator import PlainAggregator


@dataclass
class DeviceInventory:
    alice: sf.PYU = None
    bob: sf.PYU = None
    carol: sf.PYU = None
    davy: sf.PYU = None
    heu0: sf.HEU = None
    heu1: sf.HEU = None


@pytest.fixture(scope="module")
def env(request, sf_production_setup_linear_env_ray):
    devices, data = sf_production_setup_linear_env_ray

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

    x, y = data['x'], data['y']

    yield devices, {
        'x': x,
        'y': y,
    }


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

    assert auc > 0.97  # TODO:change to 98
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
