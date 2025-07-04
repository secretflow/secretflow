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
import pytest
from sklearn.metrics import roc_auc_score

import secretflow as sf
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.ml.linear.fl_lr_v import FlLogisticRegressionVertical
from secretflow.security.aggregation.plain_aggregator import PlainAggregator


def _gen_data(devices):
    from sklearn.datasets import load_breast_cancer

    from secretflow.preprocessing.scaler import StandardScaler

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

    return {'x': x, 'y': y, 'label': label}


_MPC_PARAMS_HEU = {"heu_config": {"schema": "paillier"}}


@pytest.mark.mpc(parties=3, params=_MPC_PARAMS_HEU)
def test_model_should_ok_when_fit_dataframe(sf_production_setup_devices):
    devices = sf_production_setup_devices
    data = _gen_data(devices)
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


@pytest.mark.mpc(parties=3, params=_MPC_PARAMS_HEU)
def test_model_should_ok_when_fit_ndarray(sf_production_setup_devices):
    devices = sf_production_setup_devices
    data = _gen_data(devices)
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


@pytest.mark.mpc(parties=3, params=_MPC_PARAMS_HEU)
def test_fit_should_error_when_mismatch_heu_sk_keeper(sf_production_setup_devices):
    devices = sf_production_setup_devices
    data = _gen_data(devices)
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
