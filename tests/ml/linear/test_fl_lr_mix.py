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

from sys import platform
from typing import List

import numpy as np
import pytest
import spu
from sklearn.metrics import roc_auc_score

import secretflow as sf
from secretflow.data import partition
from secretflow.data.mix import MixDataFrame
from secretflow.data.split import train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.ml.linear.fl_lr_mix import FlLogisticRegressionMix
from secretflow.security.aggregation import SecureAggregator


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

    x1, x2 = train_test_split(x, train_size=0.5, shuffle=False)
    y1, y2 = train_test_split(y, train_size=0.5, shuffle=False)

    # davy holds same x
    x2_davy = x2.partitions[devices.carol].data.to(devices.davy)
    del x2.partitions[devices.carol]
    x2.partitions[devices.davy] = partition(x2_davy)

    return {
        'x': MixDataFrame(partitions=[x1, x2]),
        'y': MixDataFrame(partitions=[y1, y2]),
        'y_arr': label.values,
    }


def _gen_heus():
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

    heu0 = sf.HEU(heu_config('alice', ['bob', 'carol']), spu.FieldType.FM128)
    heu1 = sf.HEU(heu_config('alice', ['bob', 'davy']), spu.FieldType.FM128)
    return [heu0, heu1]


@pytest.mark.skipif(platform == 'darwin', reason="macOS has accuracy issue")
@pytest.mark.mpc(parties=4)
def test_model_should_ok_when_fit_dataframe(sf_production_setup_devices):
    devices = sf_production_setup_devices
    data = _gen_data(devices)
    heus = _gen_heus()

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
        heus=heus,
        # aggr_hooks=[RouterLrAggrHook(devices.alice)],
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
