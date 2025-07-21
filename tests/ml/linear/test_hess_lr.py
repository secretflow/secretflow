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
from typing import Tuple

import numpy as np
import pytest
import spu
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.ml.linear.hess_sgd import HESSLogisticRegression


def gen_heus():
    heu_field = spu.FieldType.FM128
    sk_keeper = "alice"
    evaluator = "bob"

    def _to_party(party: str):
        return {"party": party}

    config_x = {
        "sk_keeper": _to_party(sk_keeper),
        "evaluators": [_to_party(evaluator)],
        "mode": "PHEU",
        "he_parameters": {
            "schema": "ou",
            "key_pair": {"generate": {"bit_size": 2048}},
        },
        "encoding": {
            'cleartext_type': 'DT_I32',
            'encoder': "IntegerEncoder",
            'encoder_args': {"scale": 1},
        },
    }

    heu_x = sf.HEU(config_x, heu_field)

    config_y = copy.deepcopy(config_x)
    config_y["sk_keeper"] = _to_party(evaluator)
    config_y["evaluators"] = [_to_party(sk_keeper)]

    heu_y = sf.HEU(config_y, heu_field)
    return heu_x, heu_y


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


@pytest.mark.mpc
def test_model(sf_production_setup_devices):
    devices = sf_production_setup_devices

    heu_x, heu_y = gen_heus()
    x, y = _load_dataset(devices)

    model = HESSLogisticRegression(devices.spu, heu_x, heu_y)
    model.fit(x, y, epochs=4, batch_size=64)

    print(f"w {sf.reveal(model._w)}")

    label = sf.reveal(y.partitions[devices.bob])
    yhat = sf.reveal(model.predict(x))

    auc = roc_auc_score(label, yhat)

    print(f'auc={auc}')
    assert auc > 0.99

    model.fit(x, y, epochs=4, batch_size=64, learning_rate=0.1)
    yhat = sf.reveal(model.predict(x))
    auc = roc_auc_score(label, yhat)
    print(f'auc={auc}')
    assert auc > 0.98
