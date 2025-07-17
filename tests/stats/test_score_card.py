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

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.stats import ScoreCard
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob

    sc = ScoreCard(20, 600, 20)
    pred1 = np.random.random((10, 1))
    ds1 = FedNdarray(
        partitions={pyu_alice: pyu_alice(lambda x: x)(pred1)},
        partition_way=PartitionWay.VERTICAL,
    )

    alice_pred2 = np.random.random((10, 1))
    bob_pred2 = np.random.random((10, 1))
    ds2 = FedNdarray(
        partitions={
            pyu_alice: pyu_alice(lambda x: x)(alice_pred2),
            pyu_bob: pyu_bob(lambda x: x)(bob_pred2),
        },
        partition_way=PartitionWay.HORIZONTAL,
    )

    return sf_production_setup_devices, {
        'sc': sc,
        'ds1': ds1,
        'ds2': ds2,
        'pred1': pred1,
        'alice_pred2': alice_pred2,
        'bob_pred2': bob_pred2,
    }


@pytest.mark.mpc
def test_sc(prod_env_and_data):
    env, data = prod_env_and_data
    scord = data['sc'].transform(data['ds1'])
    assert scord.shape[0] == 10
    assert len(scord.partitions) == 1
    scord1 = sf.reveal(list(scord.partitions.items())[0])
    print(f"pred  \n{data['pred1']}\n -> \n{scord1}")

    scord = data['sc'].transform(data['ds2'])
    assert scord.shape[0] == 20
    assert len(scord.partitions) == 2
    scord2_alice = sf.reveal(list(scord.partitions.items())[0])
    scord2_bob = sf.reveal(list(scord.partitions.items())[1])
    print(f"pred \n{data['alice_pred2']}\n -> \n{scord2_alice}")
    print(f"pred \n{data['bob_pred2']}\n -> \n{scord2_bob}")
