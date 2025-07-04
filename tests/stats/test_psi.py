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

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay, partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import psi_eval
from secretflow.stats.core.utils import equal_range
from tests.sf_fixtures import mpc_fixture


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice

    y_actual = FedNdarray(
        partitions={
            pyu_alice: pyu_alice(lambda x: x)(jnp.array([*range(10)]).reshape(-1, 1))
        },
        partition_way=PartitionWay.VERTICAL,
    )
    y_expected_1 = FedNdarray(
        partitions={
            pyu_alice: pyu_alice(lambda x: x)(jnp.array([*range(10)]).reshape(-1, 1))
        },
        partition_way=PartitionWay.VERTICAL,
    )

    y_expected_2_pd_dataframe = pd.DataFrame(
        {
            'y_expected': [0, 0, 0, 0, 0, 2, 3, 4, 5, 6],
        }
    )

    y_expected_2 = VDataFrame(
        partitions={
            pyu_alice: partition(
                data=pyu_alice(lambda x: x)(y_expected_2_pd_dataframe)
            ),
        }
    )

    split_points = equal_range(jnp.array([*range(10)]), 2)

    return sf_production_setup_devices, {
        "y_actual": y_actual,
        "y_expected_1": y_expected_1,
        "split_points": split_points,
        "y_expected_2": y_expected_2,
    }


@pytest.mark.mpc
def test_psi(prod_env_and_data):
    env, data = prod_env_and_data
    score_1 = reveal(
        psi_eval(data['y_actual'], data['y_expected_1'], data['split_points'])
    )
    score_2 = reveal(
        psi_eval(data['y_actual'], data['y_expected_2'], data['split_points'])
    )
    true_score_2 = (0.5 - 0.8) * np.log(0.5 / 0.8) + (0.5 - 0.2) * np.log(0.5 / 0.2)
    np.testing.assert_almost_equal(score_1, 0.0, decimal=2)
    np.testing.assert_almost_equal(score_2, true_score_2, decimal=2)
