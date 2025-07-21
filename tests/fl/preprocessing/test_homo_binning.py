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

import tempfile

import numpy as np
import pandas as pd
import pytest

from secretflow.data.horizontal import read_csv as h_read_csv
from secretflow.device import reveal
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow_fl.preprocessing.binning.homo_binning import HomoBinning
from tests.sf_fixtures import mpc_fixture

_temp_dir = tempfile.mkdtemp()


def gen_data(data_num, feature_num, is_sparse=False, use_random=False, data_bin_num=10):
    data = []
    shift_iter = 0
    header = ["x" + str(i) for i in range(feature_num)]
    index_colname_map = {}
    for index, name in enumerate(header):
        index_colname_map[index] = name
    for data_key in range(data_num):
        value = data_key % data_bin_num
        if value == 0:
            if shift_iter % data_bin_num == 0:
                value = data_bin_num - 1
            shift_iter += 1
        if not is_sparse:
            if not use_random:
                features = value * np.ones(feature_num)
            else:
                features = np.random.random(feature_num)
        else:
            pass
        data.append(features)
    data = np.array(data)
    data = pd.DataFrame(data)
    data.rename(columns=index_colname_map, inplace=True)
    return data


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob
    pyu_carol = sf_production_setup_devices.carol

    data1 = gen_data(10000, 10, use_random=False)
    data2 = gen_data(5000, 10, use_random=False)
    dfs = [data1, data2]

    file_uris = {
        pyu_alice: f'{_temp_dir}/test_alice.csv',
        pyu_bob: f'{_temp_dir}/test_bob.csv',
    }

    for df, file_uri in zip(dfs, file_uris.values()):
        df.to_csv(file_uri)

    hdf = h_read_csv(
        file_uris,
        aggregator=PlainAggregator(pyu_carol),
        comparator=PlainComparator(pyu_carol),
    )

    return sf_production_setup_devices, {
        'data1': data1,
        'data2': data2,
        'dfs': dfs,
        'file_uris': file_uris,
        'hdf': hdf,
    }


@pytest.mark.mpc(parties=3)
def test_homo_binning(prod_env_and_data):
    env, data = prod_env_and_data
    # GIVEN
    bin_obj = HomoBinning(
        bin_num=5,
        bin_indexes=[1, 2, 3, 4],
        error=1e-9,  # TODO:1e-10
        max_iter=200,
        compress_thres=30,
    )
    bin_result = bin_obj.fit_split_points(data['hdf'])
    bin_result_df = pd.DataFrame.from_dict(reveal(bin_result))

    expect_result = {
        "x0": [2.0, 4.0, 6.0, 8.0, 9.0],
        "x1": [2.0, 4.0, 6.0, 8.0, 9.0],
        "x2": [2.0, 4.0, 6.0, 8.0, 9.0],
        "x3": [2.0, 4.0, 6.0, 8.0, 9.0],
    }
    expect_df = pd.DataFrame.from_dict(expect_result)
    pd.testing.assert_frame_equal(bin_result_df, expect_df, rtol=1e-2)
