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

from collections.abc import Iterable
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from secretflow.component.core import (
    CompVDataFrame,
    VTableField,
    VTableFieldKind,
    VTableSchema,
    VTableUtils,
)
from secretflow.data import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.driver import reveal
from secretflow.preprocessing.binning.vert_bin_substitution import VertBinSubstitution
from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.utils.simulation.datasets import dataset
from tests.sf_fixtures import mpc_fixture


def woe_almost_equal(a, b):
    a_list = a["variables"]
    b_list = b["variables"]

    a_dict = {f['name']: f for f in a_list}
    b_dict = {f['name']: f for f in b_list}

    assert a_dict.keys() == b_dict.keys()

    for f_name in a_dict:
        a_f_bin = a_dict[f_name]
        b_f_bin = b_dict[f_name]
        assert a_f_bin.keys() == b_f_bin.keys()
        for k in a_f_bin:
            if isinstance(a_f_bin[k], str) or k == "categories":
                assert a_f_bin[k] == b_f_bin[k], k
            else:
                np.testing.assert_almost_equal(a_f_bin[k], b_f_bin[k], err_msg=k)


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob

    normal_data = pd.read_csv(
        dataset('linear'),
        usecols=['id'] + [f'x{i}' for i in range(1, 11)] + ['y'],
    )
    row_num = normal_data.shape[0]
    np.random.seed(0)
    normal_data['x1'] = np.random.randint(0, 2, (row_num,))
    normal_data['x2'] = np.random.randint(0, 5, (row_num,))
    v_float_data = VDataFrame(
        {
            pyu_alice: partition(data=pyu_alice(lambda: normal_data)()),
            pyu_bob: partition(data=pyu_bob(lambda: normal_data.drop("y", axis=1))()),
        }
    )

    nan_str = (
        "f1,f2,f3,y\n"
        "a,1.1,1,0\n"
        "a,2.1,2,1\n"
        "b,3.4,3,0\n"
        "b,3.7,4,1\n"
        "c,4.4,5,0\n"
        "c,3.8,6,1\n"
        "d,5.4,7,0\n"
        "d,5.2,8,1\n"
        "null,0.4,,0\n"
        ",1.2,,1\n"
        ",10.2,,0\n"
    )

    nan_str_data = pd.read_csv(StringIO(nan_str))
    assert nan_str_data['f1'].dtype == np.dtype(object)
    assert nan_str_data['f2'].dtype == np.float64
    assert nan_str_data['f3'].dtype == np.float64
    assert pd.isna(nan_str_data['f3'][8])
    assert pd.isna(nan_str_data['f1'][8])

    v_nan_data = VDataFrame(
        {
            pyu_alice: partition(data=pyu_alice(lambda: nan_str_data)()),
            pyu_bob: partition(data=pyu_bob(lambda: nan_str_data.drop("y", axis=1))()),
        }
    )

    def _build_schema(df: VDataFrame, labels: set = {"y"}) -> dict[str, VTableSchema]:
        res = {}
        for pyu, p in df.partitions.items():
            fields = []
            for name, dtype in p.dtypes.items():
                dt = VTableUtils.from_dtype(dtype)
                kind = (
                    VTableFieldKind.LABEL if name in labels else VTableFieldKind.FEATURE
                )
                fields.append(VTableField(name, dt, kind))

            res[pyu.party] = VTableSchema(fields)

        return res

    return sf_production_setup_devices, {
        'normal_data': normal_data,
        'v_float_data': CompVDataFrame.from_pandas(
            v_float_data, schemas=_build_schema(v_float_data)
        ),
        'nan_str_data': nan_str_data,
        'v_nan_data': CompVDataFrame.from_pandas(
            v_nan_data, schemas=_build_schema(v_nan_data)
        ),
    }


def _f32(v):
    if isinstance(v, Iterable):
        return list(map(np.float32, v))
    else:
        return np.float32(v)


@pytest.mark.mpc
def test_binning_nan(prod_env_and_data):
    env, data = prod_env_and_data
    ss_binning = VertWoeBinning(env.spu)
    bin_rules = ss_binning.binning(
        data['v_nan_data'],
        binning_method="chimerge",
        bin_names={env.alice: ["f1", "f3", "f2"], env.bob: ["f1", "f3", "f2"]},
        label_name="y",
        chimerge_target_bins=4,
    )

    woe_sub = VertBinSubstitution()
    sub_data = woe_sub.substitution(data['v_nan_data'], bin_rules).to_pandas()
    alice_data = reveal(sub_data.partitions[env.alice].data).drop("y", axis=1)
    bob_data = reveal(sub_data.partitions[env.bob].data)
    rules = {v['name']: v for v in reveal(bin_rules[env.alice])["variables"]}

    assert alice_data.equals(bob_data), str(alice_data) + "\n,,,,,,\n" + str(bob_data)
    f1_categories = _f32(list(set(alice_data['f1'])))
    assert np.isin(_f32(rules['f1']['filling_values']), f1_categories).all(), (
        str(rules['f1']['filling_values']) + "\n,,,,,,\n" + str(f1_categories)
    )
    assert _f32(rules['f1']['else_filling_value']) in _f32(f1_categories)
    f2_categories = _f32(list(set(alice_data['f2'])))
    assert np.isin(f2_categories, _f32(rules['f2']['filling_values'])).all()
    f3_categories = _f32(list(set(alice_data['f3'])))
    assert np.isin(_f32(rules['f3']['filling_values']), f3_categories).all()
    assert _f32(rules['f3']['else_filling_value']) in f3_categories


"""
def test_binning_nan_vert_binning(prod_env_and_data):
    env, data = prod_env_and_data
    vert_binning = VertBinning()

    rules = vert_binning.binning(
        data['v_nan_data'],
        binning_method="eq_range",
        bin_names={env.alice: ["f1", "f3", "f2"], env.bob: ["f1", "f3", "f2"]},
    )

    bin_sub = VertBinSubstitution()
    sub_data = bin_sub.substitution(data['v_nan_data'], rules).to_pandas()
    alice_data = reveal(sub_data.partitions[env.alice].data).drop("y", axis=1)
    bob_data = reveal(sub_data.partitions[env.bob].data)
    rules = {v['name']: v for v in reveal(rules[env.alice])["variables"]}

    assert alice_data.equals(bob_data), str(alice_data) + "\n,,,,,,\n" + str(bob_data)
    f1_categories = _f32(list(set(alice_data['f1'])))
    assert np.isin(_f32(rules['f1']['filling_values']), f1_categories).all(), (
        str(rules['f1']['filling_values']) + "\n,,,,,,\n" + str(f1_categories)
    )
    assert _f32(rules['f1']['else_filling_value']) == -1
    f2_categories = _f32(list(set(alice_data['f2'])))
    assert np.isin(f2_categories, _f32(rules['f2']['filling_values'])).all()
    f3_categories = _f32(list(set(alice_data['f3'])))
    assert np.isin(_f32(rules['f3']['filling_values']), f3_categories).all()
    assert _f32(rules['f3']['else_filling_value']) == -1


def test_binning_normal_vert_binning(prod_env_and_data):
    env, data = prod_env_and_data
    vert_binning = VertBinning()

    rules = vert_binning.binning(
        data['v_float_data'],
        binning_method="eq_range",
        bin_names={env.alice: ["x1", "x2", "x3"], env.bob: ["x1", "x2", "x3"]},
    )
    sub = VertBinSubstitution()
    sub_data = sub.substitution(data['v_float_data'], rules).to_pandas()
    alice_data = reveal(sub_data.partitions[env.alice].data).drop("y", axis=1)
    bob_data = reveal(sub_data.partitions[env.bob].data)
    rules = {v['name']: v for v in reveal(rules[env.alice])["variables"]}

    assert alice_data.equals(bob_data), str(alice_data) + "\n,,,,,,\n" + str(bob_data)
    f1_categories = _f32(list(set(alice_data['x1'])))
    assert np.isin(_f32(rules['x1']['filling_values']), f1_categories).all()
    f2_categories = _f32(list(set(alice_data['x2'])))
    assert np.isin(f2_categories, _f32(rules['x2']['filling_values'])).all()
    f3_categories = _f32(list(set(alice_data['x3'])))
    assert np.isin(_f32(rules['x3']['filling_values']), f3_categories).all()


def test_binning_normal(prod_env_and_data):
    env, data = prod_env_and_data
    ss_binning = VertWoeBinning(env.spu)
    bin_rules = ss_binning.binning(
        data['v_float_data'],
        bin_names={env.alice: ["x1", "x2", "x3"], env.bob: ["x1"]},
        label_name="y",
    )

    woe_sub = VertBinSubstitution()
    sub_data = woe_sub.substitution(data['v_float_data'], bin_rules).to_pandas()
    alice_data = reveal(sub_data.partitions[env.alice].data).drop("y", axis=1)
    bob_data = reveal(sub_data.partitions[env.bob].data)
    rules = {v['name']: v for v in reveal(bin_rules[env.alice])["variables"]}

    assert bob_data.equals(data['normal_data'].drop("y", axis=1)), (
        str(data['normal_data'].drop("y", axis=1)) + "\n,,,,,,\n" + str(bob_data)
    )
    f1_categories = _f32(list(set(alice_data['x1'])))
    assert np.isin(_f32(rules['x1']['filling_values']), f1_categories).all()
    f2_categories = _f32(list(set(alice_data['x2'])))
    assert np.isin(f2_categories, _f32(rules['x2']['filling_values'])).all()
    f3_categories = _f32(list(set(alice_data['x3'])))
    assert np.isin(_f32(rules['x3']['filling_values']), f3_categories).all()


def test_binning_normal_single(prod_env_and_data):
    env, data = prod_env_and_data
    ss_binning = VertWoeBinning(env.spu)
    bin_rules = ss_binning.binning(
        data['v_float_data'],
        bin_names={env.alice: ["x1", "x2", "x3"]},
        label_name="y",
    )
    woe_sub = VertBinSubstitution()
    sub_data = woe_sub.substitution(data['v_float_data'], bin_rules).to_pandas()
    alice_data = reveal(sub_data.partitions[env.alice].data).drop("y", axis=1)
    bob_data = reveal(sub_data.partitions[env.bob].data)
    rules = {v['name']: v for v in reveal(bin_rules[env.alice])["variables"]}

    f1_categories = _f32(list(set(alice_data['x1'])))
    assert np.isin(_f32(rules['x1']['filling_values']), f1_categories).all()
    f2_categories = _f32(list(set(alice_data['x2'])))
    assert np.isin(f2_categories, _f32(rules['x2']['filling_values'])).all()
    f3_categories = _f32(list(set(alice_data['x3'])))
    assert np.isin(_f32(rules['x3']['filling_values']), f3_categories).all()

    np.testing.assert_array_equal(
        bob_data.values,
        reveal(data['v_float_data'].to_pandas().partitions[env.bob].data).values,
    )


def test_fix_issue_1330(prod_env_and_data):
    env, data = prod_env_and_data
    vdf = data["v_float_data"]

    from secretflow.data.split import train_test_split

    train_data, test_data = train_test_split(
        vdf.to_pandas(check_null=False), train_size=0.8, shuffle=True, random_state=1234
    )

    binning = VertBinning()
    cols = vdf.columns
    train_feat = train_data[cols]
    bin_names = {
        party: bins if isinstance(bins, list) else [bins]
        for party, bins in train_feat.partition_columns.items()
    }
    train_data = CompDataFrame.from_pandas(train_data, [], ["y"])
    rules = binning.binning(
        vdata=train_data,
        binning_method="eq_range",
        bin_num=3,
        bin_names=bin_names,
    )

    sub = VertBinSubstitution()
    train_binned = sub.substitution(train_data, rules)
"""
