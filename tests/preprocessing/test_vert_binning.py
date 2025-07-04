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

from io import StringIO

import numpy as np
import pandas as pd
import pytest

import secretflow.distributed as sfd
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
from secretflow.preprocessing.binning.vert_binning import VertBinning
from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.utils import secure_pickle as pickle
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
                np.testing.assert_almost_equal(
                    a_f_bin[k],
                    b_f_bin[k],
                    err_msg=f"{f_name, k, a_f_bin[k], b_f_bin[k]}",
                )


def audit_ciphertext_equal(a, b):
    def get_c_in_a(s):
        pos = s.find('c:') + len('c:')
        return eval(f"0x{s[pos:]}")

    def get_c_in_b(s):
        return eval(str(s))

    for i in range(a.shape[0]):
        sa = a[i]
        sb = b[i]
        assert get_c_in_a(sa) == get_c_in_b(sb), f"{sa}\n...........\n{sb}"


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    pyu_alice = sf_production_setup_devices.alice
    pyu_bob = sf_production_setup_devices.bob
    pyu_carol = sf_production_setup_devices.carol

    normal_data = pd.read_csv(
        dataset('linear'),
        usecols=[f'x{i}' for i in range(1, 11)] + ['y'],
    )
    row_num = normal_data.shape[0]
    np.random.seed(0)
    normal_data['x1'] = np.random.randint(0, 2, (row_num,))
    normal_data['x2'] = np.random.randint(0, 5, (row_num,))
    v_float_data = VDataFrame(
        {
            pyu_alice: partition(data=pyu_alice(lambda: normal_data)()),
            pyu_bob: partition(data=pyu_bob(lambda: normal_data.drop("y", axis=1))()),
            pyu_carol: partition(
                data=pyu_carol(lambda: normal_data.drop("y", axis=1))()
            ),
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
            pyu_carol: partition(
                data=pyu_carol(lambda: nan_str_data.drop("y", axis=1))()
            ),
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


@pytest.mark.mpc(parties=3)
def test_binning_nan_chi(prod_env_and_data):
    env, data = prod_env_and_data
    he_binning = VertWoeBinning(env.heu)
    ss_binning = VertWoeBinning(env.spu)
    he_report = he_binning.binning(
        data['v_nan_data'],
        binning_method="chimerge",
        bin_names={env.alice: ["f1", "f3", "f2"], env.bob: ["f1", "f3", "f2"]},
        label_name="y",
        chimerge_target_bins=4,
    )
    ss_report = ss_binning.binning(
        data['v_nan_data'],
        binning_method="chimerge",
        bin_names={env.alice: ["f1", "f3", "f2"], env.bob: ["f1", "f3", "f2"]},
        label_name="y",
        chimerge_target_bins=4,
    )

    assert he_report.keys() == ss_report.keys()
    ss_alice = reveal(ss_report[env.alice])
    he_alice = reveal(he_report[env.alice])
    ss_bob = reveal(ss_report[env.bob])
    he_bob = reveal(he_report[env.bob])
    print("nan chi ss_alice to he_alice")
    print(ss_alice)
    woe_almost_equal(ss_alice, he_alice)
    print("nan chi ss_bob to he_alice")
    print(ss_bob)
    woe_almost_equal(ss_bob, he_alice)
    print("nan chi he_bob to he_alice")
    print(he_bob)
    woe_almost_equal(he_bob, he_alice)


@pytest.mark.mpc(parties=3)
def test_binning_nan(prod_env_and_data):
    env, data = prod_env_and_data
    he_binning = VertWoeBinning(env.heu)
    ss_binning = VertWoeBinning(env.spu)
    vert_binning = VertBinning()

    he_report = he_binning.binning(
        data['v_nan_data'],
        bin_names={env.alice: ["f1", "f3", "f2"], env.bob: ["f1", "f3", "f2"]},
        label_name="y",
        audit_log_path={
            env.alice.party: "alice.audit",
            env.bob.party: "bob.audit",
        },
    )
    ss_report = ss_binning.binning(
        data['v_nan_data'],
        bin_names={env.alice: ["f1", "f3", "f2"], env.bob: ["f1", "f3", "f2"]},
        label_name="y",
    )

    vert_binning_report = vert_binning.binning(
        data['v_nan_data'],
        binning_method="eq_range",
        bin_names={env.alice: ["f1", "f3", "f2"], env.bob: ["f1", "f3", "f2"]},
    )
    assert he_report.keys() == vert_binning_report.keys()
    assert he_report.keys() == ss_report.keys()

    print(reveal(vert_binning_report[env.alice]))

    ss_alice = reveal(ss_report[env.alice])
    he_alice = reveal(he_report[env.alice])
    ss_bob = reveal(ss_report[env.bob])
    he_bob = reveal(he_report[env.bob])
    print("nan ss_alice to he_alice")
    print(ss_alice)
    woe_almost_equal(ss_alice, he_alice)
    print("nan ss_bob to he_alice")
    print(ss_bob)
    woe_almost_equal(ss_bob, he_alice)
    print("nan he_bob to he_alice")
    print(he_bob)
    woe_almost_equal(he_bob, he_alice)

    # audit_log
    with open('alice.audit', 'rb') as f:
        a = pickle.load(f)
    with open('bob.audit', 'rb') as f:
        b = pickle.load(f)

    assert a.size == b.size
    audit_ciphertext_equal(a, b)

    with open('bob.audit.pk.pickle', 'rb') as f:
        pk = pickle.load(f)

    spk = sfd.get(env.heu.sk_keeper.public_key.remote())
    assert str(spk) == str(pk)


@pytest.mark.mpc(parties=3)
def test_binning_normal(prod_env_and_data):
    env, data = prod_env_and_data
    he_binning = VertWoeBinning(env.heu)
    ss_binning = VertWoeBinning(env.spu)

    vert_binning = VertBinning()

    he_report = he_binning.binning(
        data['v_float_data'],
        bin_names={
            env.alice: ["x1", "x2", "x3"],
            env.bob: ["x1", "x2", "x3"],
            env.carol: ["x1", "x2", "x3"],
        },
        label_name="y",
    )
    ss_report = ss_binning.binning(
        data['v_float_data'],
        bin_names={
            env.alice: ["x1", "x2", "x3"],
            env.bob: ["x1", "x2", "x3"],
            env.carol: ["x1", "x2", "x3"],
        },
        label_name="y",
    )

    vert_binning_report = vert_binning.binning(
        data['v_nan_data'],
        binning_method="eq_range",
        bin_names={
            env.alice: ["f1", "f3", "f2"],
            env.bob: ["f1", "f3", "f2"],
            env.carol: ["f1", "f3", "f2"],
        },
    )
    assert he_report.keys() == vert_binning_report.keys()

    print(reveal(vert_binning_report[env.alice]))
    assert he_report.keys() == ss_report.keys()
    ss_alice = reveal(ss_report[env.alice])
    he_alice = reveal(he_report[env.alice])
    ss_bob = reveal(ss_report[env.bob])
    he_bob = reveal(he_report[env.bob])
    ss_carol = reveal(ss_report[env.carol])
    he_carol = reveal(he_report[env.carol])
    woe_almost_equal(ss_alice, he_alice)
    woe_almost_equal(ss_bob, he_alice)
    woe_almost_equal(ss_carol, he_carol)
    woe_almost_equal(he_bob, he_alice)
    woe_almost_equal(he_carol, he_alice)


@pytest.mark.mpc(parties=3)
def test_binning_normal_chimerge(prod_env_and_data):
    env, data = prod_env_and_data
    he_binning = VertWoeBinning(env.heu)
    ss_binning = VertWoeBinning(env.spu)
    he_report = he_binning.binning(
        data['v_float_data'],
        binning_method="chimerge",
        bin_names={env.alice: ["x1", "x2", "x3"], env.bob: ["x1", "x2", "x3"]},
        label_name="y",
    )
    ss_report = ss_binning.binning(
        data['v_float_data'],
        binning_method="chimerge",
        bin_names={env.alice: ["x1", "x2", "x3"], env.bob: ["x1", "x2", "x3"]},
        label_name="y",
    )
    assert he_report.keys() == ss_report.keys()
    ss_alice = reveal(ss_report[env.alice])
    he_alice = reveal(he_report[env.alice])
    ss_bob = reveal(ss_report[env.bob])
    he_bob = reveal(he_report[env.bob])
    print("chi_alice to chi_alice")
    print(ss_alice)
    woe_almost_equal(ss_alice, he_alice)
    print("chi_bob to chi_alice")
    print(ss_bob)
    woe_almost_equal(ss_bob, he_alice)
    print("chi_bob to chi_alice")
    print(he_bob)
    woe_almost_equal(he_bob, he_alice)


@pytest.mark.mpc(parties=3)
def test_binning_normal_eq_range(prod_env_and_data):
    env, data = prod_env_and_data
    he_binning = VertWoeBinning(env.heu)
    ss_binning = VertWoeBinning(env.spu)
    he_report = he_binning.binning(
        data['v_float_data'],
        binning_method="eq_range",
        bin_names={env.alice: ["x1", "x2", "x3"], env.bob: ["x1", "x2", "x3"]},
        label_name="y",
    )
    ss_report = ss_binning.binning(
        data['v_float_data'],
        binning_method="eq_range",
        bin_names={env.alice: ["x1", "x2", "x3"], env.bob: ["x1", "x2", "x3"]},
        label_name="y",
    )
    assert he_report.keys() == ss_report.keys()
    ss_alice = reveal(ss_report[env.alice])
    he_alice = reveal(he_report[env.alice])
    ss_bob = reveal(ss_report[env.bob])
    he_bob = reveal(he_report[env.bob])
    woe_almost_equal(ss_alice, he_alice)
    woe_almost_equal(ss_bob, he_alice)
    woe_almost_equal(he_bob, he_alice)
