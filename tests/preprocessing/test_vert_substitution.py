from io import StringIO

import numpy as np
import pandas as pd

from secretflow.data.base import Partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.driver import reveal
from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.preprocessing.binning.vert_woe_substitution import VertWOESubstitution
from secretflow.utils.simulation.datasets import dataset
from tests.basecase import MultiDriverDeviceTestCase


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


class TestVertBinning(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        normal_data = pd.read_csv(
            dataset('linear'),
            usecols=['id'] + [f'x{i}' for i in range(1, 11)] + ['y'],
        )

        cls.v_float_data = VDataFrame(
            {
                cls.alice: Partition(data=cls.alice(lambda: normal_data)()),
                cls.bob: Partition(
                    data=cls.bob(lambda: normal_data.drop("y", axis=1))()
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

        cls.v_nan_data = VDataFrame(
            {
                cls.alice: Partition(data=cls.alice(lambda: nan_str_data)()),
                cls.bob: Partition(
                    data=cls.bob(lambda: nan_str_data.drop("y", axis=1))()
                ),
            }
        )

    def test_binning_nan(self):
        ss_binning = VertWoeBinning(self.spu)
        woe_rules = ss_binning.binning(
            self.v_nan_data,
            binning_method="chimerge",
            bin_names={self.alice: ["f1", "f3", "f2"], self.bob: ["f1", "f3", "f2"]},
            label_name="y",
            chimerge_target_bins=4,
        )

        woe_sub = VertWOESubstitution()
        sub_data = woe_sub.substitution(self.v_nan_data, woe_rules)
        alice_data = reveal(sub_data.partitions[self.alice].data).drop("y", axis=1)
        bob_data = reveal(sub_data.partitions[self.bob].data)
        rules = {v['name']: v for v in reveal(woe_rules[self.alice])["variables"]}

        assert alice_data.equals(bob_data), (
            str(alice_data) + "\n,,,,,,\n" + str(bob_data)
        )
        f1_categories = list(set(alice_data['f1']))
        assert np.isin(rules['f1']['woes'], f1_categories).all(), (
            str(rules['f1']['woes']) + "\n,,,,,,\n" + str(f1_categories)
        )
        assert rules['f1']['else_woe'] in f1_categories
        f2_categories = list(set(alice_data['f2']))
        assert np.isin(f2_categories, rules['f2']['woes']).all()
        f3_categories = list(set(alice_data['f3']))
        assert np.isin(rules['f3']['woes'], f3_categories).all()
        assert rules['f3']['else_woe'] in f3_categories

    def test_binning_normal(self):
        ss_binning = VertWoeBinning(self.spu)
        woe_rules = ss_binning.binning(
            self.v_float_data,
            bin_names={self.alice: ["x1", "x2", "x3"], self.bob: ["x1", "x2", "x3"]},
            label_name="y",
        )

        woe_sub = VertWOESubstitution()
        sub_data = woe_sub.substitution(self.v_float_data, woe_rules)
        alice_data = reveal(sub_data.partitions[self.alice].data).drop("y", axis=1)
        bob_data = reveal(sub_data.partitions[self.bob].data)
        rules = {v['name']: v for v in reveal(woe_rules[self.alice])["variables"]}

        assert alice_data.equals(bob_data), (
            str(alice_data) + "\n,,,,,,\n" + str(bob_data)
        )
        f1_categories = list(set(alice_data['x1']))
        assert np.isin(rules['x1']['woes'], f1_categories).all()
        f2_categories = list(set(alice_data['x2']))
        assert np.isin(f2_categories, rules['x2']['woes']).all()
        f3_categories = list(set(alice_data['x3']))
        assert np.isin(rules['x3']['woes'], f3_categories).all()
