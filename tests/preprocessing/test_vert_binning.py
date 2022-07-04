from secretflow.device.driver import reveal
from tests.basecase import DeviceTestCase
from io import StringIO
import pandas as pd
import numpy as np
import ray

from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.data.base import Partition


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


def audit_ciphertext_equal(a, b):
    def get_c_in_a(s):
        pos = s.find('c:') + len('c:')
        return eval(f"0x{s[pos:]}")

    def get_c_in_b(s):
        pos = s.find('Ciphertext:') + len('Ciphertext:')
        return eval(s[pos:])

    for sa, sb in zip(a, b):
        assert get_c_in_a(sa) == get_c_in_b(sb), f"{sa}\n...........\n{sb}"


class TestVertBinning(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        normal_data = pd.read_csv("tests/datasets/linear/vertical/linear_a.csv")

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

    def test_binning_nan_chi(self):
        he_binning = VertWoeBinning(self.heu)
        ss_binning = VertWoeBinning(self.spu)
        he_report = he_binning.binning(
            self.v_nan_data,
            binning_method="chimerge",
            bin_names={self.alice: ["f1", "f3", "f2"], self.bob: ["f1", "f3", "f2"]},
            label_name="y",
            chimerge_target_bins=4,
        )
        ss_report = ss_binning.binning(
            self.v_nan_data,
            binning_method="chimerge",
            bin_names={self.alice: ["f1", "f3", "f2"], self.bob: ["f1", "f3", "f2"]},
            label_name="y",
            chimerge_target_bins=4,
        )
        assert he_report.keys() == ss_report.keys()
        ss_alice = reveal(ss_report[self.alice])
        he_alice = reveal(he_report[self.alice])
        ss_bob = reveal(ss_report[self.bob])
        he_bob = reveal(he_report[self.bob])
        print("nan chi ss_alice to he_alice")
        print(ss_alice)
        woe_almost_equal(ss_alice, he_alice)
        print("nan chi ss_bob to he_alice")
        print(ss_bob)
        woe_almost_equal(ss_bob, he_alice)
        print("nan chi he_bob to he_alice")
        print(he_bob)
        woe_almost_equal(he_bob, he_alice)

    def test_binning_nan(self):
        he_binning = VertWoeBinning(self.heu)
        ss_binning = VertWoeBinning(self.spu)
        he_report = he_binning.binning(
            self.v_nan_data,
            bin_names={self.alice: ["f1", "f3", "f2"], self.bob: ["f1", "f3", "f2"]},
            label_name="y",
            audit_log_path={
                self.alice.party: "alice.audit",
                self.bob.party: "bob.audit",
            },
        )
        ss_report = ss_binning.binning(
            self.v_nan_data,
            bin_names={self.alice: ["f1", "f3", "f2"], self.bob: ["f1", "f3", "f2"]},
            label_name="y",
        )
        assert he_report.keys() == ss_report.keys()
        ss_alice = reveal(ss_report[self.alice])
        he_alice = reveal(he_report[self.alice])
        ss_bob = reveal(ss_report[self.bob])
        he_bob = reveal(he_report[self.bob])
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
        a = np.load('alice.audit.npy')
        b = np.load('bob.audit.npy')
        assert a.size == b.size
        audit_ciphertext_equal(a, b)
        import cloudpickle as pickle

        with open('bob.audit.pk.pickle', 'rb') as f:
            pk = pickle.load(f)

        spk = ray.get(self.heu.sk_keeper.public_key.remote())
        assert str(spk) == str(pk)

    def test_binning_normal(self):
        he_binning = VertWoeBinning(self.heu)
        ss_binning = VertWoeBinning(self.spu)
        he_report = he_binning.binning(
            self.v_float_data,
            bin_names={self.alice: ["x1", "x2", "x3"], self.bob: ["x1", "x2", "x3"]},
            label_name="y",
        )
        ss_report = ss_binning.binning(
            self.v_float_data,
            bin_names={self.alice: ["x1", "x2", "x3"], self.bob: ["x1", "x2", "x3"]},
            label_name="y",
        )
        assert he_report.keys() == ss_report.keys()
        ss_alice = reveal(ss_report[self.alice])
        he_alice = reveal(he_report[self.alice])
        ss_bob = reveal(ss_report[self.bob])
        he_bob = reveal(he_report[self.bob])
        print("ss_alice to ss_alice")
        print(ss_alice)
        woe_almost_equal(ss_alice, he_alice)
        print("ss_bob to ss_alice")
        print(ss_bob)
        woe_almost_equal(ss_bob, he_alice)
        print("ss_bob to ss_alice")
        print(he_bob)
        woe_almost_equal(he_bob, he_alice)

    def test_binning_normal_chimerge(self):
        he_binning = VertWoeBinning(self.heu)
        ss_binning = VertWoeBinning(self.spu)
        he_report = he_binning.binning(
            self.v_float_data,
            binning_method="chimerge",
            bin_names={self.alice: ["x1", "x2", "x3"], self.bob: ["x1", "x2", "x3"]},
            label_name="y",
        )
        ss_report = ss_binning.binning(
            self.v_float_data,
            binning_method="chimerge",
            bin_names={self.alice: ["x1", "x2", "x3"], self.bob: ["x1", "x2", "x3"]},
            label_name="y",
        )
        assert he_report.keys() == ss_report.keys()
        ss_alice = reveal(ss_report[self.alice])
        he_alice = reveal(he_report[self.alice])
        ss_bob = reveal(ss_report[self.bob])
        he_bob = reveal(he_report[self.bob])
        print("chi_alice to chi_alice")
        print(ss_alice)
        woe_almost_equal(ss_alice, he_alice)
        print("chi_bob to chi_alice")
        print(ss_bob)
        woe_almost_equal(ss_bob, he_alice)
        print("chi_bob to chi_alice")
        print(he_bob)
        woe_almost_equal(he_bob, he_alice)
