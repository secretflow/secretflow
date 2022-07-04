import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from secretflow.data.vertical import read_csv
from secretflow.preprocessing.scaler import StandardScaler
from secretflow.stats import VertPearsonR

from tests.basecase import DeviceTestCase


class TestVertPearsonR(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.alice_path = "tests/datasets/linear/vertical/linear_a.csv"
        cls.bob_path = "tests/datasets/linear/vertical/linear_b.csv"
        cls.vdata = read_csv(
            {cls.alice: cls.alice_path, cls.bob: cls.bob_path},
            dtypes={
                cls.alice: {"x%d" % i: np.float32 for i in range(1, 11)},
                cls.bob: {"x%d" % i: np.float32 for i in range(11, 21)},
            },
        )

    def scipy_pearsonr(self):
        ret = np.ones((20, 20))
        data = pd.concat(
            [
                pd.read_csv(self.alice_path).drop(['id1', 'y'], axis=1),
                pd.read_csv(self.bob_path).drop(['id2'], axis=1),
            ],
            axis=1,
        )

        for i in range(20):
            for j in range(i, 20):
                if i == j:
                    ret[i, i] = 1
                else:
                    p = pearsonr(data["x%d" % (i + 1)], data["x%d" % (j + 1)])
                    ret[i, j] = p[0]
                    ret[j, i] = p[0]

        return ret

    def test_pearsonr(self):
        v_pearsonr = VertPearsonR(self.spu)
        data = self.vdata
        scaler = StandardScaler()
        std_data = scaler.fit_transform(data)
        ss_pearsonr_1 = v_pearsonr.pearsonr(data)
        ss_pearsonr_2 = v_pearsonr.pearsonr(std_data, False)
        scipy_pearsonr = self.scipy_pearsonr()
        np.testing.assert_almost_equal(ss_pearsonr_1, scipy_pearsonr, decimal=2)
        np.testing.assert_almost_equal(ss_pearsonr_2, scipy_pearsonr, decimal=2)
