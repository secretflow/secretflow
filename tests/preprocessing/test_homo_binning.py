import pandas as pd

from secretflow.device import reveal
from secretflow.data.horizontal import read_csv as h_read_csv
from secretflow.preprocessing.binning.homo_binning import HomoBinning
from secretflow.security.aggregation.device_aggregator import DeviceAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from tests.basecase import DeviceTestCase


class TestHomoBinning(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        path_alice = 'tests/datasets/simi_data/horizontal/linear_alice.csv'
        path_bob = 'tests/datasets/simi_data/horizontal/linear_bob.csv'
        cls.hdf_alice = pd.read_csv(path_alice)
        cls.hdf_bob = pd.read_csv(path_bob)
        cls.hdf = h_read_csv(
            {cls.alice: path_alice, cls.bob: path_bob},
            aggregator=DeviceAggregator(cls.carol),
            comparator=PlainComparator(cls.carol),
        )

    def test_homo_binning(self):
        # GIVEN
        bin_obj = HomoBinning(
            bin_num=5,
            bin_indexes=[1, 2, 3, 4],
            error=1e-10,
            max_iter=200,
            compress_thres=30,
        )
        bin_result = bin_obj.fit_split_points(self.hdf)
        bin_result_df = pd.DataFrame.from_dict(reveal(bin_result))

        expect_result = {
            "x0": [2.0, 4.0, 6.0, 8.0, 9.0],
            "x1": [2.0, 4.0, 6.0, 8.0, 9.0],
            "x2": [2.0, 4.0, 6.0, 8.0, 9.0],
            "x3": [2.0, 4.0, 6.0, 8.0, 9.0],
        }
        expect_df = pd.DataFrame.from_dict(expect_result)
        pd.testing.assert_frame_equal(bin_result_df, expect_df, rtol=1e-2)
