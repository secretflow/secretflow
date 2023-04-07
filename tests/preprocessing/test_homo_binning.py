import tempfile

import numpy as np
import pandas as pd

from secretflow.data.horizontal import read_csv as h_read_csv
from secretflow.device import reveal
from secretflow.preprocessing.binning.homo_binning import HomoBinning
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from tests.basecase import MultiDriverDeviceTestCase

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


class TestHomoBinning(MultiDriverDeviceTestCase):

    data1 = gen_data(10000, 10, use_random=False)
    data2 = gen_data(5000, 10, use_random=False)
    dfs = [data1, data2]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        file_uris = {
            cls.alice: f'{_temp_dir}/test_alice.csv',
            cls.bob: f'{_temp_dir}/test_bob.csv',
        }

        for df, file_uri in zip(cls.dfs, file_uris.values()):
            df.to_csv(file_uri)

        cls.hdf = h_read_csv(
            file_uris,
            aggregator=PlainAggregator(cls.carol),
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
