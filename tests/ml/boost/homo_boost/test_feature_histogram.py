import unittest

import numpy as np
import pandas as pd

from secretflow.ml.boost.homo_boost.tree_core.feature_histogram import (
    FeatureHistogram,
    HistogramBag,
)


def gen_data(data_num, feature_num, use_random=False, data_bin_num=10):
    data = []
    header = ["x" + str(i) for i in range(feature_num)]
    index_colname_map = {}
    for index, name in enumerate(header):
        index_colname_map[index] = name
    for data_key in range(data_num):
        value = data_key % data_bin_num
        if not use_random:
            features = value * np.ones(feature_num)
        else:
            features = np.random.random(feature_num)
        data.append(features)

    data = np.array(data)
    data = pd.DataFrame(data)
    data.rename(columns=index_colname_map, inplace=True)
    return data


class TestFeatureHistogram(unittest.TestCase):
    def setUp(self):
        self.feature_histogram = FeatureHistogram()

        self.node_map = {0: 0, 1: 1, 2: 2, 3: 3}

        # dataset 设置
        self.sample_num = 1000
        self.feature_num = 10
        self.data_bin_num = 10
        self.use_random = False
        self.header = ["x" + str(i) for i in range(self.feature_num)]

        # 单测固定下来grad 和 hess 列表
        grad = [1.0 for i in range(1000)]
        hess = [1.0 for i in range(1000)]

        data_frame_list = []
        for node in range(len(self.node_map)):
            t_df = gen_data(
                self.sample_num,
                self.feature_num,
                use_random=self.use_random,
                data_bin_num=self.data_bin_num,
            )
            t_df['hess'] = hess
            t_df['grad'] = grad
            data_frame_list.append(t_df)
        self.data_frame_list = data_frame_list
        self.valid_feature = {}
        for col in range(len(self.header)):
            self.valid_feature[col] = True
        # 创建 bin_split_points
        split_point_list = None
        if self.use_random:
            split_point_list = np.linspace(0.0, 1.0, self.data_bin_num + 1)[1:]
        else:
            split_point_list = np.linspace(
                0.0, self.data_bin_num - 1, self.data_bin_num
            )
        self.bin_split_points = np.array(
            [split_point_list for i in range(len(self.header))]
        )

    def test_calculate_histogram(self):
        histograms = self.feature_histogram.calculate_histogram(
            self.data_frame_list,
            self.bin_split_points,
            self.valid_feature,
            use_missing=False,
            grad_key="grad",
            hess_key="hess",
        )
        # histogram参考xgboost实现改为小于threshold
        expect_zero_histogram = [
            [
                [[j * 100 for i in range(3)] for j in range(self.data_bin_num)]
                for k in range(self.feature_num)
            ]
            for r in range(len(self.node_map))
        ]

        np_histograms = np.array(histograms)
        np_expect_histogram = np.array(expect_zero_histogram, dtype=np.float64)
        np.testing.assert_array_equal(np_histograms, np_expect_histogram)

    def test_histogram_bag(self):
        histograms = self.feature_histogram.calculate_histogram(
            self.data_frame_list,
            self.bin_split_points,
            valid_features=self.valid_feature,
            use_missing=False,
            grad_key="grad",
            hess_key="hess",
        )
        histogram_bags = []
        for node_id in self.node_map:
            histogram_bag = HistogramBag(histograms[node_id], node_id, -1)
            histogram_bags.append(histogram_bag)

        expect_sum_histogram = np.array(
            [
                [[j * 200 for i in range(3)] for j in range(self.data_bin_num)]
                for k in range(self.feature_num)
            ]
        )

        expect_zero_histogram = np.array(
            [
                [[0.0 for i in range(3)] for j in range(self.data_bin_num)]
                for k in range(self.feature_num)
            ]
        )

        # test for sum
        histogram_sum = histogram_bags[0] + histogram_bags[1]
        np.testing.assert_array_equal(expect_sum_histogram, np.array(histogram_sum))
        # test for sub
        histogram_sub = histogram_bags[0] - histogram_bags[1]
        np.testing.assert_array_equal(expect_zero_histogram, np.array(histogram_sub))
        # test for len
        histogram_len = len(histogram_bag[0])
        np.testing.assert_equal(histogram_len, 10)


if __name__ == '__main__':
    unittest.main()
