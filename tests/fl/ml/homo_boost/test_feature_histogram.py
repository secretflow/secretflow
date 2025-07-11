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

import numpy as np
import pandas as pd
import pytest

from secretflow_fl.ml.boost.homo_boost.tree_core.feature_histogram import (
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


class TestFeatureHistogram:
    @pytest.fixture()
    def set_up(self):
        feature_histogram = FeatureHistogram()

        node_map = {0: 0, 1: 1, 2: 2, 3: 3}

        # dataset 设置
        sample_num = 1000
        feature_num = 10
        data_bin_num = 10
        use_random = False
        header = ["x" + str(i) for i in range(feature_num)]

        # 单测固定下来grad 和 hess 列表
        grad = [1.0 for i in range(1000)]
        hess = [1.0 for i in range(1000)]

        data_frame_list = []
        for node in range(len(node_map)):
            t_df = gen_data(
                sample_num,
                feature_num,
                use_random=use_random,
                data_bin_num=data_bin_num,
            )
            t_df["hess"] = hess
            t_df["grad"] = grad
            data_frame_list.append(t_df)
        valid_feature = {}
        for col in range(len(header)):
            valid_feature[col] = True
        # 创建 bin_split_points
        split_point_list = None
        if use_random:
            split_point_list = np.linspace(0.0, 1.0, data_bin_num + 1)[1:]
        else:
            split_point_list = np.linspace(0.0, data_bin_num - 1, data_bin_num)
        bin_split_points = np.array([split_point_list for i in range(len(header))])

        yield {
            "feature_histogram": feature_histogram,
            "data_frame_list": data_frame_list,
            "bin_split_points": bin_split_points,
            "valid_feature": valid_feature,
            "data_bin_num": data_bin_num,
            "feature_num": feature_num,
            "node_map": node_map,
        }

    def test_calculate_histogram(self, set_up):
        histograms = set_up["feature_histogram"].calculate_histogram(
            set_up["data_frame_list"],
            set_up["bin_split_points"],
            set_up["valid_feature"],
            use_missing=False,
            grad_key="grad",
            hess_key="hess",
        )
        # histogram参考xgboost实现改为小于threshold
        expect_zero_histogram = [
            [
                [[j * 100 for i in range(3)] for j in range(set_up["data_bin_num"])]
                for k in range(set_up["feature_num"])
            ]
            for r in range(len(set_up["node_map"]))
        ]

        np_histograms = np.array(histograms)
        np_expect_histogram = np.array(expect_zero_histogram, dtype=np.float64)
        np.testing.assert_array_equal(np_histograms, np_expect_histogram)

    def test_histogram_bag(self, set_up):
        histograms = set_up["feature_histogram"].calculate_histogram(
            set_up["data_frame_list"],
            set_up["bin_split_points"],
            set_up["valid_feature"],
            use_missing=False,
            grad_key="grad",
            hess_key="hess",
        )
        histogram_bags = []
        for node_id in set_up["node_map"]:
            histogram_bag = HistogramBag(histograms[node_id], node_id, -1)
            histogram_bags.append(histogram_bag)

        expect_sum_histogram = np.array(
            [
                [[j * 200 for i in range(3)] for j in range(set_up["data_bin_num"])]
                for k in range(set_up["feature_num"])
            ]
        )

        expect_zero_histogram = np.array(
            [
                [[0.0 for i in range(3)] for j in range(set_up["data_bin_num"])]
                for k in range(set_up["feature_num"])
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
