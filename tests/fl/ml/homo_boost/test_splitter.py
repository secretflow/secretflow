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
from numpy.lib.histograms import histogram

from secretflow_fl.ml.boost.homo_boost.tree_core.splitter import Splitter


def gen_histogram(data_size, feature_num, use_random=False, data_bin_num=10):
    header = ["x" + str(i) for i in range(feature_num)]
    index_colname_map = {}
    valid_features = {}
    for index, name in enumerate(header):
        index_colname_map[index] = name
        valid_features[index] = True
    single_histogram = []
    # use missing_bin设置为1，否则设置为0
    missing_bin = 0

    # 创建 bin_split_points
    bin_split_points = []
    for _ in header:
        if use_random:
            bin_split_points.append(np.linspace(0.0, 1.0, data_bin_num + 1)[1:])
        else:
            bin_split_points.append(
                np.linspace(0.0, data_bin_num, data_bin_num + 1)[1:]
            )
    bin_split_points = np.array(bin_split_points)
    # 创建测试数据集
    data = []
    shift_iter = 0
    for data_key in range(data_size):
        value = data_key % data_bin_num
        if value == 0:
            if shift_iter % data_bin_num == 0:
                value = data_bin_num - 1
            shift_iter += 1
        if not use_random:
            features = value * np.ones(feature_num)
        else:
            features = np.random.random(feature_num)
        data.append(features)
    data = np.array(data)

    # 根据生成的数据，生成histogram
    for fid, _ in enumerate(header):
        f_histogram = []
        if valid_features is not None and valid_features[fid] is False:
            single_histogram.append(f_histogram)
            continue
        else:
            f_data = sorted(data[:, fid])
            hist, bin_edge = histogram(f_data, bins=bin_split_points[fid])
            for j in range(bin_split_points[fid].shape[0] + missing_bin):
                sum_of_grad = np.sum(f_data[: hist[:j].sum()])
                sum_of_hess = np.sum(np.sqrt(f_data[: hist[:j].sum()]))
                count = hist[:j].sum()
                f_histogram.append([sum_of_grad, sum_of_hess, count])
            single_histogram.append(f_histogram)
    return single_histogram, valid_features


def test_splitter():
    splitter = Splitter(
        criterion_method="xgboost",
        criterion_params=[0.1, 0, 15],
        min_impurity_split=1e-2,
        min_sample_split=2,
        min_leaf_node=1,
        min_child_weight=1,
    )

    node_histograms = []
    for i in range(3):
        single_histogram, valid_features = gen_histogram(
            1000, 10, use_random=False, data_bin_num=10
        )
        node_histograms.append(single_histogram)
    tree_node_splitinfo = splitter.find_split(
        node_histograms, valid_features=valid_features
    )
    expect_split_dict = {
        "best_fid": 0,
        "best_bid": 5,
        "sum_grad": 1050.0,
        "sum_hess": 636.9871167691952,
        "gain": 352.34775031175195,
        "missing_dir": 1,
        "sample_count": 500,
    }
    splitinfo = tree_node_splitinfo[0]
    for key in expect_split_dict:
        np.testing.assert_almost_equal(
            expect_split_dict[key], getattr(splitinfo, key), decimal=6
        )
