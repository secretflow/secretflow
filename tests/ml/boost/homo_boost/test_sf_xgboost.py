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

import os
import tempfile

import numpy as np
import pandas as pd

from secretflow.data.horizontal import read_csv as h_read_csv
from secretflow.ml.boost.homo_boost import SFXgboost
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator

_temp_dir = tempfile.mkdtemp()


def gen_data(data_num, feature_num, use_random=True, data_bin_num=10, prefix="x"):
    data = []
    label = []
    header = [prefix + str(i) for i in range(feature_num)]
    index_colname_map = {}
    for index, name in enumerate(header):
        index_colname_map[index] = name
    for data_key in range(data_num):
        value = data_key % data_bin_num

        if not use_random:
            features = value * np.ones(feature_num) * 0.1
            # feature value positive if greater 5,else negative
            if value < 5:
                random_label = 0.0
            else:
                random_label = 1.0

        else:
            # random is used, which can approximatily complete binary split
            features = np.random.random(feature_num)

            random_label = np.random.randint(0, 2)
        data.append(features)

        label.append(random_label)

    data = pd.DataFrame(np.array(data))

    data.rename(columns=index_colname_map, inplace=True)
    data_with_label = data
    data_with_label['label'] = np.array(label)

    return data_with_label


def test_homo_xgboost(sf_production_setup_devices_grpc_ray):
    data_size = 300000
    num_feature = 10
    bin_num = 10

    data1 = gen_data(data_size // 2, num_feature, use_random=True, data_bin_num=bin_num)
    data2 = gen_data(data_size // 2, num_feature, use_random=True, data_bin_num=bin_num)
    dfs = [data1, data2]

    file_uris = {
        sf_production_setup_devices_grpc_ray.alice: f'{_temp_dir}/test_alice.csv',
        sf_production_setup_devices_grpc_ray.bob: f'{_temp_dir}/test_bob.csv',
    }
    for df, file_uri in zip(dfs, file_uris.values()):
        df.to_csv(file_uri, index=False)

    hdf = h_read_csv(
        file_uris,
        aggregator=PlainAggregator(sf_production_setup_devices_grpc_ray.carol),
        comparator=PlainComparator(sf_production_setup_devices_grpc_ray.carol),
    )

    bst = SFXgboost(
        server=sf_production_setup_devices_grpc_ray.davy,
        clients=[
            sf_production_setup_devices_grpc_ray.alice,
            sf_production_setup_devices_grpc_ray.bob,
        ],
    )
    params = {
        'max_depth': 4,
        'eta': 1.0,
        'objective': 'binary:logistic',
        'verbosity': 0,
        'tree_method': 'hist',
        'min_child_weight': 1,
        'lambda': 0.1,
        'alpha': 0,
        'max_bin': 10,
        'colsample_bytree': 1.0,
        'eval_metric': 'logloss',
        'hess_key': 'hess',  # 标记增加的hessian列名
        'grad_key': 'grad',  # 标记增加的grad列名
        'label_key': 'label',  # 标记hdataframe中label列名
    }

    bst.train(hdf, hdf, params=params, num_boost_round=4)
    model_path = {
        sf_production_setup_devices_grpc_ray.alice: "./test_xgboost_alice.json",
        sf_production_setup_devices_grpc_ray.bob: "./test_xgboost_bob.json",
    }
    bst.save_model(model_path)
    for path in model_path.values():
        assert os.path.isfile(path)
    dump_path = {
        sf_production_setup_devices_grpc_ray.alice: "./test_xgboost_alice.dump",
        sf_production_setup_devices_grpc_ray.bob: "./test_xgboost_bob.dump",
    }
    bst.dump_model(dump_path)
    for path in dump_path.values():
        assert os.path.isfile(path)
    result = bst.eval(model_path=model_path, hdata=hdf, params=params)
    print(result)
    bst_ft = SFXgboost(
        server=sf_production_setup_devices_grpc_ray.davy,
        clients=[
            sf_production_setup_devices_grpc_ray.alice,
            sf_production_setup_devices_grpc_ray.bob,
        ],
    )

    bst_ft.train(
        hdf,
        hdf,
        params=params,
        num_boost_round=4,
        xgb_model=model_path,
    )
    for path in model_path.values():
        try:
            os.remove(path)
        except OSError:
            pass
    for path in dump_path.values():
        try:
            os.remove(path)
        except OSError:
            pass


def test_homo_xgboost_cn(sf_production_setup_devices_grpc_ray):
    data_size = 300000
    num_feature = 10
    bin_num = 10

    data1 = gen_data(
        data_size // 2,
        num_feature,
        use_random=True,
        data_bin_num=bin_num,
        prefix="特征",
    )
    data2 = gen_data(
        data_size // 2,
        num_feature,
        use_random=True,
        data_bin_num=bin_num,
        prefix="特征",
    )
    dfs = [data1, data2]

    file_uris = {
        sf_production_setup_devices_grpc_ray.alice: f'{_temp_dir}/test_alice_cn.csv',
        sf_production_setup_devices_grpc_ray.bob: f'{_temp_dir}/test_bob_cn.csv',
    }
    for df, file_uri in zip(dfs, file_uris.values()):
        df.to_csv(file_uri, index=False)

    hdf = h_read_csv(
        file_uris,
        aggregator=PlainAggregator(sf_production_setup_devices_grpc_ray.carol),
        comparator=PlainComparator(sf_production_setup_devices_grpc_ray.carol),
    )

    bst = SFXgboost(
        server=sf_production_setup_devices_grpc_ray.davy,
        clients=[
            sf_production_setup_devices_grpc_ray.alice,
            sf_production_setup_devices_grpc_ray.bob,
        ],
    )
    params = {
        'max_depth': 4,
        'eta': 1.0,
        'objective': 'binary:logistic',
        'verbosity': 0,
        'tree_method': 'hist',
        'min_child_weight': 1,
        'lambda': 0.1,
        'alpha': 0,
        'max_bin': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'hess_key': 'hess',  # 标记增加的hessian列名
        'grad_key': 'grad',  # 标记增加的grad列名
        'label_key': 'label',  # 标记hdataframe中label列名
    }
    bst.train(hdf, hdf, params=params, num_boost_round=2)
    model_path = {
        sf_production_setup_devices_grpc_ray.alice: "./test_xgboost_alice_cn.json",
        sf_production_setup_devices_grpc_ray.bob: "./test_xgboost_bob_cn.json",
    }
    bst.save_model(model_path)
    for path in model_path.values():
        assert os.path.isfile(path)
    dump_path = {
        sf_production_setup_devices_grpc_ray.alice: "./test_xgboost_alice_cn.dump",
        sf_production_setup_devices_grpc_ray.bob: "./test_xgboost_bob_cn.dump",
    }
    bst.dump_model(dump_path)
    for path in dump_path.values():
        assert os.path.isfile(path)
    result = bst.eval(model_path=model_path, hdata=hdf, params=params)
    print(result)
    bst_ft = SFXgboost(
        server=sf_production_setup_devices_grpc_ray.davy,
        clients=[
            sf_production_setup_devices_grpc_ray.alice,
            sf_production_setup_devices_grpc_ray.bob,
        ],
    )

    bst_ft.train(
        hdf,
        hdf,
        params=params,
        num_boost_round=4,
        xgb_model=model_path,
    )
    for path in model_path.values():
        try:
            os.remove(path)
        except OSError:
            pass
    for path in dump_path.values():
        try:
            os.remove(path)
        except OSError:
            pass
