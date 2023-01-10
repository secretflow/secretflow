import os
import tempfile

import numpy as np
import pandas as pd

from secretflow.data.horizontal import read_csv as h_read_csv
from secretflow.ml.boost.homo_boost import SFXgboost
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator

from tests.basecase import (MultiDriverDeviceTestCase,
                            SingleDriverDeviceTestCase)

_temp_dir = tempfile.mkdtemp()


def gen_data(data_num, feature_num, use_random=True, data_bin_num=10):
    data = []
    label = []
    header = ["x" + str(i) for i in range(feature_num)]
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


class TestHomoXgboost(SingleDriverDeviceTestCase, MultiDriverDeviceTestCase):

    fields = []
    data_size = 300000
    num_feature = 10
    bin_num = 10

    data1 = gen_data(data_size // 2, num_feature, use_random=True, data_bin_num=bin_num)
    data2 = gen_data(data_size // 2, num_feature, use_random=True, data_bin_num=bin_num)
    dfs = [data1, data2]

    @classmethod
    def setUpClass(cls) -> None:
        super(TestHomoXgboost, cls).setUpClass()
        file_uris = {
            cls.alice: f'{_temp_dir}/test_alice.csv',
            cls.bob: f'{_temp_dir}/test_bob.csv',
        }
        for df, file_uri in zip(cls.dfs, file_uris.values()):
            df.to_csv(file_uri, index=False)

        cls.hdf = h_read_csv(
            file_uris,
            aggregator=PlainAggregator(cls.carol),
            comparator=PlainComparator(cls.carol),
        )

    def test_homo_xgboost(self):

        bst = SFXgboost(server=self.davy, clients=[self.alice, self.bob])
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

        bst.train(self.hdf, self.hdf, params=params, num_boost_round=4)
        model_path = {
            self.alice: "./test_xgboost_alice.json",
            self.bob: "./test_xgboost_bob.json",
        }
        bst.save_model(model_path)
        for path in model_path.values():
            self.assertTrue(os.path.isfile(path))
        dump_path = {
            self.alice: "./test_xgboost_alice.dump",
            self.bob: "./test_xgboost_bob.dump",
        }
        bst.dump_model(dump_path)
        for path in dump_path.values():
            self.assertTrue(os.path.isfile(path))
        result = bst.eval(model_path=model_path, hdata=self.hdf, params=params)
        print(result)
        bst_ft = SFXgboost(server=self.davy, clients=[self.alice, self.bob])

        bst_ft.train(
            self.hdf,
            self.hdf,
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
