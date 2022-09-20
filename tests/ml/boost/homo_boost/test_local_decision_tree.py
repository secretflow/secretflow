import os
import unittest

import numpy as np
import pandas as pd
import xgboost as xgb

from secretflow.ml.boost.homo_boost.tree_core.decision_tree import DecisionTree
from secretflow.ml.boost.homo_boost.tree_core.loss_function import LossFunction
from secretflow.ml.boost.homo_boost.tree_param import TreeParam


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
            # feature value 大于5，设为正样本，小于等于5设为负样本
            if value > 5:
                random_label = 1.0
            else:
                random_label = 0.0

        else:
            features = np.random.random(feature_num)
            random_label = np.round(np.random.random())
        data.append(features)

        label.append(random_label)

    data = pd.DataFrame(np.array(data))
    data['label'] = np.array(label)
    data.rename(columns=index_colname_map, inplace=True)
    return data


class TestFeatureHistogram(unittest.TestCase):
    def setUp(self):
        # dataset 设置
        self.sample_num = 10000
        self.feature_num = 10
        self.data_bin_num = 10
        self.use_random = False
        self.header = ["x" + str(i) for i in range(self.feature_num)]

        self.data = gen_data(
            self.sample_num,
            self.feature_num,
            use_random=self.use_random,
            data_bin_num=self.data_bin_num,
        )

        # 创建 bin_split_points
        bin_split_points = []
        valid_features = {}
        for fid in range(len(self.header)):
            valid_features[fid] = True
            if self.use_random:
                bin_split_points.append(
                    np.linspace(0.0, 1.0, self.data_bin_num + 1)[1:]
                )
            else:
                bin_split_points.append(
                    np.linspace(0.0, 0.1 * (self.data_bin_num - 1), self.data_bin_num)
                )
        self.valid_features = valid_features
        self.bin_split_points = np.array(bin_split_points)

        # prepare xgboost train DMatrix
        label = self.data['label']
        data = self.data.drop(columns=['label'])

        self.dTrain = xgb.DMatrix(data, label)

        # prepare xgboost test DMatrix
        test_data = gen_data(
            10, self.feature_num, use_random=False, data_bin_num=self.data_bin_num
        )
        self.dTest = xgb.DMatrix(test_data.drop(columns=['label']))

    def tearDown(self) -> None:
        model_file_list = ["temp.json", "temp.dump", "xgb_model.json", "xgb_model.dump"]
        for filename in model_file_list:
            if os.path.isfile(filename):
                os.remove(filename)

    def test_local_build_tree(self):
        self.data = self.data.drop(columns=['label'])

        param = {
            'max_depth': 4,
            'eta': 1.0,
            'objective': 'binary:logistic',
            'verbosity': 0,
            'tree_method': 'hist',
            'min_child_weight': 1,
            'lambda': 0.1,
            'alpha': 0,
            'max_bin': 10,
            'gamma': 0,
        }
        bst = xgb.Booster(param, [self.dTrain])

        xgboost_pred = bst.predict(self.dTrain, output_margin=True, training=True)
        # 把xgboost计算出来的grad和hess附在 dataframe上
        obj_func = LossFunction(param['objective']).obj_function()
        self.data['grad'], self.data['hess'] = obj_func(xgboost_pred, self.dTrain)

        tree_param = TreeParam(
            max_depth=4,
            eta=1.0,
            objective='binary:logistic',
            verbosity=0,
            tree_method='hist',
            reg_lambda=0.1,
            reg_alpha=0,
            gamma=0,
            colsample_bytree=1.0,
        )

        decision_tree = DecisionTree(
            tree_param=tree_param,
            data=self.data,
            bin_split_points=self.bin_split_points,
            tree_id=0,
            group_id=0,
            iter_round=0,
            grad_key="grad",
            hess_key="hess",
            label_key="label",
        )
        decision_tree.fit()
        tree_nodes = decision_tree.tree_node
        # 第一次调用时候需要先init model，后续迭代可以直接修改model
        decision_tree.init_xgboost_model("temp.json")
        # 将本次计算得到的树模型更新到xgboost模型中
        decision_tree.save_xgboost_model("temp.json", tree_nodes)

        bst.load_model('temp.json')  # load data
        bst.dump_model("temp.dump")

        ypred = bst.predict(self.dTest, training=True, output_margin=False)

        # 用xgboost训练模型
        xgb_bst = xgb.train(param, self.dTrain, num_boost_round=1, obj=obj_func)
        xgb_result = xgb_bst.predict(self.dTest, output_margin=False)
        xgb_bst.save_model("xgb_model.json")
        xgb_bst.dump_model("xgb_model.dump")

        # 比较xgb和federate模型的预测结果是否一致
        np.testing.assert_array_equal(ypred, xgb_result)


if __name__ == '__main__':
    unittest.main()
