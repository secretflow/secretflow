# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import logging

from secretflow.device.driver import reveal, wait
from secretflow.ml.boost.ss_xgb_v import Xgb
from secretflow.data import FedNdarray, PartitionWay
from secretflow.utils.simulation.datasets import (
    load_linear,
    load_dermatology,
    create_df,
    dataset,
)
from tests.basecase import ABY3DeviceTestCase

from sklearn.metrics import roc_auc_score, mean_squared_error


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestVertBinning(ABY3DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def run_xgb(self, test_name, v_data, label_data, y, logistic, subsample, colsample):
        xgb = Xgb(self.spu)
        start = time.time()
        params = {
            'num_boost_round': 3,
            'max_depth': 3,
            'sketch_eps': 0.25,
            'objective': 'logistic' if logistic else 'linear',
            'reg_lambda': 0.1,
            'subsample': subsample,
            'colsample_bytree': colsample,
            'base_score': 0.5,
        }
        model = xgb.train(params, v_data, label_data)
        reveal(model.weights[-1])
        print(f"{test_name} train time: {time.time() - start}")
        start = time.time()
        spu_yhat = model.predict(v_data)
        yhat = reveal(spu_yhat)
        print(f"{test_name} predict time: {time.time() - start}")
        if logistic:
            print(f"{test_name} auc: {roc_auc_score(y, yhat)}")
        else:
            print(f"{test_name} mse: {mean_squared_error(y, yhat)}")

        fed_yhat = model.predict(v_data, self.alice)
        assert len(fed_yhat.partitions) == 1 and self.alice in fed_yhat.partitions
        yhat = reveal(fed_yhat.partitions[self.alice])
        assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
        if logistic:
            print(f"{test_name} auc: {roc_auc_score(y, yhat)}")
        else:
            print(f"{test_name} mse: {mean_squared_error(y, yhat)}")

    def _run_npc_linear(self, test_name, parts, label_device):
        vdf = load_linear(parts=parts)

        label_data = vdf['y']
        y = reveal(label_data.partitions[label_device].data).values
        label_data = (label_data.values)[:500, :]
        y = y[:500, :]

        v_data = vdf.drop(columns="y").values
        v_data = v_data[:500, :]
        label_data = label_data[:500, :]

        self.run_xgb(test_name, v_data, label_data, y, True, 0.9, 1)

    def test_2pc_linear(self):
        parts = {self.alice: (1, 11), self.bob: (11, 22)}
        self._run_npc_linear("2pc_linear", parts, self.bob)

    def test_3pc_linear(self):
        parts = {self.alice: (1, 8), self.bob: (8, 16), self.carol: (16, 22)}
        self._run_npc_linear("3pc_linear", parts, self.carol)

    def test_4pc_linear(self):
        parts = {
            self.alice: (1, 6),
            self.bob: (6, 12),
            self.carol: (12, 18),
            self.davy: (18, 22),
        }
        self._run_npc_linear("4pc_linear", parts, self.davy)

    def test_breast_cancer(self):
        from sklearn.datasets import load_breast_cancer

        ds = load_breast_cancer()
        x, y = ds['data'], ds['target']

        v_data = FedNdarray(
            {
                self.alice: (self.alice(lambda: x[:, :15])()),
                self.bob: (self.bob(lambda: x[:, 15:])()),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        label_data = FedNdarray(
            {self.alice: (self.alice(lambda: y)())},
            partition_way=PartitionWay.VERTICAL,
        )

        self.run_xgb("breast_cancer", v_data, label_data, y, True, 1, 0.9)

    def test_dermatology(self):
        vdf = load_dermatology(parts={self.alice: (0, 17), self.bob: (17, 35)}, axis=1)

        label_data = vdf['class']
        y = reveal(label_data.partitions[self.bob].data).values
        v_data = vdf.drop(columns="class").values
        label_data = label_data.values

        self.run_xgb("dermatology", v_data, label_data, y, False, 0.9, 0.9)


if __name__ == '__main__':
    # HOW TO RUN:
    # 0. change args following <<< !!! >>> flag.
    #    you need change input data path & train settings before run.
    # 1. install requirements following INSTALLATION.md
    # 2. set env
    #    export PYTHONPATH=$PYTHONPATH:bazel-bin
    # 3. run
    #    python tests/ml/boost/ss_xgb_v/test_vert_ss_xgb.py

    # use aby3 in this example.
    cluster = ABY3DeviceTestCase()
    cluster.setUpClass()
    # init log
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # prepare data
    start = time.time()
    # read dataset.
    vdf = create_df(
        # load file 'dataset('linear')' as train dataset.
        # <<< !!! >>> replace dataset path to your own local file.
        dataset('linear'),
        # split 1-10 columns to alice and 11-21 columns to bob which include y col.
        # <<< !!! >>> replace parts range to your own dataset's columns count.
        parts={cluster.alice: (1, 11), cluster.bob: (11, 22)},
        # split by vertical. DON'T change this.
        axis=1,
    )
    # split y out of dataset,
    # <<< !!! >>> change 'y' if label column name is not y in dataset.
    label_data = vdf['y']
    # v_data remains all features.
    v_data = vdf.drop(columns="y")
    # <<< !!! >>> change cluster.bob if y not belong to bob.
    y = reveal(label_data.partitions[cluster.bob].data)
    wait([p.data for p in v_data.partitions.values()])
    logging.info(f"IO times: {time.time() - start}s")

    # run ss xgb
    xgb = Xgb(cluster.spu)
    params = {
        # <<< !!! >>> change args to your test settings.
        # for more detail, see Xgb.train.__doc__
        'num_boost_round': 3,
        'max_depth': 3,
        'learning_rate': 0.3,
        'sketch_eps': 0.25,
        'objective': 'logistic',
        'reg_lambda': 0.1,
        'subsample': 1,
        'colsample_bytree': 1,
        'base_score': 0.5,
    }
    start = time.time()
    model = xgb.train(params, v_data, label_data)
    logging.info(f"main train time: {time.time() - start}")
    start = time.time()
    spu_yhat = model.predict(v_data)
    yhat = reveal(spu_yhat)
    logging.info(f"main predict time: {time.time() - start}")
    logging.info(f"main auc: {roc_auc_score(y, yhat)}")
