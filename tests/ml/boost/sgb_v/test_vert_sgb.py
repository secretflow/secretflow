# Copyright 2023 Ant Group Co., Ltd.
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
import time

from sklearn.metrics import mean_squared_error, roc_auc_score

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.boost.sgb_v import Sgb
from secretflow.utils.simulation.datasets import load_dermatology, load_linear
from tests.basecase import ABY3MultiDriverDeviceTestCase

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestSGB(ABY3MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def run_sgb(
        self,
        test_name,
        v_data,
        label_data,
        y,
        logistic,
        subsample,
        colsample,
        audit_dict={},
    ):
        sgb = Sgb(self.heu)
        start = time.perf_counter()
        params = {
            'num_boost_round': 2,
            'max_depth': 3,
            'sketch_eps': 0.25,
            'objective': 'logistic' if logistic else 'linear',
            'reg_lambda': 0.1,
            'subsample': subsample,
            'colsample_by_tree': colsample,
            'base_score': 0.5,
        }
        model = sgb.train(params, v_data, label_data, audit_dict)
        reveal(model.trees[-1])
        print(f"{test_name} train time: {time.perf_counter() - start}")
        start = time.perf_counter()
        yhat = model.predict(v_data)
        yhat = reveal(yhat)
        print(f"{test_name} predict time: {time.perf_counter() - start}")
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

        self.run_sgb(test_name, v_data, label_data, y, True, 0.9, 1)

    def test_2pc_linear(self):
        parts = {self.bob: (1, 11), self.alice: (11, 22)}
        self._run_npc_linear("2pc_linear", parts, self.alice)

    def test_3pc_linear(self):
        parts = {self.carol: (1, 8), self.bob: (8, 16), self.alice: (16, 22)}
        self._run_npc_linear("3pc_linear", parts, self.alice)

    def test_4pc_linear(self):
        parts = {
            self.davy: (1, 6),
            self.bob: (6, 12),
            self.carol: (12, 18),
            self.alice: (18, 22),
        }
        self._run_npc_linear("4pc_linear", parts, self.alice)

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

        self.run_sgb("breast_cancer", v_data, label_data, y, True, 1, 0.9)

    def test_dermatology(self):
        vdf = load_dermatology(parts={self.bob: (0, 17), self.alice: (17, 35)}, axis=1)

        label_data = vdf['class']
        y = reveal(label_data.partitions[self.alice].data).values
        v_data = vdf.drop(columns="class").values
        label_data = label_data.values

        audit_dict = {self.alice.party: "./audit_alice", self.bob.party: "./audit_bob"}
        self.run_sgb("dermatology", v_data, label_data, y, False, 0.9, 0.9, audit_dict)
