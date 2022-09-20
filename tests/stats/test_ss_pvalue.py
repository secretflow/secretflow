# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
import math
from secretflow.ml.linear import LinearModel, RegType, SSRegression
from secretflow.data.vertical import VDataFrame
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from secretflow.utils.sigmoid import SigType

from secretflow.stats import SSPValue
from secretflow.data.base import Partition

from tests.basecase import DeviceTestCase

from secretflow.utils.simulation.datasets import dataset

from sklearn import linear_model
import scipy.stats as stat


class TestVertPvalue(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        pyus_dict = {cls.alice: 1, cls.bob: 1, cls.carol: 1, cls.davy: 1}
        cls.pyus = list(pyus_dict.keys())

    def _build_splited_ds(self, x, y, parties):
        assert x.shape[1] >= parties
        assert len(self.pyus) >= parties
        step = math.ceil(x.shape[1] / parties)
        fed_x = VDataFrame({})
        for r in range(parties):
            start = r * step
            end = start + step if r != parties - 1 else x.shape[1]
            split_x = x[:, start:end]
            pyu_x = self.pyus[r](lambda: pd.DataFrame(split_x))()
            fed_x.partitions[self.pyus[r]] = Partition(data=pyu_x)
        pyu_y = self.pyus[parties - 1](lambda: pd.DataFrame(y))()
        fed_y = VDataFrame({self.pyus[parties - 1]: Partition(data=pyu_y)})
        return fed_x, fed_y

    def _run_ss(self, x, y, p, w, parties, reg: RegType):
        # weights to spu
        pyu_w = self.alice(lambda: np.array(w))()
        spu_w = pyu_w.to(self.spu)

        # x,y to pyu
        pyu_x, pyu_y = self._build_splited_ds(x, y, parties)
        spu_model = LinearModel(spu_w, reg, SigType.T1)

        sspv = SSPValue(self.spu)
        pvalues = sspv.pvalues(pyu_x, pyu_y, spu_model)
        p = np.array(p)
        abs_err = np.absolute(pvalues - p)
        radio_err = abs_err / np.maximum(pvalues, p)

        # for pvalue < 0.2, check abs err < 0.01
        abs_assert = np.select([p < 0.2], [abs_err], 0)
        assert np.amax(abs_assert) < 0.01, f"\n{abs_assert}"
        # else check radio error < 20%
        radio_assert = np.select([p >= 0.2], [radio_err], 0)
        assert np.amax(radio_assert) < 0.2, f"\n{radio_err}"

        print(f"pvalues\n{pvalues}\np\n{p}\nabs_err\n{abs_err}\n")

    def _run_test(self, x, y, reg: RegType):
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        ones_x = sm.add_constant(x)

        if reg == RegType.Linear:
            sm_model = sm.OLS(y, ones_x).fit()
            weights = list(sm_model.params)
            pvalues = list(sm_model.pvalues)
        else:
            # breast_cancer & linear dataset not converged using sm.Logit
            # not sure WHY, use sklearn instead.
            sk_model = linear_model.LogisticRegression()
            sk_model.fit(x, y)
            weights = [sk_model.intercept_[0]]
            weights.extend(list(sk_model.coef_[0]))
            denom = 2.0 * (1.0 + np.cosh(sk_model.decision_function(x)))
            denom = np.tile(denom, (ones_x.shape[1], 1)).T
            F_ij = np.dot((ones_x / denom).T, ones_x)
            Cramer_Rao = np.linalg.inv(F_ij)
            sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
            z_scores = weights / sigma_estimates
            pvalues = [stat.norm.sf(abs(x)) * 2 for x in z_scores]

        bias = weights.pop(0)
        weights.append(bias)
        bias = pvalues.pop(0)
        pvalues.append(bias)
        self._run_ss(x, y, pvalues, weights, 2, reg)
        self._run_ss(x, y, pvalues, weights, 3, reg)

    def test_linear_ds(self):
        ds = pd.read_csv(dataset('linear'))
        y = ds['y'].values
        x = ds.drop(['y', 'id'], axis=1).values

        self._run_test(x, y, RegType.Logistic)
        self._run_test(x, y, RegType.Linear)

    def test_breast_cancer_ds(self):
        from sklearn.datasets import load_breast_cancer

        ds = load_breast_cancer()
        x, y = ds['data'], ds['target']

        self._run_test(x, y, RegType.Logistic)
        self._run_test(x, y, RegType.Linear)

    def test_ss_lr_logistic(self):
        ds = pd.read_csv(dataset('linear'))
        y = ds['y'].values
        x = ds.drop(['y', 'id'], axis=1).values
        x, y = self._build_splited_ds(x, y, 2)

        sslr = SSRegression(self.spu)
        sslr.fit(x, y, 3, 0.3, 128, 't1', 'logistic', 'l2', 0.5)
        model = sslr.save_model()
        sspv = SSPValue(self.spu)
        pvalues = sspv.pvalues(x, y, model)
        print(f" test_ss_lr_logistic {pvalues}\n")

    def test_ss_lr_linear(self):
        ds = pd.read_csv(dataset('linear'))
        y = ds['y'].values
        x = ds.drop(['y', 'id'], axis=1).values
        x, y = self._build_splited_ds(x, y, 2)

        sslr = SSRegression(self.spu)
        sslr.fit(x, y, 3, 0.3, 128, 't1', 'linear', 'l2', 0.5)
        model = sslr.save_model()
        sspv = SSPValue(self.spu)
        pvalues = sspv.pvalues(x, y, model)
        print(f" test_ss_lr_linear {pvalues}\n")
