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
import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay, partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import BiClassificationEval


@pytest.mark.mpc
def test_auc(sf_production_setup_devices):
    np.random.seed(42)
    y_true = np.round(np.random.random((800000,)).reshape((-1, 1)))
    y_pred = np.random.random((800000,)).reshape((-1, 1))
    y_pred_jax = jnp.array(y_pred)
    bucket_size = 2
    y_true_pd_dataframe = pd.DataFrame(
        {
            'y_true': y_true.reshape(-1),
        }
    )

    y_true_fed = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: partition(
                data=sf_production_setup_devices.alice(lambda x: x)(y_true_pd_dataframe)
            ),
        }
    )

    y_pred_fed = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda x: x
            )(y_pred_jax)
        },
        partition_way=PartitionWay.VERTICAL,
    )

    biclassification_evaluator = BiClassificationEval(
        y_true_fed, y_pred_fed, bucket_size
    )
    reports = reveal(biclassification_evaluator.get_all_reports())
    true_score = roc_auc_score(y_true, y_pred)
    score = float(reports.summary_report.auc)
    np.testing.assert_almost_equal(true_score, score, decimal=2)


@pytest.mark.mpc
def test_auc_nan(sf_production_setup_devices):
    np.random.seed(42)
    y_true = np.round(np.random.random((8000,)).reshape((-1, 1)))
    y_pred = np.random.random((8000,)).reshape((-1, 1)) + np.nan
    y_pred_jax = jnp.array(y_pred)
    bucket_size = 2
    y_true_pd_dataframe = pd.DataFrame(
        {
            'y_true': y_true.reshape(-1),
        }
    )

    y_true_fed = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: partition(
                data=sf_production_setup_devices.alice(lambda x: x)(y_true_pd_dataframe)
            ),
        }
    )

    y_pred_fed = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda x: x
            )(y_pred_jax)
        },
        partition_way=PartitionWay.VERTICAL,
    )

    biclassification_evaluator = BiClassificationEval(
        y_true_fed, y_pred_fed, bucket_size
    )
    reports = reveal(biclassification_evaluator.get_all_reports())
    score = float(reports.summary_report.auc)
    logging.info('score: %s', score)
