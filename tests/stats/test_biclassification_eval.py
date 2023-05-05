import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import BiClassificationEval


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    y_true = np.array([0, 0, 1, 1, 1]).reshape((-1, 1))
    y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.1]).reshape((-1, 1))
    y_pred_jax = jnp.array(y_pred)
    bucket_size = 2
    y_true_pd_dataframe = pd.DataFrame(
        {
            'y_true': [0, 0, 1, 1, 1],
        }
    )

    y_true_fed = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: Partition(
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

    yield sf_production_setup_devices, {
        "y_true": y_true,
        "y_pred": y_pred,
        "reports": reports,
    }


def test_auc(prod_env_and_data):
    env, data = prod_env_and_data
    true_score = roc_auc_score(data['y_true'], data['y_pred'])
    score = float(data['reports'].summary_report.auc)
    np.testing.assert_almost_equal(true_score, score, decimal=2)
