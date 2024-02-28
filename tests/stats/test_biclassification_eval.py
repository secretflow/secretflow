import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay, partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import BiClassificationEval


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
