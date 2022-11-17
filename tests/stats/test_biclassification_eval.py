import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import BiClassificationEval
from tests.basecase import DeviceTestCase


class TestBiClassificationReport(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.y_true = np.array([0, 0, 1, 1, 1]).reshape((-1, 1))
        cls.y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.1]).reshape((-1, 1))
        cls.y_pred_jax = jnp.array(cls.y_pred)
        cls.bucket_size = 2
        cls.y_true_pd_dataframe = pd.DataFrame(
            {
                'y_true': [0, 0, 1, 1, 1],
            }
        )

        cls.y_true_fed = VDataFrame(
            partitions={
                cls.alice: Partition(
                    data=cls.alice(lambda x: x)(cls.y_true_pd_dataframe)
                ),
            }
        )

        cls.y_pred_fed = FedNdarray(
            partitions={cls.alice: cls.alice(lambda x: x)(cls.y_pred_jax)},
            partition_way=PartitionWay.VERTICAL,
        )

        biclassification_evaluator = BiClassificationEval(
            cls.y_true_fed, cls.y_pred_fed, cls.bucket_size
        )
        cls.reports = reveal(biclassification_evaluator.get_all_reports())

    def test_auc(self):
        true_score = roc_auc_score(self.y_true, self.y_pred)
        score = float(self.reports.summary_report.auc)
        np.testing.assert_almost_equal(true_score, score, decimal=2)
