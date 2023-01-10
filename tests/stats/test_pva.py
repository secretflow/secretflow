import jax.numpy as jnp
import numpy as np
import pandas as pd

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import pva_eval
from tests.basecase import MultiDriverDeviceTestCase


class TestPVA(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.y_actual_pd_dataframe = pd.DataFrame(
            {
                'y_expected': [*range(10)],
            }
        )
        cls.y_actual = VDataFrame(
            partitions={
                cls.alice: Partition(
                    data=cls.alice(lambda x: x)(cls.y_actual_pd_dataframe)
                ),
            }
        )

        cls.y_prediction = FedNdarray(
            partitions={
                cls.alice: cls.alice(lambda x: x)(
                    jnp.array([0.1 for _ in range(10)]).reshape(-1, 1)
                )
            },
            partition_way=PartitionWay.VERTICAL,
        )
        cls.target = 8

    def test_pva(self):
        score = reveal(pva_eval(self.y_actual, self.y_prediction, self.target))
        np.testing.assert_almost_equal(score, 0.0, decimal=2)
