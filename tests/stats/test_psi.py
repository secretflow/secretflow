import jax.numpy as jnp
import numpy as np
import pandas as pd

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import psi_eval
from secretflow.stats.core.utils import equal_range
from tests.basecase import DeviceTestCase


class TestPSI(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.y_actual = FedNdarray(
            partitions={
                cls.alice: cls.alice(lambda x: x)(
                    jnp.array([*range(10)]).reshape(-1, 1)
                )
            },
            partition_way=PartitionWay.VERTICAL,
        )
        cls.y_expected_1 = FedNdarray(
            partitions={
                cls.alice: cls.alice(lambda x: x)(
                    jnp.array([*range(10)]).reshape(-1, 1)
                )
            },
            partition_way=PartitionWay.VERTICAL,
        )

        cls.y_expected_2_pd_dataframe = pd.DataFrame(
            {
                'y_expected': [0, 0, 0, 0, 0, 2, 3, 4, 5, 6],
            }
        )

        cls.y_expected_2 = VDataFrame(
            partitions={
                cls.alice: Partition(
                    data=cls.alice(lambda x: x)(cls.y_expected_2_pd_dataframe)
                ),
            }
        )

        cls.split_points = equal_range(jnp.array([*range(10)]), 2)

    def test_psi(self):
        score_1 = reveal(psi_eval(self.y_actual, self.y_expected_1, self.split_points))
        score_2 = reveal(psi_eval(self.y_actual, self.y_expected_2, self.split_points))
        true_score_2 = (0.5 - 0.8) * np.log(0.5 / 0.8) + (0.5 - 0.2) * np.log(0.5 / 0.2)
        np.testing.assert_almost_equal(score_1, 0.0, decimal=2)
        np.testing.assert_almost_equal(score_2, true_score_2, decimal=2)
