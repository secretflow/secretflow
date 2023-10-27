import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import psi_eval
from secretflow.stats.core.utils import equal_range


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    y_actual = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda x: x
            )(jnp.array([*range(10)]).reshape(-1, 1))
        },
        partition_way=PartitionWay.VERTICAL,
    )
    y_expected_1 = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda x: x
            )(jnp.array([*range(10)]).reshape(-1, 1))
        },
        partition_way=PartitionWay.VERTICAL,
    )

    y_expected_2_pd_dataframe = pd.DataFrame(
        {
            'y_expected': [0, 0, 0, 0, 0, 2, 3, 4, 5, 6],
        }
    )

    y_expected_2 = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: partition(
                data=sf_production_setup_devices.alice(lambda x: x)(
                    y_expected_2_pd_dataframe
                )
            ),
        }
    )

    split_points = equal_range(jnp.array([*range(10)]), 2)

    yield sf_production_setup_devices, {
        "y_actual": y_actual,
        "y_expected_1": y_expected_1,
        "split_points": split_points,
        "y_expected_2": y_expected_2,
    }


def test_psi(prod_env_and_data):
    env, data = prod_env_and_data
    score_1 = reveal(
        psi_eval(data['y_actual'], data['y_expected_1'], data['split_points'])
    )
    score_2 = reveal(
        psi_eval(data['y_actual'], data['y_expected_2'], data['split_points'])
    )
    true_score_2 = (0.5 - 0.8) * np.log(0.5 / 0.8) + (0.5 - 0.2) * np.log(0.5 / 0.2)
    np.testing.assert_almost_equal(score_1, 0.0, decimal=2)
    np.testing.assert_almost_equal(score_2, true_score_2, decimal=2)
