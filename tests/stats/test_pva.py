import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from secretflow import reveal
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame
from secretflow.stats import pva_eval


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    y_actual_pd_dataframe = pd.DataFrame(
        {
            'y_expected': [*range(10)],
        }
    )
    y_actual = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: Partition(
                data=sf_production_setup_devices.alice(lambda x: x)(
                    y_actual_pd_dataframe
                )
            ),
        }
    )

    y_prediction = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda x: x
            )(jnp.array([0.1 for _ in range(10)]).reshape(-1, 1))
        },
        partition_way=PartitionWay.VERTICAL,
    )
    target = 8

    yield sf_production_setup_devices, {
        'y_actual': y_actual,
        "y_prediction": y_prediction,
        "target": target,
    }


def test_pva(prod_env_and_data):
    env, data = prod_env_and_data
    score = reveal(pva_eval(data['y_actual'], data['y_prediction'], data['target']))
    np.testing.assert_almost_equal(score, 0.0, decimal=2)
