import numpy as np
import pytest

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.stats import ScoreCard


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    sc = ScoreCard(20, 600, 20)
    pred1 = np.random.random((10, 1))
    ds1 = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda x: x
            )(pred1)
        },
        partition_way=PartitionWay.VERTICAL,
    )

    alice_pred2 = np.random.random((10, 1))
    bob_pred2 = np.random.random((10, 1))
    ds2 = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda x: x
            )(alice_pred2),
            sf_production_setup_devices.bob: sf_production_setup_devices.bob(
                lambda x: x
            )(bob_pred2),
        },
        partition_way=PartitionWay.HORIZONTAL,
    )

    yield sf_production_setup_devices, {
        'sc': sc,
        'ds1': ds1,
        'ds2': ds2,
        'pred1': pred1,
        'alice_pred2': alice_pred2,
        'bob_pred2': bob_pred2,
    }


def test_sc(prod_env_and_data):
    env, data = prod_env_and_data
    scord = data['sc'].transform(data['ds1'])
    assert scord.shape[0] == 10
    assert len(scord.partitions) == 1
    scord1 = sf.reveal(list(scord.partitions.items())[0])
    print(f"pred \n{data['pred1']}\n -> \n{scord1}")

    scord = data['sc'].transform(data['ds2'])
    assert scord.shape[0] == 20
    assert len(scord.partitions) == 2
    scord2_alice = sf.reveal(list(scord.partitions.items())[0])
    scord2_bob = sf.reveal(list(scord.partitions.items())[1])
    print(f"pred \n{data['alice_pred2']}\n -> \n{scord2_alice}")
    print(f"pred \n{data['bob_pred2']}\n -> \n{scord2_bob}")
