import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

from secretflow import reveal
from secretflow.data import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.preprocessing.encoder import VOrdinalEncoder as OrdinalEncoder
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.utils.simulation.datasets import load_iris


@pytest.fixture(scope='module')
def prod_env_and_ordinal_encoder_data(sf_production_setup_devices):
    hdf = load_iris(
        parts=[sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        aggregator=PlainAggregator(sf_production_setup_devices.alice),
        comparator=PlainComparator(sf_production_setup_devices.carol),
    )
    hdf_alice = reveal(hdf.partitions[sf_production_setup_devices.alice].data)
    hdf_bob = reveal(hdf.partitions[sf_production_setup_devices.bob].data)

    vdf_alice = pd.DataFrame(
        {
            'a1': ['K5', 'K1', None, 'K6'],
            'a2': ['A5', 'A5', 'A2', 'A2'],
            'a3': [5, 1, 2, 6],
        }
    )

    vdf_bob = pd.DataFrame(
        {
            'b4': [10.2, 20.5, None, -0.4],
            'b5': ['B3', 'B2', 'B3', 'B4'],
            'b6': [3, 1, 9, 4],
        }
    )
    vdf = VDataFrame(
        {
            sf_production_setup_devices.alice: partition(
                data=sf_production_setup_devices.alice(lambda: vdf_alice)()
            ),
            sf_production_setup_devices.bob: partition(
                data=sf_production_setup_devices.bob(lambda: vdf_bob)()
            ),
        }
    )

    yield sf_production_setup_devices, {
        'hdf': hdf,
        'hdf_alice': hdf_alice,
        'hdf_bob': hdf_bob,
        'vdf_alice': vdf_alice,
        'vdf_bob': vdf_bob,
        'vdf': vdf,
    }


class TestOrdinalEncoder:
    def test_on_vdataframe_should_ok(self, prod_env_and_ordinal_encoder_data):
        env, data = prod_env_and_ordinal_encoder_data
        # GIVEN
        encoder = OrdinalEncoder()
        # WHEN
        value = encoder.fit_transform(data['vdf']['a2'])
        # THEN
        sk_encoder = SkOrdinalEncoder()
        expect_alice = sk_encoder.fit_transform(data['vdf_alice'][['a2']])[np.newaxis].T
        np.testing.assert_array_almost_equal(
            reveal(value.partitions[env.alice].data), expect_alice.reshape(4, 1)
        )
