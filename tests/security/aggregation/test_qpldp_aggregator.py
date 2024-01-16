import numpy as np
import pytest

import secretflow as sf
from secretflow.security.aggregation.experiment.qpldp_aggregator import QPLDPAggregator
from tests.security.aggregation.test_aggregator_base import AggregatorBase


@pytest.skip('Experimental, not work.', allow_module_level=True)
class TestQPLDPAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, QPLDPAggregator(
            sf_production_setup_devices.carol,
        )

    def test_average_on_list_with_weights_should_ok(
        test_average_on_list_with_weights_should_ok, env_and_aggregator
    ):
        env, aggregator = env_and_aggregator
        # GIVEN
        a = env.alice(
            lambda: [
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[21, 22, 23], [24, 25, 26]]),
            ]
        )()
        b = env.bob(
            lambda: [
                np.array([[11, 12, 13], [14, 15, 16]]),
                np.array([[31, 32, 33], [34, 35, 36]]),
            ]
        )()

        # WHEN
        avg = sf.reveal(aggregator.average([a, b], axis=0, weights=[2, 3]))

        # THEN
        np.testing.assert_almost_equal(
            avg[0], np.array([[7, 8, 9], [10, 11, 12]]), decimal=1
        )
        np.testing.assert_almost_equal(
            avg[1], np.array([[27, 28, 29], [30, 31, 32]]), decimal=1
        )

    def test_average_on_list_without_weights_should_ok(self, env_and_aggregator):
        env, aggregator = env_and_aggregator
        # GIVEN
        a = env.alice(
            lambda: [
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[21, 22, 23], [24, 25, 26]]),
            ]
        )()
        b = env.bob(
            lambda: [
                np.array([[11, 12, 13], [14, 15, 16]]),
                np.array([[31, 32, 33], [34, 35, 36]]),
            ]
        )()

        # WHEN
        avg = sf.reveal(aggregator.average([a, b], axis=0))

        # THEN
        np.testing.assert_almost_equal(
            avg[0], np.array([[6, 7, 8], [9, 10, 11]]), decimal=0
        )
        np.testing.assert_almost_equal(
            avg[1], np.array([[26, 27, 28], [29, 30, 31]]), decimal=0
        )
