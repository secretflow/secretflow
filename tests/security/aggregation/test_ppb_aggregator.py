import pytest

from secretflow.security.aggregation.experiment.ppb_aggregator import PPBAggregator
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class TestSecureAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, PPBAggregator(
            sf_production_setup_devices.carol,
            [sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        )
