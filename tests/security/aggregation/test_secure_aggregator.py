import pytest

from secretflow.security.aggregation.secure_aggregator import SecureAggregator
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class TestSecureAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, SecureAggregator(
            sf_production_setup_devices.carol,
            [sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        )
