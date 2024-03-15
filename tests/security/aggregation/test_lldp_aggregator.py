import pytest

from secretflow.security.aggregation.experiment.lldp_aggregator import LLDPAggregator
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class TestLLDPAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, LLDPAggregator(
            sf_production_setup_devices.carol,
        )
