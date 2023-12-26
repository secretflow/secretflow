import pytest

from secretflow.security.aggregation.experiment.ldp_aggregator import (
    LDPAggregator,
)
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class TestLDPAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, LDPAggregator(
            sf_production_setup_devices.carol,
        )
