import pytest

from secretflow.security.aggregation.experiment.qpldp_aggregator import (
    QPLDPAggregator,
)
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class TestQPLDPAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, QPLDPAggregator(
            sf_production_setup_devices.carol,
        )
