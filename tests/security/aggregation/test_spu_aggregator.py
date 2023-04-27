import pytest

from secretflow.security.aggregation.spu_aggregator import SPUAggregator
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class TestSPUAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, SPUAggregator(
            sf_production_setup_devices.spu
        )
