import pytest

from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.aggregation.sparse_plain_aggregator import (
    SparsePlainAggregator,
)
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class TestPlainAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, PlainAggregator(
            sf_production_setup_devices.carol
        )


class TestSparsePlainAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, SparsePlainAggregator(
            sf_production_setup_devices.carol
        )
