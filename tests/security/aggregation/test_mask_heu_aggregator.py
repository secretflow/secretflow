import pytest

from secretflow.security.aggregation.experiment.mask_heu_aggregator import (
    MaskHeuAggregator,
)
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class TestMaskHeuAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, MaskHeuAggregator(
            sf_production_setup_devices.carol,
            [sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        )
