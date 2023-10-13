import sys

sys.path.append("../../../..")
import pytest

from secretflow.security.aggregation.experiment.mask_heu_aggregation import (
    mask_heu_aggregator,
)
from tests.security.aggregation.test_aggregator_base import AggregatorBase


class Testmask_heu_aggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, mask_heu_aggregator(
            sf_production_setup_devices.carol,
            [sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        )
