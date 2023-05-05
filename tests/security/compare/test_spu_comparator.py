import pytest

from secretflow.security.compare.spu_comparator import SPUComparator
from tests.security.compare.test_comparator_base import ComparatorBase


class TestSPUComparator(ComparatorBase):
    @pytest.fixture()
    def env_and_comparator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, SPUComparator(
            sf_production_setup_devices.spu
        )
