from secretflow.security.compare.plain_comparator import PlainComparator
from tests.security.compare.test_comparator_base import ComparatorBase
import pytest


class TestPlainComparator(ComparatorBase):
    @pytest.fixture()
    def env_and_comparator(self, sf_production_setup_devices):
        yield sf_production_setup_devices, PlainComparator(
            sf_production_setup_devices.alice
        )
