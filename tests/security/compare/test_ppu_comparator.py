from secretflow.security.compare.ppu_comparator import PPUComparator

from tests.basecase import DeviceTestCase
from tests.security.compare.test_comparator_base import TestComparatorBase


class TestPPUComparator(DeviceTestCase, TestComparatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.comparator = PPUComparator(cls.ppu)
