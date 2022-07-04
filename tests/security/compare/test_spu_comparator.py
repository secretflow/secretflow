from secretflow.security.compare.spu_comparator import SPUComparator
from tests.basecase import DeviceTestCase
from tests.security.compare.test_comparator_base import TestComparatorBase


class TestSPUComparator(DeviceTestCase, TestComparatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.comparator = SPUComparator(cls.spu)
