from secretflow.security.compare.spu_comparator import SPUComparator
from tests.basecase import MultiDriverDeviceTestCase
from tests.security.compare.test_comparator_base import TestComparatorBase


class TestSPUComparator(MultiDriverDeviceTestCase, TestComparatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.comparator = SPUComparator(cls.spu)
