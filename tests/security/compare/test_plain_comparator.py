from secretflow.security.compare.plain_comparator import PlainComparator
from tests.basecase import MultiDriverDeviceTestCase
from tests.security.compare.test_comparator_base import TestComparatorBase


class TestPlainComparator(MultiDriverDeviceTestCase, TestComparatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.comparator = PlainComparator(cls.alice)
