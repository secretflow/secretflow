from secretflow.security.compare.plain_comparator import PlainComparator
from tests.basecase import DeviceTestCase
from tests.security.compare.test_comparator_base import TestComparatorBase


class TestPlainComparator(DeviceTestCase, TestComparatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.comparator = PlainComparator(cls.alice)
