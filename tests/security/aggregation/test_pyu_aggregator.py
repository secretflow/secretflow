from secretflow.security.aggregation.plain_aggregator import PlainAggregator

from secretflow.security.aggregation.sparse_plain_aggregator import (
    SparsePlainAggregator,
)
from tests.basecase import MultiDriverDeviceTestCase
from tests.security.aggregation.test_aggregator_base import TestAggregatorBase


class TestPlainAggregator(MultiDriverDeviceTestCase, TestAggregatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.aggregator = PlainAggregator(cls.carol)


class TestSparsePlainAggregator(MultiDriverDeviceTestCase, TestAggregatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.aggregator = SparsePlainAggregator(cls.carol)
