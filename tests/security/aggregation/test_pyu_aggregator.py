from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from tests.basecase import DeviceTestCase
from tests.security.aggregation.test_aggregator_base import TestAggregatorBase


class TestPlainAggregator(DeviceTestCase, TestAggregatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.aggregator = PlainAggregator(cls.carol)
