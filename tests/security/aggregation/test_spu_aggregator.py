from secretflow.security.aggregation.spu_aggregator import SPUAggregator
from tests.basecase import DeviceTestCase
from tests.security.aggregation.test_aggregator_base import TestAggregatorBase


class TestSPUAggregator(DeviceTestCase, TestAggregatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.aggregator = SPUAggregator(cls.spu)
