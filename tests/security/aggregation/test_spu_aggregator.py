from secretflow.security.aggregation.spu_aggregator import SPUAggregator

from tests.basecase import MultiDriverDeviceTestCase
from tests.security.aggregation.test_aggregator_base import TestAggregatorBase


class TestSPUAggregator(MultiDriverDeviceTestCase, TestAggregatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.aggregator = SPUAggregator(cls.spu)
