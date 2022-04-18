from secretflow.security.aggregation.device_aggregator import DeviceAggregator

from tests.basecase import DeviceTestCase
from tests.security.aggregation.test_aggregator_base import TestAggregatorBase


class TestPPUAggregator(DeviceTestCase, TestAggregatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.aggregator = DeviceAggregator(cls.ppu)
