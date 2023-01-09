from secretflow.security.aggregation.secure_aggregator import SecureAggregator
from tests.basecase import MultiDriverDeviceTestCase
from tests.security.aggregation.test_aggregator_base import TestAggregatorBase


class TestSecureAggregator(MultiDriverDeviceTestCase, TestAggregatorBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.aggregator = SecureAggregator(cls.carol, [cls.alice, cls.bob])
