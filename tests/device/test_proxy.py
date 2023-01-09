from typing import Tuple

import numpy as np

from secretflow import reveal
from secretflow.device import PYUObject, proxy

from tests.basecase import (MultiDriverDeviceTestCase,
                            SingleDriverDeviceTestCase)


@proxy(PYUObject)
class Model:
    def __init__(self, builder):
        self.weights = builder()
        self.dataset_x = None
        self.dataset_y = None

    def build_dataset(self, x, y):
        self.dataset_x = x
        self.dataset_y = y

    def retrieve_dataset(self):
        return self.dataset_x, self.dataset_y

    def get_weights(self):
        return self.weights

    def compute_weights(self) -> Tuple[np.ndarray, int]:
        return self.weights, 100


class TestProxy(MultiDriverDeviceTestCase, SingleDriverDeviceTestCase):
    def setUp(self) -> None:
        self.model = Model(lambda: np.ones((3, 4)), device=self.alice)

    def test_init_without_device(self):
        with self.assertRaisesRegex(AssertionError, 'missing device argument'):
            Model(lambda: np.ones((3, 4)))

    def test_init_with_mismatch_device(self):
        with self.assertRaisesRegex(AssertionError, 'unexpected device type'):
            Model(lambda: np.ones((3, 4)), device=self.spu)

    def test_call_with_mismatch_device(self):
        x, y = self.alice(np.random.rand)(3, 4), self.bob(np.random.rand)(3)
        with self.assertRaisesRegex(AssertionError, 'unexpected device object'):
            self.model.build_dataset(x, y)

    def test_single_return(self):
        weights = self.model.get_weights()
        self.assertEqual(weights.device, self.alice)
        weights = reveal(weights)
        np.testing.assert_equal(weights, np.ones((3, 4)))

    def test_multiple_return(self):
        weights, n = self.model.compute_weights()
        self.assertEqual(weights.device, self.alice)
        self.assertEqual(n.device, self.alice)

        weights, n = reveal(weights), reveal(n)
        np.testing.assert_equal(weights, np.ones((3, 4)))
        self.assertEqual(n, 100)

    def test_multiple_return_without_annotation(self):
        x, y = self.alice(np.random.rand)(3, 4), self.alice(np.random.rand)(3)
        self.model.build_dataset(x, y)

        res = self.model.retrieve_dataset()
        self.assertEqual(res.device, self.alice)

        x, y = reveal([x, y])
        x_, y_ = reveal(res)
        np.testing.assert_equal(x_, x)
        np.testing.assert_equal(y_, y)
