import numpy as np


class TestAggregatorBase:
    def test_sum_on_single_should_ok(self):
        # GIVEN
        a = self.alice(lambda: np.array([[1., 2., 3], [4., 5., 6.]]))()
        b = self.bob(lambda: np.array([[11., 12., 13.], [14, 15., 16.]]))()

        # WHEN
        sum = self.aggregator.sum([a, b], axis=0)

        # THEN
        np.testing.assert_equal(sum, np.array([[12., 14., 16.], [18., 20., 22.]]))

    def test_sum_on_list_should_ok(self):
        # GIVEN
        a = self.alice(lambda: [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[21, 22, 23], [24, 25, 26]])])()
        b = self.bob(lambda: [np.array([[11, 12, 13], [14, 15, 16]]), np.array([[31, 32, 33], [34, 35, 36]])])()

        # WHEN
        sum = self.aggregator.sum([a, b], axis=0)

        # THEN
        np.testing.assert_equal(sum[0], np.array([[12, 14, 16], [18, 20, 22]]))
        np.testing.assert_equal(sum[1], np.array([[52, 54, 56], [58, 60, 62]]))

    def test_average_on_single_without_weights_should_ok(self):
        # GIVEN
        a = self.alice(lambda: np.array([[1., 2., 3.], [4., 5., 6.]]))()
        b = self.bob(lambda: np.array([[11., 12., 13.], [14., 15., 16.]]))()

        # WHEN
        sum = self.aggregator.average([a, b], axis=0)

        # THEN
        np.testing.assert_equal(sum, np.array([[6., 7., 8.], [9., 10., 11.]]))

    def test_average_on_list_without_weights_should_ok(self):
        # GIVEN
        a = self.alice(lambda: [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[21, 22, 23], [24, 25, 26]])])()
        b = self.bob(lambda: [np.array([[11, 12, 13], [14, 15, 16]]), np.array([[31, 32, 33], [34, 35, 36]])])()

        # WHEN
        sum = self.aggregator.average([a, b], axis=0)

        # THEN
        np.testing.assert_equal(sum[0], np.array([[6, 7, 8], [9, 10, 11]]))
        np.testing.assert_equal(sum[1], np.array([[26, 27, 28], [29, 30, 31]]))

    def test_average_with_weights_should_ok(self):
        # GIVEN
        a = self.alice(lambda: np.array([[1, 2, 3], [4, 5, 6]]))()
        b = self.bob(lambda: np.array([[11, 12, 13], [14, 15, 16]]))()

        # WHEN
        sum = self.aggregator.average([a, b], axis=0, weights=[2, 3])

        # THEN
        np.testing.assert_equal(sum, np.array([[7, 8, 9], [10, 11, 12]]))

    def test_average_on_list_with_weights_should_ok(self):
        # GIVEN
        a = self.alice(lambda: [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[21, 22, 23], [24, 25, 26]])])()
        b = self.bob(lambda: [np.array([[11, 12, 13], [14, 15, 16]]), np.array([[31, 32, 33], [34, 35, 36]])])()

        # WHEN
        sum = self.aggregator.average([a, b], axis=0, weights=[2, 3])

        # THEN
        np.testing.assert_equal(sum[0], np.array([[7, 8, 9], [10, 11, 12]]))
        np.testing.assert_equal(sum[1], np.array([[27, 28, 29], [30, 31, 32]]))
