import numpy as np

from secretflow import reveal


class TestComparatorBase:
    def test_min_should_ok(self):
        # GIVEN
        a = self.alice(lambda: np.array([[1, 2, 3], [14, 15, 16]]))()
        b = self.bob(lambda: np.array([[11, 12, 13], [4, 5, 6]]))()

        # WHEN
        min = reveal(self.comparator.min([a, b], axis=0))

        # THEN
        np.testing.assert_equal(reveal(min), np.array([[1, 2, 3], [4, 5, 6]]))

    def test_max_should_ok(self):
        # GIVEN
        a = self.alice(lambda: np.array([[1, 2, 3], [14, 15, 16]]))()
        b = self.bob(lambda: np.array([[11, 12, 13], [4, 5, 6]]))()

        # WHEN
        max = reveal(self.comparator.max([a, b], axis=0))

        # THEN
        np.testing.assert_equal(reveal(max), np.array([[11, 12, 13], [14, 15, 16]]))
