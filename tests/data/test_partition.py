import pandas as pd

from secretflow import reveal
from secretflow.utils.simulation.datasets import load_iris

from tests.basecase import DeviceTestCase


class TestPartition(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        iris = load_iris(parts=[cls.alice])
        cls.part = iris.partitions[cls.alice]
        cls.df = reveal(cls.part.data)

    def test_mean_should_ok(self):
        # WHEN
        value = self.part.mean(numeric_only=True)

        # THEN
        expected = self.df.mean(numeric_only=True)
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_min_should_ok(self):
        # WHEN
        value = self.part.min()

        # THEN
        expected = self.df.min()
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_max_should_ok(self):
        # WHEN
        value = self.part.max()

        # THEN
        expected = self.df.max()
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_count_should_ok(self):
        # WHEN
        value = self.part.count()

        # THEN
        expected = self.df.count()
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_dtypes_should_ok(self):
        # WHEN
        value = self.part.dtypes

        # THEN
        expected = self.df.dtypes
        pd.testing.assert_series_equal(value, expected)

    def test_index_should_ok(self):
        # WHEN
        value = self.part.index

        # THEN
        expected = self.df.index
        pd.testing.assert_index_equal(value, expected)

    def test_len_should_ok(self):
        # WHEN
        value = len(self.part)

        # THEN
        expected = len(self.df)
        self.assertEqual(value, expected)

    def test_iloc_should_ok(self):
        # WHEN
        value = self.part.iloc(0)
        # THEN
        expected = self.df.iloc[0]
        pd.testing.assert_series_equal(reveal(value.data), expected)

        # WHEN
        value = self.part.iloc([0, 1])
        # THEN
        expected = self.df.iloc[[0, 1]]
        pd.testing.assert_frame_equal(reveal(value.data), expected)

    def test_getitem_should_ok(self):
        # WHEN
        value = self.part['sepal_length']
        # THEN
        expected = self.df[['sepal_length']]
        pd.testing.assert_frame_equal(reveal(value.data), expected)

        # WHEN
        value = self.part[['sepal_length', 'sepal_width']]
        # THEN
        expected = self.df[['sepal_length', 'sepal_width']]
        pd.testing.assert_frame_equal(reveal(value.data), expected)

    def test_setitem_should_ok(self):
        # WHEN
        value = self.part.copy()
        value['sepal_length'] = 2

        # THEN
        expected = self.df.copy(deep=True)
        expected['sepal_length'] = 2
        pd.testing.assert_frame_equal(reveal(value.data), expected)

    def test_setitem_on_partition_should_ok(self):
        # WHEN
        value = self.part.copy()
        value['sepal_length'] = self.part['sepal_width']

        # THEN
        expected = self.df.copy(deep=True)
        expected['sepal_length'] = expected['sepal_width']
        pd.testing.assert_frame_equal(reveal(value.data), expected)

    def test_setitem_on_different_partition_should_ok(self):
        # WHEN and THEN
        with self.assertRaisesRegex(
            AssertionError, 'Can not assign a partition with different device.'
        ):
            part = load_iris(parts=[self.bob]).partitions[self.bob]
            value = self.part.copy()
            value['sepal_length'] = part['sepal_width']

    def test_drop_should_ok(self):
        # Case 1: not inplace.
        # WHEN
        value = self.part.drop(columns=['sepal_length'], inplace=False)

        # THEN
        expected = self.df.drop(columns=['sepal_length'], inplace=False)
        pd.testing.assert_frame_equal(reveal(value.data), expected)

        # Case 2: inplace.
        # WHEN
        value = self.part.copy()
        value.drop(columns=['sepal_length'], inplace=True)

        # THEN
        expected = self.df.copy(deep=True)
        expected.drop(columns=['sepal_length'], inplace=True)
        pd.testing.assert_frame_equal(reveal(value.data), expected)

    def test_fillna_should_ok(self):
        # Case 1: not inplace.
        # WHEN
        value = self.part.fillna(value='test', inplace=False)

        # THEN
        expected = self.df.fillna(value='test', inplace=False)
        pd.testing.assert_frame_equal(reveal(value.data), expected)

        # Case 2: inplace.
        # WHEN
        value = self.part.copy()
        value.fillna(value='test', inplace=True)

        # THEN
        expected = self.df.copy(deep=True)
        expected.fillna(value='test', inplace=True)
        pd.testing.assert_frame_equal(reveal(value.data), expected)
