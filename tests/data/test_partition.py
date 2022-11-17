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

    def test_var_should_ok(self):
        # WHEN
        value = self.part.var(numeric_only=True)

        # THEN
        expected = self.df.var(numeric_only=True)
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_std_should_ok(self):
        # WHEN
        value = self.part.std(numeric_only=True)

        # THEN
        expected = self.df.std(numeric_only=True)
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_sem_should_ok(self):
        # WHEN
        value = self.part.sem(numeric_only=True)

        # THEN
        expected = self.df.sem(numeric_only=True)
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_skew_should_ok(self):
        # WHEN
        value = self.part.skew(numeric_only=True)

        # THEN
        expected = self.df.skew(numeric_only=True)
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_kurtosis_should_ok(self):
        # WHEN
        value = self.part.kurtosis(numeric_only=True)

        # THEN
        expected = self.df.kurtosis(numeric_only=True)
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_quantile_should_ok(self):
        # WHEN
        value = self.part.quantile()

        # THEN
        expected = self.df.quantile()
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

    def test_pow_should_ok(self):
        # WHEN
        value = self.part.select_dtypes('number').pow(2.3).sum()

        # THEN
        expected = self.df.select_dtypes('number').pow(2.3).sum()
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_select_dtypes_should_ok(self):
        # WHEN
        value = self.part.select_dtypes('number').mean()

        # THEN
        expected = self.df.select_dtypes('number').mean()
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_subtract_should_ok(self):
        # WHEN
        part_num = self.part.select_dtypes('number')
        means = part_num.mean()
        value = part_num.subtract(means)[part_num.columns].mean(numeric_only=True)

        # THEN
        df_num = self.df.select_dtypes('number')
        df_means = df_num.mean()
        expected = df_num.subtract(df_means)[df_num.columns].mean(numeric_only=True)
        pd.testing.assert_series_equal(reveal(value.data), expected)

    def test_round_should_ok(self):
        # WHEN
        value = self.part.round(1)

        # THEN
        expected = self.df.round(1)
        pd.testing.assert_frame_equal(reveal(value.data), expected)

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

    def test_setitem_on_different_partition_should_error(self):
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

    def test_replace_should_ok(self):
        # WHEN
        val = self.df.iloc[1, 1]
        val_to = 0.31312
        value = self.part.replace(val, val_to)

        # THEN
        expected = self.df.replace(val, val_to)
        pd.testing.assert_frame_equal(reveal(value.data), expected)

    def test_mode_should_ok(self):
        # WHEN
        value = self.part.mode()

        # THEN
        expected = self.df.mode().iloc[0, :]
        pd.testing.assert_series_equal(reveal(value.data), expected)
