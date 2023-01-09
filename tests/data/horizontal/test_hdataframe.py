import numpy as np
import pandas as pd

from secretflow import reveal
from secretflow.security.aggregation import PlainAggregator, SPUAggregator
from secretflow.security.compare import PlainComparator, SPUComparator
from secretflow.utils.simulation.datasets import load_iris

from tests.basecase import MultiDriverDeviceTestCase


class TestHDataFrame(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.df_plain = load_iris(
            parts=[cls.alice, cls.bob],
            aggregator=PlainAggregator(cls.alice),
            comparator=PlainComparator(cls.alice),
        )
        cls.df_spu = load_iris(
            parts=[cls.alice, cls.bob],
            aggregator=SPUAggregator(cls.spu),
            comparator=SPUComparator(cls.spu),
        )
        cls.df_alice = reveal(cls.df_plain.partitions[cls.alice].data)
        cls.df_bob = reveal(cls.df_plain.partitions[cls.bob].data)

    def test_mean_with_plain_aggr_should_ok(self):
        # WHEN
        mean = self.df_plain.mean(numeric_only=True)

        # THEN
        expected = np.average(
            [
                self.df_alice.mean(numeric_only=True),
                self.df_bob.mean(numeric_only=True),
            ],
            weights=[
                self.df_alice.count(numeric_only=True),
                self.df_bob.count(numeric_only=True),
            ],
            axis=0,
        )
        pd.testing.assert_series_equal(
            mean,
            pd.Series(
                expected,
                index=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            ),
        )

    def test_mean_with_spu_aggr_should_ok(self):
        # WHEN
        mean = self.df_spu.mean(numeric_only=True)

        # THEN
        expected = np.average(
            [
                self.df_alice.mean(numeric_only=True),
                self.df_bob.mean(numeric_only=True),
            ],
            weights=[
                self.df_alice.count(numeric_only=True),
                self.df_bob.count(numeric_only=True),
            ],
            axis=0,
        )
        pd.testing.assert_series_equal(
            mean,
            pd.Series(
                expected,
                index=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            ),
        )

    def test_min_with_plain_comp_should_ok(self):
        # WHEN
        min = self.df_plain.min(numeric_only=True)

        # THEN
        expected = np.minimum(
            self.df_alice.min(numeric_only=True), self.df_bob.min(numeric_only=True)
        )
        pd.testing.assert_series_equal(min, expected)

    def test_min_with_spu_comp_should_ok(self):
        # WHEN
        min = self.df_spu.min(numeric_only=True)

        # THEN
        expected = np.minimum(
            self.df_alice.min(numeric_only=True), self.df_bob.min(numeric_only=True)
        )
        pd.testing.assert_series_equal(min, expected)

    def test_max_with_plain_comp_should_ok(self):
        # WHEN
        max = self.df_plain.max(numeric_only=True)

        # THEN
        expected = np.maximum(
            self.df_alice.max(numeric_only=True), self.df_bob.max(numeric_only=True)
        )
        pd.testing.assert_series_equal(max, expected)

    def test_max_with_spu_comp_should_ok(self):
        # WHEN
        max = self.df_spu.max(numeric_only=True)

        # THEN
        expected = np.maximum(
            self.df_alice.max(numeric_only=True), self.df_bob.max(numeric_only=True)
        )
        pd.testing.assert_series_equal(max, expected)

    def test_count_with_plain_aggr_should_ok(self):
        # WHEN
        count = self.df_plain.count()

        # THEN
        expected = self.df_alice.count() + self.df_bob.count()
        pd.testing.assert_series_equal(count, expected)

    def test_count_with_spu_aggr_should_ok(self):
        # WHEN
        count = self.df_spu.count()

        # THEN
        expected = self.df_alice.count() + self.df_bob.count()
        pd.testing.assert_series_equal(count, expected)

    def test_count_na_with_plain_aggr_should_ok(self):
        # WHEN
        # Note currently, our device execution may result in different types
        # compared to original pandas, like int32 not int64
        count = self.df_plain.isna().sum().astype(np.int64)

        # THEN
        expected = self.df_alice.isna().sum() + self.df_bob.isna().sum()
        pd.testing.assert_series_equal(count, expected)

    def test_count_na_with_spu_aggr_should_ok(self):
        # WHEN
        # Note currently, our device execution may result in different types
        # compared to original pandas, like int32 not int64
        count = self.df_spu.isna().sum().astype(np.int64)

        # THEN
        expected = self.df_alice.isna().sum() + self.df_bob.isna().sum()
        pd.testing.assert_series_equal(count, expected)

    def test_len_should_ok(self):
        # WHEN
        length = len(self.df_plain)

        # THEN
        expected = len(self.df_alice) + len(self.df_bob)
        self.assertEqual(length, expected)

    def test_getitem_should_ok(self):
        # Case 1: single item.
        # WHEN
        value = self.df_plain['sepal_length']
        # THEN
        expected_alice = self.df_alice[['sepal_length']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )
        expected_bob = self.df_bob[['sepal_length']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data), expected_bob
        )

        # Case 2: multi items.
        # WHEN
        value = self.df_plain[['sepal_length', 'sepal_width']]
        # THEN
        expected_alice = self.df_alice[['sepal_length', 'sepal_width']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )
        expected_bob = self.df_bob[['sepal_length', 'sepal_width']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data), expected_bob
        )

    def test_setitem_should_ok(self):
        # GIVEN
        hdf = self.df_plain.copy()

        # Case 1: single item.
        # WHEN
        hdf['sepal_length'] = 'test'
        # THEN
        expected_alice = self.df_alice
        expected_alice['sepal_length'] = 'test'
        pd.testing.assert_frame_equal(
            reveal(hdf.partitions[self.alice].data), expected_alice
        )
        expected_bob = self.df_bob
        expected_bob['sepal_length'] = 'test'
        pd.testing.assert_frame_equal(
            reveal(hdf.partitions[self.bob].data), expected_bob
        )

        # Case 2: multi items.
        # WHEN
        hdf[['sepal_length', 'sepal_width']] = self.df_alice[
            ['sepal_length', 'sepal_width']
        ]
        # THEN
        expected_alice = self.df_alice
        pd.testing.assert_frame_equal(
            reveal(hdf.partitions[self.alice].data), expected_alice
        )
        expected_bob = self.df_bob
        expected_bob[['sepal_length', 'sepal_width']] = self.df_alice[
            ['sepal_length', 'sepal_width']
        ]
        pd.testing.assert_frame_equal(
            reveal(hdf.partitions[self.bob].data), expected_bob
        )

    def test_drop(self):
        # GIVEN
        hdf = self.df_plain.copy()

        # Case 1: not inplace.
        # WHEN
        new_hdf = hdf.drop(columns='sepal_length', inplace=False)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(new_hdf.partitions[self.alice].data),
            self.df_alice.drop(columns='sepal_length', inplace=False),
        )
        pd.testing.assert_frame_equal(
            reveal(new_hdf.partitions[self.bob].data),
            self.df_bob.drop(columns='sepal_length', inplace=False),
        )

        # Case 2: inplace.
        # WHEN
        hdf.drop(columns='sepal_length', inplace=True)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(hdf.partitions[self.alice].data),
            self.df_alice.drop(columns='sepal_length', inplace=False),
        )
        pd.testing.assert_frame_equal(
            reveal(hdf.partitions[self.bob].data),
            self.df_bob.drop(columns='sepal_length', inplace=False),
        )

    def test_fillna(self):
        # GIVEN
        hdf = self.df_plain.copy()

        # Case 1: not inplace.
        # WHEN
        new_hdf = hdf.fillna(value='test', inplace=False)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(new_hdf.partitions[self.alice].data),
            self.df_alice.fillna(value='test', inplace=False),
        )
        pd.testing.assert_frame_equal(
            reveal(new_hdf.partitions[self.bob].data),
            self.df_bob.fillna(value='test', inplace=False),
        )

        # Case 2: inplace.
        # WHEN
        hdf.fillna(value='test', inplace=True)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(hdf.partitions[self.alice].data),
            self.df_alice.fillna(value='test', inplace=False),
        )
        pd.testing.assert_frame_equal(
            reveal(hdf.partitions[self.bob].data),
            self.df_bob.fillna(value='test', inplace=False),
        )

    def test_astype_should_ok(self):
        # GIVEN
        hdf = self.df_plain[
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        ]
        hdf.fillna(value=1, inplace=True)

        # Case 1: single dtype
        # WHEN
        new_hdf = hdf.astype(np.int32)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(new_hdf.partitions[self.alice].data),
            self.df_alice.iloc[:, 0:4].fillna(1).astype(np.int32),
        )
        pd.testing.assert_frame_equal(
            reveal(new_hdf.partitions[self.bob].data),
            self.df_bob.iloc[:, 0:4].fillna(1).astype(np.int32),
        )

        # Case 2: dtype dict.
        # WHEN
        dtype = {'sepal_length': np.int32, 'sepal_width': np.int32}
        new_hdf = hdf.astype(dtype)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(new_hdf.partitions[self.alice].data),
            self.df_alice.iloc[:, 0:4].fillna(1).astype(dtype),
        )
        pd.testing.assert_frame_equal(
            reveal(new_hdf.partitions[self.bob].data),
            self.df_bob.iloc[:, 0:4].fillna(1).astype(dtype),
        )
