import numpy as np
import pandas as pd

from secretflow import reveal
from secretflow.data.horizontal import read_csv
from secretflow.security.aggregation import PlainAggregator, SPUAggregator
from secretflow.security.compare import PlainComparator, SPUComparator

from tests.basecase import DeviceTestCase


class TestHDataFrame(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        path_alice = 'tests/datasets/iris/horizontal/iris.alice.csv'
        path_bob = 'tests/datasets/iris/horizontal/iris.bob.csv'
        cls.df_alice = pd.read_csv(path_alice)
        cls.df_bob = pd.read_csv(path_bob)
        cls.filepath = {cls.alice: path_alice, cls.bob: path_bob}

    def test_mean_with_plain_aggr_should_ok(self):
        # GIVEN
        hdf = read_csv(self.filepath, aggregator=PlainAggregator(self.carol))

        # WHEN
        mean = hdf.mean(numeric_only=True)

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
        # GIVEN
        hdf = read_csv(self.filepath, aggregator=SPUAggregator(self.spu))

        # WHEN
        mean = hdf.mean(numeric_only=True)

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
        # GIVEN
        hdf = read_csv(self.filepath, comparator=PlainComparator(self.carol))

        # WHEN
        min = hdf.min(numeric_only=True)

        # THEN
        expected = np.minimum(
            self.df_alice.min(numeric_only=True), self.df_bob.min(numeric_only=True)
        )
        pd.testing.assert_series_equal(min, expected)

    def test_min_with_spu_comp_should_ok(self):
        # GIVEN
        hdf = read_csv(self.filepath, comparator=SPUComparator(self.spu))

        # WHEN
        min = hdf.min(numeric_only=True)

        # THEN
        expected = np.minimum(
            self.df_alice.min(numeric_only=True), self.df_bob.min(numeric_only=True)
        )
        pd.testing.assert_series_equal(min, expected)

    def test_max_with_plain_comp_should_ok(self):
        # GIVEN
        hdf = read_csv(self.filepath, comparator=PlainComparator(self.carol))

        # WHEN
        max = hdf.max(numeric_only=True)

        # THEN
        expected = np.maximum(
            self.df_alice.max(numeric_only=True), self.df_bob.max(numeric_only=True)
        )
        pd.testing.assert_series_equal(max, expected)

    def test_max_with_spu_comp_should_ok(self):
        # GIVEN
        hdf = read_csv(self.filepath, comparator=SPUComparator(self.spu))

        # WHEN
        max = hdf.max(numeric_only=True)

        # THEN
        expected = np.maximum(
            self.df_alice.max(numeric_only=True), self.df_bob.max(numeric_only=True)
        )
        pd.testing.assert_series_equal(max, expected)

    def test_count_with_plain_aggr_should_ok(self):
        # GIVEN
        hdf = read_csv(self.filepath, aggregator=PlainAggregator(self.carol))

        # WHEN
        count = hdf.count()

        # THEN
        expected = self.df_alice.count() + self.df_bob.count()
        pd.testing.assert_series_equal(count, expected)

    def test_count_with_spu_aggr_should_ok(self):
        # GIVEN
        hdf = read_csv(self.filepath, aggregator=SPUAggregator(self.spu))

        # WHEN
        count = hdf.count()

        # THEN
        expected = self.df_alice.count() + self.df_bob.count()
        pd.testing.assert_series_equal(count, expected)

    def test_len_should_ok(self):
        # GIVEN
        hdf = read_csv(self.filepath)

        # WHEN
        length = len(hdf)

        # THEN
        expected = len(self.df_alice) + len(self.df_bob)
        self.assertEqual(length, expected)

    def test_getitem_should_ok(self):
        # GIVEN
        hdf = read_csv(self.filepath)

        # Case 1: single item.
        # WHEN
        value = hdf['sepal_length']
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
        value = hdf[['sepal_length', 'sepal_width']]
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
        hdf = read_csv(self.filepath)

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
        hdf = read_csv(self.filepath)

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
        hdf = read_csv(self.filepath)

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
