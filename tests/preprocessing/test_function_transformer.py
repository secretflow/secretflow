import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import FunctionTransformer as SkFunctionTransformer

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.mix.dataframe import MixDataFrame
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.preprocessing import LogroundTransformer
from secretflow.preprocessing.transformer import _FunctionTransformer
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.utils.simulation.datasets import load_iris

from tests.basecase import DeviceTestCase


class TestFunctionTransformer(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.hdf = load_iris(
            parts=[cls.alice, cls.bob],
            aggregator=PlainAggregator(cls.alice),
            comparator=PlainComparator(cls.carol),
        )
        cls.hdf_alice = reveal(cls.hdf.partitions[cls.alice].data)
        cls.hdf_bob = reveal(cls.hdf.partitions[cls.bob].data)

        vdf_alice = pd.DataFrame(
            {
                'a1': ['K5', 'K1', None, 'K6'],
                'a2': ['A5', 'A1', 'A2', 'A6'],
                'a3': [5, 1, 2, 6],
            }
        )

        vdf_bob = pd.DataFrame(
            {
                'b4': [10.2, 20.5, None, -0.4],
                'b5': ['B3', None, 'B9', 'B4'],
                'b6': [3, 1, 9, 4],
            }
        )

        cls.vdf_alice = vdf_alice
        cls.vdf_bob = vdf_bob
        cls.vdf = VDataFrame(
            {
                cls.alice: Partition(data=cls.alice(lambda: vdf_alice)()),
                cls.bob: Partition(data=cls.bob(lambda: vdf_bob)()),
            }
        )

    def test_on_vdataframe_should_ok(self):
        # GIVEN
        transformer = _FunctionTransformer(partial(np.add, 1))

        # WHEN
        value = transformer.fit_transform(self.vdf[['a3', 'b4', 'b6']])

        # THEN
        sk_transformer = SkFunctionTransformer(partial(np.add, 1))
        expect_alice = sk_transformer.fit_transform(self.vdf_alice[['a3']])
        pd.testing.assert_frame_equal(reveal(value.partitions[self.alice].data), expect_alice)

        expect_bob = sk_transformer.fit_transform(self.vdf_bob[['b4', 'b6']])
        pd.testing.assert_frame_equal(reveal(value.partitions[self.bob].data), expect_bob)

    def test_on_h_mixdataframe_should_ok(self):
        # GIVEN
        df_part0 = pd.DataFrame(
            {
                'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
                'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
                'a3': [5, 1, 2, 6, 15, None, 23, 6],
            }
        )

        df_part1 = pd.DataFrame(
            {
                'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4],
                'b5': ['B3', None, 'B9', 'B4', 'A3', None, 'C9', 'E4'],
                'b6': [3, 1, 9, 4, 31, 12, 9, 21],
            }
        )
        h_part0 = VDataFrame(
            {
                self.alice: Partition(data=self.alice(lambda: df_part0.iloc[:4, :])()),
                self.bob: Partition(data=self.bob(lambda: df_part1.iloc[:4, :])()),
            }
        )
        h_part1 = VDataFrame(
            {
                self.alice: Partition(data=self.alice(lambda: df_part0.iloc[4:, :])()),
                self.bob: Partition(data=self.bob(lambda: df_part1.iloc[4:, :])()),
            }
        )
        h_mix = MixDataFrame(partitions=[h_part0, h_part1])

        transformer = _FunctionTransformer(partial(np.add, 1))

        # WHEN
        value = transformer.fit_transform(h_mix[['a3', 'b4', 'b6']])

        # THEN
        sk_transformer = SkFunctionTransformer(partial(np.add, 1))
        expect_alice = sk_transformer.fit_transform(df_part0[['a3']])
        pd.testing.assert_frame_equal(
            pd.concat(
                [
                    reveal(value.partitions[0].partitions[self.alice].data),
                    reveal(value.partitions[1].partitions[self.alice].data),
                ]
            ),
            expect_alice,
        )
        expect_bob = sk_transformer.fit_transform(df_part1[['b4', 'b6']])
        pd.testing.assert_frame_equal(
            pd.concat(
                [
                    reveal(value.partitions[0].partitions[self.bob].data),
                    reveal(value.partitions[1].partitions[self.bob].data),
                ]
            ),
            expect_bob,
        )

    def test_on_v_mixdataframe_should_ok(self):
        # GIVEN
        df_part0 = pd.DataFrame(
            {
                'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
                'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
                'a3': [5, 1, 2, 6, 15, None, 23, 6],
            }
        )

        df_part1 = pd.DataFrame(
            {
                'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4],
                'b5': ['B3', None, 'B9', 'B4', 'A3', None, 'C9', 'E4'],
                'b6': [3, 1, 9, 4, 31, 12, 9, 21],
            }
        )
        v_part0 = HDataFrame(
            {
                self.alice: Partition(data=self.alice(lambda: df_part0.iloc[:4, :])()),
                self.bob: Partition(data=self.bob(lambda: df_part0.iloc[4:, :])()),
            },
            aggregator=PlainAggregator(self.carol),
            comparator=PlainComparator(self.carol),
        )
        v_part1 = HDataFrame(
            {
                self.alice: Partition(data=self.alice(lambda: df_part1.iloc[:4, :])()),
                self.bob: Partition(data=self.bob(lambda: df_part1.iloc[4:, :])()),
            },
            aggregator=PlainAggregator(self.carol),
            comparator=PlainComparator(self.carol),
        )
        v_mix = MixDataFrame(partitions=[v_part0, v_part1])

        transformer = _FunctionTransformer(partial(np.add, 1))

        # WHEN
        value = transformer.fit_transform(v_mix[['a3', 'b4', 'b6']])

        # THEN
        sk_transformer = SkFunctionTransformer(partial(np.add, 1))
        expect_alice = sk_transformer.fit_transform(df_part0[['a3']])
        pd.testing.assert_frame_equal(
            pd.concat(
                [
                    reveal(value.partitions[0].partitions[self.alice].data),
                    reveal(value.partitions[0].partitions[self.bob].data),
                ]
            ),
            expect_alice,
        )
        expect_bob = sk_transformer.fit_transform(df_part1[['b4', 'b6']])
        pd.testing.assert_frame_equal(
            pd.concat(
                [
                    reveal(value.partitions[1].partitions[self.alice].data),
                    reveal(value.partitions[1].partitions[self.bob].data),
                ]
            ),
            expect_bob,
        )

    def test_should_error_when_not_dataframe(self):
        transformer = _FunctionTransformer(partial(np.add, 1))
        with self.assertRaisesRegex(
            AssertionError, 'Accepts HDataFrame/VDataFrame/MixDataFrame only'
        ):
            transformer.fit(['test'])
        transformer.fit(self.vdf['a3'])
        with self.assertRaisesRegex(
            AssertionError, 'Accepts HDataFrame/VDataFrame/MixDataFrame only'
        ):
            transformer.transform('test')

    def test_transform_should_ok_when_not_fit(self):
        # GIVEN
        selected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        transformer = _FunctionTransformer(partial(np.add, 1))

        # WHEN
        value = transformer.transform(self.hdf[selected_cols])

        # THEN
        self.assertIsNotNone(value)

    def test_loground_on_vdataframe_should_ok(self):
        # GIVEN
        transformer = LogroundTransformer(decimals=2, bias=1)

        # WHEN
        value = transformer.fit_transform(self.vdf[['a3', 'b4', 'b6']])

        # THEN
        def loground(x: pd.DataFrame):
            return x.add(1).apply(np.log2).round(2)

        sk_transformer = SkFunctionTransformer(loground)
        expect_alice = sk_transformer.fit_transform(self.vdf_alice[['a3']])
        pd.testing.assert_frame_equal(reveal(value.partitions[self.alice].data), expect_alice)

        expect_bob = sk_transformer.fit_transform(self.vdf_bob[['b4', 'b6']])
        pd.testing.assert_frame_equal(reveal(value.partitions[self.bob].data), expect_bob)

    def test_loground_on_hdataframe_should_ok(self):
        # GIVEN
        selected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        transformer = LogroundTransformer(decimals=2, bias=1)

        # WHEN
        value = transformer.fit_transform(self.hdf[selected_cols])

        # THEN
        def loground(x: pd.DataFrame):
            return x.add(1).apply(np.log2).round(2)

        sk_transformer = SkFunctionTransformer(loground)
        sk_transformer.fit(
            pd.concat([self.hdf_alice[selected_cols], self.hdf_bob[selected_cols]])
        )
        expect_alice = sk_transformer.transform(self.hdf_alice[selected_cols])
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expect_alice,
        )
        expect_bob = sk_transformer.transform(self.hdf_bob[selected_cols])
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data), expect_bob
        )
