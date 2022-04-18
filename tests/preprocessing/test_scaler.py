import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.horizontal import read_csv as h_read_csv
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.mix.dataframe import MixDataFrame
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.security.aggregation.device_aggregator import \
    DeviceAggregator
from secretflow.security.compare.plain_comparator import \
    PlainComparator
from tests.basecase import DeviceTestCase


class TestScaler(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        path_alice = 'tests/datasets/iris/horizontal/iris.alice.csv'
        path_bob = 'tests/datasets/iris/horizontal/iris.bob.csv'
        cls.hdf_alice = pd.read_csv(path_alice)
        cls.hdf_bob = pd.read_csv(path_bob)
        cls.hdf = h_read_csv(
            {cls.alice: path_alice, cls.bob: path_bob},
            aggregator=DeviceAggregator(cls.carol),
            comparator=PlainComparator(cls.carol))

        vdf_alice = pd.DataFrame({'a1': ['K5', 'K1', None, 'K6'],
                                  'a2': ['A5', 'A1', 'A2', 'A6'],
                                  'a3': [5, 1, 2, 6]})

        vdf_bob = pd.DataFrame({'b4': [10.2, 20.5, None, -0.4],
                                'b5': ['B3', None, 'B9', 'B4'],
                                'b6': [3, 1, 9, 4]})

        cls.vdf_alice = vdf_alice
        cls.vdf_bob = vdf_bob
        cls.vdf = VDataFrame(
            {cls.alice: Partition(data=cls.alice(lambda: vdf_alice)()),
             cls.bob: Partition(data=cls.bob(lambda: vdf_bob)())})

    def test_on_hdataframe_should_ok(self):
        # GIVEN
        selected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        scaler = MinMaxScaler()

        # WHEN
        value = scaler.fit_transform(self.hdf[selected_cols])

        # THEN
        sk_scaler = SkMinMaxScaler()
        sk_scaler.fit(pd.concat([self.hdf_alice[selected_cols], self.hdf_bob[selected_cols]]))
        expect_alice = sk_scaler.transform(self.hdf_alice[selected_cols])
        np.testing.assert_almost_equal(reveal(value.partitions[self.alice].data), expect_alice, )
        expect_bob = sk_scaler.transform(self.hdf_bob[selected_cols])
        np.testing.assert_almost_equal(reveal(value.partitions[self.bob].data), expect_bob)

    def test_on_vdataframe_should_ok(self):
        # GIVEN
        scaler = MinMaxScaler()

        # WHEN
        value = scaler.fit_transform(self.vdf[['a3', 'b4', 'b6']])

        # THEN
        sk_scaler = SkMinMaxScaler()
        expect_alice = sk_scaler.fit_transform(self.vdf_alice[['a3']])
        np.testing.assert_equal(reveal(value.partitions[self.alice].data), expect_alice)

        expect_bob = sk_scaler.fit_transform(self.vdf_bob[['b4', 'b6']])
        np.testing.assert_equal(reveal(value.partitions[self.bob].data), expect_bob)

    def test_on_h_mixdataframe_should_ok(self):
        # GIVEN
        df_part0 = pd.DataFrame({'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
                                 'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
                                 'a3': [5, 1, 2, 6, 15, None, 23, 6]})

        df_part1 = pd.DataFrame({'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4],
                                 'b5': ['B3', None, 'B9', 'B4', 'A3', None, 'C9', 'E4'],
                                 'b6': [3, 1, 9, 4, 31, 12, 9, 21]})
        h_part0 = VDataFrame(
            {self.alice: Partition(data=self.alice(lambda: df_part0.iloc[:4, :])()),
             self.bob: Partition(data=self.bob(lambda: df_part1.iloc[:4, :])())})
        h_part1 = VDataFrame(
            {self.alice: Partition(data=self.alice(lambda: df_part0.iloc[4:, :])()),
             self.bob: Partition(data=self.bob(lambda: df_part1.iloc[4:, :])())})
        h_mix = MixDataFrame(partitions=[h_part0, h_part1])

        scaler = MinMaxScaler()

        # WHEN
        value = scaler.fit_transform(h_mix[['a3', 'b4', 'b6']])

        # THEN
        sk_scaler = SkMinMaxScaler()
        expect_alice = sk_scaler.fit_transform(df_part0[['a3']])
        np.testing.assert_equal(
            pd.concat(
                [reveal(value.partitions[0].partitions[self.alice].data),
                 reveal(value.partitions[1].partitions[self.alice].data)]),
            expect_alice)
        expect_bob = sk_scaler.fit_transform(df_part1[['b4', 'b6']])
        np.testing.assert_equal(
            pd.concat(
                [reveal(value.partitions[0].partitions[self.bob].data),
                 reveal(value.partitions[1].partitions[self.bob].data)]),
            expect_bob)

    def test_on_v_mixdataframe_should_ok(self):
        # GIVEN
        df_part0 = pd.DataFrame({'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
                                 'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
                                 'a3': [5, 1, 2, 6, 15, None, 23, 6]})

        df_part1 = pd.DataFrame({'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4],
                                 'b5': ['B3', None, 'B9', 'B4', 'A3', None, 'C9', 'E4'],
                                 'b6': [3, 1, 9, 4, 31, 12, 9, 21]})
        v_part0 = HDataFrame(
            {self.alice: Partition(data=self.alice(lambda: df_part0.iloc[:4, :])()),
             self.bob: Partition(data=self.bob(lambda: df_part0.iloc[4:, :])())},
            aggregator=DeviceAggregator(self.carol), comparator=PlainComparator(self.carol))
        v_part1 = HDataFrame(
            {self.alice: Partition(data=self.alice(lambda: df_part1.iloc[:4, :])()),
             self.bob: Partition(data=self.bob(lambda: df_part1.iloc[4:, :])())},
            aggregator=DeviceAggregator(self.carol), comparator=PlainComparator(self.carol))
        v_mix = MixDataFrame(partitions=[v_part0, v_part1])

        scaler = MinMaxScaler()

        # WHEN
        value = scaler.fit_transform(v_mix[['a3', 'b4', 'b6']])

        # THEN
        sk_scaler = SkMinMaxScaler()
        expect_alice = sk_scaler.fit_transform(df_part0[['a3']])
        np.testing.assert_equal(
            pd.concat(
                [reveal(value.partitions[0].partitions[self.alice].data),
                 reveal(value.partitions[0].partitions[self.bob].data)]),
            expect_alice)
        expect_bob = sk_scaler.fit_transform(df_part1[['b4', 'b6']])
        np.testing.assert_almost_equal(
            pd.concat(
                [reveal(value.partitions[1].partitions[self.alice].data),
                 reveal(value.partitions[1].partitions[self.bob].data)]),
            expect_bob)

    def test_should_error_when_not_dataframe(self):
        scaler = MinMaxScaler()
        with self.assertRaisesRegex(AssertionError, 'Accepts HDataFrame/VDataFrame/MixDataFrame only'):
            scaler.fit(['test'])
        scaler.fit(self.vdf['a3'])
        with self.assertRaisesRegex(AssertionError, 'Accepts HDataFrame/VDataFrame/MixDataFrame only'):
            scaler.transform('test')

    def test_transform_should_error_when_not_fit(self):
        with self.assertRaisesRegex(AssertionError, 'Scaler has not been fit yet.'):
            MinMaxScaler().transform('test')
