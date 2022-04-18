import pandas as pd

from secretflow.data.base import Partition
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.mix import MixDataFrame
from secretflow.data.vertical import VDataFrame
from secretflow.security.aggregation import DeviceAggregator
from secretflow.security.compare import PlainComparator
from secretflow.utils.errors import InvalidArgumentError

from tests.basecase import DeviceTestCase
from secretflow import reveal


class TestHMixDataFrame(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        df_part0 = pd.DataFrame({'a1': ['A1', 'B1', None, 'D1', None, 'B4', 'C4', 'D4'],
                                 'a2': ['A2', 'B2', 'C2', 'D2', 'A5', 'B5', 'C5', 'D5'],
                                 'a3': [5, 1, 2, 6, 15, None, 23, 6]})

        df_part1 = pd.DataFrame({'b4': [10.2, 20.5, None, -0.4, None, 0.5, None, -10.4],
                                 'b5': ['B3', None, 'B9', 'B4', 'A3', None, 'C9', 'E4'],
                                 'b6': [3, 1, 9, 4, 31, 12, 9, 21]})
        cls.df_part0, cls.df_part1 = df_part0, df_part1

        cls.h_part0 = VDataFrame(
            {cls.alice: Partition(data=cls.alice(lambda: df_part0.iloc[:4, :])()),
             cls.bob: Partition(data=cls.bob(lambda: df_part1.iloc[:4, :])())})
        cls.h_part1 = VDataFrame(
            {cls.alice: Partition(data=cls.alice(lambda: df_part0.iloc[4:, :])()),
             cls.bob: Partition(data=cls.bob(lambda: df_part1.iloc[4:, :])())})
        cls.h_mix = MixDataFrame(partitions=[cls.h_part0, cls.h_part1])

        cls.v_part0 = HDataFrame(
            {cls.alice: Partition(data=cls.alice(lambda: df_part0.iloc[:4, :])()),
             cls.bob: Partition(data=cls.bob(lambda: df_part0.iloc[4:, :])())},
            aggregator=DeviceAggregator(cls.carol), comparator=PlainComparator(cls.carol))
        cls.v_part1 = HDataFrame(
            {cls.alice: Partition(data=cls.alice(lambda: df_part1.iloc[:4, :])()),
             cls.bob: Partition(data=cls.bob(lambda: df_part1.iloc[4:, :])())},
            aggregator=DeviceAggregator(cls.carol), comparator=PlainComparator(cls.carol))
        cls.v_mix = MixDataFrame(partitions=[cls.v_part0, cls.v_part1])

    def test_mean_should_ok(self):
        # WHEN
        value_h, value_v = self.h_mix.mean(numeric_only=True), self.v_mix.mean(numeric_only=True)

        # THEN
        expected = pd.concat([self.df_part0.mean(numeric_only=True), self.df_part1.mean(numeric_only=True)])
        pd.testing.assert_series_equal(value_h, expected)
        pd.testing.assert_series_equal(value_v, expected)

    def test_min_should_ok(self):
        # WHEN
        value_h, value_v = self.h_mix.min(numeric_only=True), self.v_mix.min(numeric_only=True)

        # THEN
        expected = pd.concat([self.df_part0.min(numeric_only=True), self.df_part1.min(numeric_only=True)])
        pd.testing.assert_series_equal(value_h, expected)
        pd.testing.assert_series_equal(value_v, expected)

    def test_max_should_ok(self):
        # WHEN
        value_h, value_v = self.h_mix.max(numeric_only=True), self.v_mix.max(numeric_only=True)

        # THEN
        expected = pd.concat([self.df_part0.max(numeric_only=True), self.df_part1.max(numeric_only=True)])
        pd.testing.assert_series_equal(value_h, expected)
        pd.testing.assert_series_equal(value_v, expected)

    def test_count_should_ok(self):
        # WHEN
        value_h, value_v = self.h_mix.count(), self.v_mix.count()

        # THEN
        expected = pd.concat([self.df_part0.count(), self.df_part1.count()])
        pd.testing.assert_series_equal(value_h, expected)
        pd.testing.assert_series_equal(value_v, expected)

    def test_len_should_ok(self):
        # WHEN
        value_h, value_v = len(self.h_mix), len(self.v_mix)

        # THEN
        expected = len(self.df_part0)
        self.assertEqual(value_h, expected)
        self.assertEqual(value_v, expected)

    def test_getitem_should_ok(self):
        # Case 1: single item.
        # WHEN
        value_h, value_v = self.h_mix['a1'], self.v_mix['a1']
        # THEN
        expected_alice = self.df_part0[['a1']]
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value_h.partitions[0].partitions[self.alice].data),
                       reveal(value_h.partitions[1].partitions[self.alice].data)]), expected_alice)
        pd.testing.assert_frame_equal(
            pd.concat(
                [reveal(value_v.partitions[0].partitions[self.alice].data),
                 reveal(value_v.partitions[0].partitions[self.bob].data)]),
            expected_alice)

        # Case 2: multi items.
        # WHEN
        value_h, value_v = self.h_mix[['a2', 'b4', 'b5']], self.v_mix[['a2', 'b4', 'b5']]
        # THEN
        expected_alice = self.df_part0[['a2']]
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value_h.partitions[0].partitions[self.alice].data),
                       reveal(value_h.partitions[1].partitions[self.alice].data)]),
            expected_alice)
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value_v.partitions[0].partitions[self.alice].data),
                       reveal(value_v.partitions[0].partitions[self.bob].data)]),
            expected_alice)
        expected_bob = self.df_part1[['b4', 'b5']]
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value_h.partitions[0].partitions[self.bob].data),
                       reveal(value_h.partitions[1].partitions[self.bob].data)]),
            expected_bob)
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value_v.partitions[1].partitions[self.alice].data),
                       reveal(value_v.partitions[1].partitions[self.bob].data)]),
            expected_bob)

    def test_setitem_should_ok_when_single_value(self):
        # WHEN
        value_h, value_v = self.h_mix.copy(), self.v_mix.copy()

        # WEHN
        value_h['a1'] = 'test'
        value_v['a1'] = 'test'

        # THEN
        self.assertTrue((
                                pd.concat([reveal(value_h.partitions[0].partitions[self.alice].data),
                                           reveal(value_h.partitions[1].partitions[self.alice].data)])['a1'] == 'test').all())
        self.assertTrue((
                                pd.concat([reveal(value_v.partitions[0].partitions[self.alice].data),
                                           reveal(value_v.partitions[0].partitions[self.bob].data)])['a1'] == 'test').all())

    def test_setitem_should_ok_when_hmix(self):
        # GIVEN
        value = self.h_mix.copy()
        v_alice = pd.DataFrame({'a1': [f'a{i}' for i in range(8)]})

        v_bob = pd.DataFrame({'b4': [10.5 + i for i in range(8)],
                              'b6': [i for i in range(8)]})
        part0 = VDataFrame(
            {self.alice: Partition(data=self.alice(lambda: v_alice.iloc[:4, :])()),
             self.bob: Partition(data=self.bob(lambda: v_bob.iloc[:4, :])())})
        part1 = VDataFrame(
            {self.alice: Partition(data=self.alice(lambda: v_alice.iloc[4:, :])()),
             self.bob: Partition(data=self.bob(lambda: v_bob.iloc[4:, :])())})
        to = MixDataFrame(partitions=[part0, part1])

        # WHEN
        value[['a1', 'b4', 'b6']] = to

        # THEN
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value.partitions[0].partitions[self.alice].data),
                       reveal(value.partitions[1].partitions[self.alice].data)])[['a1']],
            v_alice)
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value.partitions[0].partitions[self.bob].data),
                       reveal(value.partitions[1].partitions[self.bob].data)])[['b4', 'b6']],
            v_bob)

    def test_setitem_should_ok_when_vmix(self):
        # GIVEN
        value = self.v_mix.copy()
        v_alice = pd.DataFrame({'a1': [f'a{i}' for i in range(8)]})

        v_bob = pd.DataFrame({'b4': [10.5 + i for i in range(8)],
                              'b6': [i for i in range(8)]})
        part0 = HDataFrame(
            {self.alice: Partition(data=self.alice(lambda: v_alice.iloc[:4, :])()),
             self.bob: Partition(data=self.bob(lambda: v_alice.iloc[4:, :])())})
        part1 = HDataFrame(
            {self.alice: Partition(data=self.alice(lambda: v_bob.iloc[:4, :])()),
             self.bob: Partition(data=self.bob(lambda: v_bob.iloc[4:, :])())})
        to = MixDataFrame(partitions=[part0, part1])

        # WHEN
        value[['a1', 'b4', 'b6']] = to

        # THEN
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value.partitions[0].partitions[self.alice].data),
                       reveal(value.partitions[0].partitions[self.bob].data)])[['a1']],
            v_alice)
        pd.testing.assert_frame_equal(
            pd.concat([reveal(value.partitions[1].partitions[self.alice].data),
                       reveal(value.partitions[1].partitions[self.bob].data)])[['b4', 'b6']],
            v_bob)

    def test_setitem_should_error_when_wrong_value_type(self):
        # GIVEN
        value = self.h_mix.copy()
        to = pd.DataFrame({'a1': [f'test{i}' for i in range(8)]})

        # WHEN & THEN
        with self.assertRaisesRegex(InvalidArgumentError,
                                    'Can not assgin a HDataFrame/VDataFrame/Partition to MixDataFrame.'):
            value['a1'] = Partition(data=self.alice(lambda: to)())
        with self.assertRaisesRegex(InvalidArgumentError,
                                    'Can not assgin a HDataFrame/VDataFrame/Partition to MixDataFrame.'):
            value['a1'] = self.h_part0
        with self.assertRaisesRegex(InvalidArgumentError,
                                    'Can not assgin a HDataFrame/VDataFrame/Partition to MixDataFrame.'):
            value['a1'] = self.v_part1

    def test_construct_should_error_when_diff_part_types(self):
        with self.assertRaisesRegex(AssertionError, 'All partitions should have same type'):
            MixDataFrame([self.h_part0, self.v_part0])

    def test_construct_should_error_when_none_or_empty_parts(self):
        with self.assertRaisesRegex(AssertionError, 'Partitions should not be None or empty.'):
            MixDataFrame()

        with self.assertRaisesRegex(AssertionError, 'Partitions should not be None or empty.'):
            MixDataFrame([])

    def test_set_partitions_should_error_when_diff_types(self):
        with self.assertRaisesRegex(AssertionError, 'All partitions should have same type'):
            mix = MixDataFrame([self.h_part0, self.h_part1])
            mix.partitions = [self.h_part0, self.v_part0]

    def test_set_partitions_should_error_when_none_or_empty_parts(self):
        with self.assertRaisesRegex(AssertionError, 'Partitions should not be None or empty.'):
            mix = MixDataFrame([self.h_part0, self.h_part1])
            mix.partitions = None

        with self.assertRaisesRegex(AssertionError, 'Partitions should not be None or empty.'):
            mix = MixDataFrame([self.h_part0, self.h_part1])
            mix.partitions = []
