import numpy as np
import pandas as pd

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame
from secretflow.utils.errors import NotFoundError

from tests.basecase import DeviceTestCase


class TestVDataFrame(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        df_alice = pd.DataFrame(
            {
                'a1': ['K5', 'K1', None, 'K6'],
                'a2': ['A5', 'A1', 'A2', 'A6'],
                'a3': [5, 1, 2, 6],
            }
        )

        df_bob = pd.DataFrame(
            {
                'b4': [10.2, 20.5, None, -0.4],
                'b5': ['B3', None, 'B9', 'B4'],
                'b6': [3, 1, 9, 4],
            }
        )

        cls.df_alice = df_alice
        cls.df_bob = df_bob
        cls.df = VDataFrame(
            {
                cls.alice: Partition(data=cls.alice(lambda: df_alice)()),
                cls.bob: Partition(data=cls.bob(lambda: df_bob)()),
            }
        )

    def test_columns_should_ok(self):
        # WHEN
        columns = self.df.columns

        # THEN
        pd.testing.assert_index_equal(
            columns, self.df_alice.columns.append(self.df_bob.columns)
        )

    def test_min_should_ok(self):
        # WHEN
        value = self.df.min(numeric_only=True)

        # THEN
        expected_alice = self.df_alice.min(numeric_only=True)
        self.assertEqual(value['a3'], expected_alice['a3'])
        expected_bob = self.df_bob.min(numeric_only=True)
        pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)

    def test_max_should_ok(self):
        # WHEN
        value = self.df.max(numeric_only=True)

        # THEN
        expected_alice = self.df_alice.max(numeric_only=True)
        self.assertEqual(value['a3'], expected_alice['a3'])
        expected_bob = self.df_bob.max(numeric_only=True)
        pd.testing.assert_series_equal(value[['b4', 'b6']], expected_bob)

    def test_mean_should_ok(self):
        # WHEN
        value = self.df.mean(numeric_only=True)

        # THEN
        expected_alice = self.df_alice.mean(numeric_only=True)
        pd.testing.assert_series_equal(value[expected_alice.index], expected_alice)
        expected_bob = self.df_bob.mean(numeric_only=True)
        pd.testing.assert_series_equal(value[expected_bob.index], expected_bob)

    def test_count_should_ok(self):
        # WHEN
        value = self.df.count()

        # THEN
        expected_alice = pd.concat([self.df_alice, self.df_bob], axis=1).count()
        pd.testing.assert_series_equal(value, expected_alice)

    def test_get_single_item_should_ok(self):
        # WHEN
        value = self.df['a1']

        # THEN
        expected_alice = self.df_alice[['a1']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )

    def test_get_non_exist_items_should_error(self):
        # WHEN and THEN
        with self.assertRaisesRegex(NotFoundError, 'does not exist'):
            self.df['a1', 'non_exist']

    def test_get_multi_items_should_ok(self):
        # WHEN
        value = self.df[['a1', 'b4']]
        # THEN
        expected_alice = self.df_alice[['a1']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )
        expected_bob = self.df_bob[['b4']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data), expected_bob
        )

        # WHEN
        value = self.df[['a1', 'a2', 'b5']]
        # THEN
        expected_alice = self.df_alice[['a1', 'a2']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )
        expected_bob = self.df_bob[['b5']]
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data), expected_bob
        )

    def test_set_item_should_ok(self):
        # Case 1: single item.
        # WHEN
        value = self.df
        value['a1'] = 'test'
        # THEN
        expected_alice = self.df_alice.copy(deep=True)
        expected_alice['a1'] = 'test'
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )

        # Case 2: multi items on different parties.
        # WHEN
        value = self.df
        value[['a1', 'b4', 'b5']] = 'test'
        # THEN
        expected_alice = self.df_alice.copy(deep=True)
        expected_alice['a1'] = 'test'
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )
        expected_bob = self.df_bob.copy(deep=True)
        expected_bob[['b4', 'b5']] = 'test'
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data), expected_bob
        )

    def test_set_item_on_partition_should_ok(self):
        # WHEN
        value = self.df
        value['a1'] = value['a2']
        # THEN
        expected_alice = self.df_alice.copy(deep=True)
        expected_alice['a1'] = expected_alice['a2']
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )

    def test_set_item_on_non_exist_partition_should_error(self):
        # WHEN
        with self.assertRaisesRegex(
            AssertionError,
            'Device of the partition to assgin is not in this dataframe devices.',
        ):
            part = Partition(
                self.carol(
                    lambda: pd.DataFrame(
                        {
                            'a1': ['K5', 'K1', None, 'K6'],
                            'a2': ['A5', 'A1', 'A2', 'A6'],
                            'a3': [5, 1, 2, 6],
                        }
                    )
                )()
            )
            value = self.df
            value['a1'] = part['a2']

    def test_set_item_on_vdataframe_should_ok(self):
        # WHEN
        value = self.df
        value[['a1', 'b4']] = self.df[['a2', 'b5']]

        # THEN
        expected_alice = self.df_alice.copy(deep=True)
        expected_alice['a1'] = expected_alice['a2']
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data), expected_alice
        )

        expected_bob = self.df_bob.copy(deep=True)
        expected_bob['b4'] = expected_bob['b5']
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data), expected_bob
        )

    def test_set_item_on_different_vdataframe_should_error(self):
        with self.assertRaisesRegex(
            AssertionError,
            'Partitions to assgin is not same with this dataframe partitions.',
        ):
            df = VDataFrame(
                {
                    self.alice: Partition(
                        data=self.alice(
                            lambda: pd.DataFrame(
                                {
                                    'a1': ['K5', 'K1', None, 'K6'],
                                    'a2': ['A5', 'A1', 'A2', 'A6'],
                                    'a3': [5, 1, 2, 6],
                                }
                            )
                        )()
                    ),
                    self.carol: Partition(
                        data=self.carol(
                            lambda: pd.DataFrame(
                                {
                                    'b4': [10.2, 20.5, None, -0.4],
                                    'b5': ['B3', None, 'B9', 'B4'],
                                    'b6': [3, 1, 9, 4],
                                }
                            )
                        )()
                    ),
                }
            )
            value = self.df
            value[['a1', 'b4']] = df[['a2', 'b5']]

    def test_drop(self):
        # Case 1: not inplace.
        # WHEN
        value = self.df.drop(columns='a1', inplace=False)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data),
            self.df_alice.drop(columns='a1', inplace=False),
        )
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data), self.df_bob
        )

        # Case 2: inplace.
        # WHEN
        value = self.df.copy()
        value.drop(columns=['a1', 'b4', 'b5'], inplace=True)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data),
            self.df_alice.drop(columns='a1', inplace=False),
        )
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data),
            self.df_bob.drop(columns=['b4', 'b5'], inplace=False),
        )

    def test_fillna(self):
        # Case 1: not inplace.
        # WHEN
        value = self.df.fillna(value='test', inplace=False)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data),
            self.df_alice.fillna(value='test', inplace=False),
        )
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data),
            self.df_bob.fillna(value='test', inplace=False),
        )

        # Case 2: inplace.
        # WHEN
        value = self.df.copy()
        value.fillna(value='test', inplace=True)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.alice].data),
            self.df_alice.fillna(value='test', inplace=False),
        )
        pd.testing.assert_frame_equal(
            reveal(value.partitions[self.bob].data),
            self.df_bob.fillna(value='test', inplace=False),
        )

    def test_astype_should_ok(self):
        # GIVEN
        df = self.df[['a3', 'b4']].fillna(value=1, inplace=False)

        # Case 1: single dtype
        # WHEN
        new_df = df.astype(np.int32)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(new_df.partitions[self.alice].data),
            self.df_alice[['a3']].fillna(1).astype(np.int32),
        )
        pd.testing.assert_frame_equal(
            reveal(new_df.partitions[self.bob].data),
            self.df_bob[['b4']].fillna(1).astype(np.int32),
        )

        # Case 2: dtype dict.
        # WHEN
        dtype = {'a3': np.int32}
        new_df = df.astype(dtype)
        # THEN
        pd.testing.assert_frame_equal(
            reveal(new_df.partitions[self.alice].data),
            self.df_alice[['a3']].fillna(1).astype(dtype),
        )
        pd.testing.assert_frame_equal(
            reveal(new_df.partitions[self.bob].data),
            self.df_bob[['b4']].fillna(1),
        )
