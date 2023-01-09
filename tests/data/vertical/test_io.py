import os
import tempfile

import numpy as np
import pandas as pd
import unittest

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame, read_csv, to_csv
from tests.basecase import MultiDriverDeviceTestCase


class TestVDataFrameIO(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        df1 = pd.DataFrame(
            {
                'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
                'c3': [5, 1, 2, 6, 4, 3],
            }
        )

        df2 = pd.DataFrame(
            {
                'c1': ['K3', 'K1', 'K9', 'K4'],
                'c4': ['B3', 'B1', 'B9', 'B4'],
                'c5': [3, 1, 9, 4],
            }
        )

        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()

        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        cls.filepath = {cls.alice: path1, cls.bob: path2}

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        for path in cls.filepath.values():
            os.remove(path)

    @staticmethod
    def cleartmp(paths):
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_read_csv(self):
        df = read_csv(
            self.filepath, spu=self.spu, keys='c1', drop_keys={self.alice: 'c1'}
        )

        expected_alice = pd.DataFrame({'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]})
        df_alice = reveal(df.partitions[self.alice].data)
        pd.testing.assert_frame_equal(df_alice.reset_index(drop=True), expected_alice)

        expected_bob = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c4': ['B1', 'B3', 'B4'], 'c5': [1, 3, 4]}
        )
        df_bob = reveal(df.partitions[self.bob].data)
        pd.testing.assert_frame_equal(df_bob.reset_index(drop=True), expected_bob)

    def test_read_csv_drop_keys(self):
        df = read_csv(self.filepath, spu=self.spu, keys='c1', drop_keys='c1')

        expected = pd.DataFrame({'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]})
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.alice].data).reset_index(drop=True), expected
        )

        expected = pd.DataFrame({'c4': ['B1', 'B3', 'B4'], 'c5': [1, 3, 4]})
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.bob].data).reset_index(drop=True), expected
        )

    def test_read_csv_with_dtypes(self):
        dtypes = {
            self.alice: {'c1': str, 'c2': str},
            self.bob: {'c1': str, 'c5': np.int64},
        }
        df = read_csv(
            self.filepath, spu=self.spu, keys='c1', dtypes=dtypes, drop_keys='c1'
        )

        expected = pd.DataFrame({'c2': ['A1', 'A3', 'A4']})
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.alice].data).reset_index(drop=True), expected
        )

        expected = pd.DataFrame({'c5': [1, 3, 4]})
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.bob].data).reset_index(drop=True), expected
        )

    @unittest.skip('spu reset not works now FIXME @raofei')
    def read_csv_mismatch_dtypes(self):
        dtypes = {
            self.alice: {'c1': str, 'c6': str},
            self.bob: {'c1': str, 'c5': np.int64},
        }
        with self.assertRaisesRegex(ValueError, 'Usecols do not match columns'):
            read_csv(
                self.filepath, spu=self.spu, keys='c1', dtypes=dtypes, drop_keys='c1'
            )

        # reset spu to clear corrupted state
        self.spu.reset()

    def test_read_csv_duplicated_cols(self):
        df1 = pd.DataFrame(
            {
                'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
                'c3': [5, 1, 2, 6, 4, 3],
            }
        )

        df2 = pd.DataFrame(
            {
                'c1': ['K3', 'K1', 'K9', 'K4'],
                'c2': ['B3', 'B1', 'B9', 'B4'],
                'c5': [3, 1, 9, 4],
            }
        )

        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()

        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        filepath = {self.alice: path1, self.bob: path2}
        with self.assertRaisesRegex(AssertionError, 'duplicate in multiple devices'):
            read_csv(filepath, spu=self.spu, keys='c1', drop_keys='c1')

        for path in filepath.values():
            os.remove(path)

    def test_read_csv_drop_keys_out_of_scope(self):
        df1 = pd.DataFrame(
            {
                'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
                'c3': [5, 1, 2, 6, 4, 3],
            }
        )

        df2 = pd.DataFrame(
            {
                'c1': ['K3', 'K1', 'K9', 'K4'],
                'c2': ['B3', 'B1', 'B9', 'B4'],
                'c5': [3, 1, 9, 4],
            }
        )

        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()

        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        filepath = {self.alice: path1, self.bob: path2}
        with self.assertRaisesRegex(
            AssertionError, 'can not find on device_psi_key_set of device'
        ):
            read_csv(
                filepath,
                spu=self.spu,
                keys=['c1', 'c2'],
                drop_keys={self.alice: ['c1', 'c3'], self.bob: ['c2']},
            )

        for path in filepath.values():
            os.remove(path)

    def test_read_csv_without_psi(self):
        df1 = pd.DataFrame({'c2': ['A5', 'A1', 'A2', 'A6'], 'c3': [5, 1, 2, 6]})

        df2 = pd.DataFrame({'c4': ['B3', 'B1', 'B9', 'B4'], 'c5': [3, 1, 9, 4]})

        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()

        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        filepath = {self.alice: path1, self.bob: path2}
        dtypes = {
            self.alice: {'c2': str, 'c3': np.int64},
            self.bob: {'c4': str, 'c5': np.int64},
        }
        df = read_csv(filepath, dtypes=dtypes)

        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.alice].data).reset_index(drop=True), df1
        )
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.bob].data).reset_index(drop=True), df2
        )

        self.cleartmp([path1, path2])

    def test_read_csv_without_psi_mismatch_length(self):
        df1 = pd.DataFrame(
            {'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'], 'c3': [5, 1, 2, 6, 4, 3]}
        )

        df2 = pd.DataFrame({'c4': ['B3', 'B1', 'B9', 'B4'], 'c5': [3, 1, 9, 4]})

        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()

        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        filepath = {self.alice: path1, self.bob: path2}
        dtypes = {
            self.alice: {'c2': str, 'c3': np.int64},
            self.bob: {'c4': str, 'c5': np.int64},
        }
        with self.assertRaisesRegex(AssertionError, 'number of samples must be equal'):
            read_csv(filepath, dtypes=dtypes)

        self.cleartmp([path1, path2])

    def test_to_csv_should_ok(self):
        # GIVEN
        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()
        file_uris = {self.alice: path1, self.bob: path2}
        df1 = pd.DataFrame({'c2': ['A5', 'A1', 'A2', 'A6'], 'c3': [5, 1, 2, 6]})

        df2 = pd.DataFrame({'c4': ['B3', 'B1', 'B9', 'B4'], 'c5': [3, 1, 9, 4]})

        df = VDataFrame(
            {
                self.alice: Partition(self.alice(lambda df: df)(df1)),
                self.bob: Partition(self.bob(lambda df: df)(df2)),
            }
        )

        # WHEN
        to_csv(df, file_uris, index=False)

        # THEN
        # Waiting a while for to_csv finish.
        import time

        time.sleep(5)
        actual_df = read_csv(file_uris)
        pd.testing.assert_frame_equal(
            reveal(actual_df.partitions[self.alice].data), df1
        )
        pd.testing.assert_frame_equal(reveal(actual_df.partitions[self.bob].data), df2)
        self.cleartmp([path1, path2])
