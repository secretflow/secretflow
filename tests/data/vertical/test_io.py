import os
import tempfile

import numpy as np
import pandas as pd

import secretflow.data.vertical as vd
from tests.basecase import DeviceTestCase
from secretflow import reveal


class TestReadCSV(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        df1 = pd.DataFrame({'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                            'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
                            'c3': [5, 1, 2, 6, 4, 3]})

        df2 = pd.DataFrame({'c1': ['K3', 'K1', 'K9', 'K4'],
                            'c4': ['B3', 'B1', 'B9', 'B4'],
                            'c5': [3, 1, 9, 4]})

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

    def test_read_csv(self):
        df = vd.read_csv(self.filepath, ppu=self.ppu,
                         keys='c1', drop_keys=False)

        expected_alice = pd.DataFrame({'c1': ['K1', 'K3', 'K4'],
                                       'c2': ['A1', 'A3', 'A4'],
                                       'c3': [1, 3, 4]})
        df_alice = reveal(df.partitions[self.alice].data)
        pd.testing.assert_frame_equal(
            df_alice.reset_index(drop=True), expected_alice)

        expected_bob = pd.DataFrame({'c1': ['K1', 'K3', 'K4'],
                                     'c4': ['B1', 'B3', 'B4'],
                                     'c5': [1, 3, 4]})
        df_bob = reveal(df.partitions[self.bob].data)
        pd.testing.assert_frame_equal(
            df_bob.reset_index(drop=True), expected_bob)

    def test_read_csv_drop_keys(self):
        df = vd.read_csv(self.filepath, ppu=self.ppu,
                         keys='c1', drop_keys=True)

        expected = pd.DataFrame({'c2': ['A1', 'A3', 'A4'],
                                 'c3': [1, 3, 4]})
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.alice].data).reset_index(drop=True), expected)

        expected = pd.DataFrame({'c4': ['B1', 'B3', 'B4'],
                                 'c5': [1, 3, 4]})
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.bob].data).reset_index(drop=True), expected)

    def test_read_csv_with_dtypes(self):
        dtypes = {self.alice: {'c1': np.str, 'c2': np.str},
                  self.bob: {'c1': np.str, 'c5': np.int64}}
        df = vd.read_csv(self.filepath, ppu=self.ppu,
                         keys='c1', dtypes=dtypes, drop_keys=True)

        expected = pd.DataFrame({'c2': ['A1', 'A3', 'A4']})
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.alice].data).reset_index(drop=True), expected)

        expected = pd.DataFrame({'c5': [1, 3, 4]})
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.bob].data).reset_index(drop=True), expected)

    def test_read_csv_mismatch_dtypes(self):
        dtypes = {self.alice: {'c1': np.str, 'c6': np.str},
                  self.bob: {'c1': np.str, 'c5': np.int64}}
        with self.assertRaisesRegex(ValueError, 'Usecols do not match columns'):
            vd.read_csv(self.filepath, ppu=self.ppu, keys='c1',
                        dtypes=dtypes, drop_keys=True)

        # reset ppu to clear corrupted state
        self.ppu.reset()

    def test_read_csv_duplicated_cols(self):
        df1 = pd.DataFrame({'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                            'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
                            'c3': [5, 1, 2, 6, 4, 3]})

        df2 = pd.DataFrame({'c1': ['K3', 'K1', 'K9', 'K4'],
                            'c2': ['B3', 'B1', 'B9', 'B4'],
                            'c5': [3, 1, 9, 4]})

        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()

        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        filepath = {self.alice: path1, self.bob: path2}
        with self.assertRaisesRegex(AssertionError, 'duplicate in multiple devices'):
            vd.read_csv(filepath, ppu=self.ppu, keys='c1', drop_keys=True)

        for path in filepath.values():
            os.remove(path)

    def test_read_csv_without_psi(self):
        df1 = pd.DataFrame({'c2': ['A5', 'A1', 'A2', 'A6'],
                            'c3': [5, 1, 2, 6]})

        df2 = pd.DataFrame({'c4': ['B3', 'B1', 'B9', 'B4'],
                            'c5': [3, 1, 9, 4]})

        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()

        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        filepath = {self.alice: path1, self.bob: path2}
        dtypes = {self.alice: {'c2': np.str, 'c3': np.int64},
                  self.bob: {'c4': np.str, 'c5': np.int64}}
        df = vd.read_csv(filepath, dtypes=dtypes)

        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.alice].data).reset_index(drop=True), df1)
        pd.testing.assert_frame_equal(
            reveal(df.partitions[self.bob].data).reset_index(drop=True), df2)

    def test_read_csv_without_psi_mismatch_length(self):
        df1 = pd.DataFrame({'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
                            'c3': [5, 1, 2, 6, 4, 3]})

        df2 = pd.DataFrame({'c4': ['B3', 'B1', 'B9', 'B4'],
                            'c5': [3, 1, 9, 4]})

        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()

        df1.to_csv(path1, index=False)
        df2.to_csv(path2, index=False)

        filepath = {self.alice: path1, self.bob: path2}
        dtypes = {self.alice: {'c2': np.str, 'c3': np.int64},
                  self.bob: {'c4': np.str, 'c5': np.int64}}
        with self.assertRaisesRegex(AssertionError, 'number of samples must be equal'):
            vd.read_csv(filepath, dtypes=dtypes)
