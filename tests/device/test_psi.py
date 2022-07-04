import os
import shutil

import pandas as pd

import secretflow as sf
from tests.basecase import DeviceTestCase


class TestDevicePSI2PC(DeviceTestCase):
    def setUp(self) -> None:
        da = pd.DataFrame(
            {
                'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
                'c3': [5, 1, 2, 6, 4, 3],
            }
        )

        db = pd.DataFrame(
            {
                'c1': ['K3', 'K1', 'K9', 'K4'],
                'c2': ['B3', 'A1', 'A9', 'A4'],
                'c3': [3, 1, 9, 4],
            }
        )

        db2 = pd.DataFrame(
            {
                'c1': ['K3', 'K1', 'K1', 'K4'],
                'c2': ['B3', 'A1', 'A1', 'A4'],
                'c3': ['C3', 'C1', 'D1', 'C4'],
                'c4': [3, 1, 9, 4],
            }
        )

        db3 = pd.DataFrame(
            {'c1': ['K7', 'K8', 'K9'], 'c2': ['A7', 'A8', 'A9'], 'c3': [7, 8, 9]}
        )

        self.da = sf.to(self.alice, da)
        self.db = sf.to(self.bob, db)
        self.db2 = sf.to(self.bob, db2)
        self.db3 = sf.to(self.bob, db3)
        self.dc = sf.to(self.carol, db)

    def test_single_col(self):
        da, db = self.spu.psi_df('c1', [self.da, self.db])

        expected = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)

        expected = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'B3', 'A4'], 'c3': [1, 3, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)

    def test_multiple_col(self):
        da, db = self.spu.psi_df(['c1', 'c2'], [self.da, self.db])

        expected = pd.DataFrame({'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]})
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)

    def test_invalid_device(self):
        with self.assertRaisesRegex(AssertionError, 'not co-located'):
            da, dc = self.spu.psi_df(['c1', 'c2'], [self.da, self.dc])
            sf.reveal([da, dc])

    def test_duplicate_col(self):
        with self.assertRaisesRegex(RuntimeError, 'Found duplicated keys'):
            da, db = self.spu.psi_df(['c1'], [self.da, self.db2])
            sf.reveal([da, db])

        # reset spu to clear corrupted state
        self.spu.reset()

    def test_missing_col(self):
        with self.assertRaisesRegex(RuntimeError, "can't find feature names 'c4'"):
            da, db = self.spu.psi_df(['c1', 'c4'], [self.da, self.db2])
            sf.reveal([da, db])

        # reset spu to clear corrupted state
        self.spu.reset()

    def test_no_intersection(self):
        da, db = self.spu.psi_df('c1', [self.da, self.db3])
        expected = pd.DataFrame({'c1': [], 'c2': [], 'c3': []}).astype('object')
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)

    def test_psi_csv(self):
        input_path = {self.alice: '.data/alice.csv', self.bob: '.data/bob.csv'}
        output_path = {self.alice: '.data/alice_psi.csv', self.bob: '.data/bob_psi.csv'}

        os.makedirs('.data', exist_ok=True)
        sf.reveal(self.da).to_csv(input_path[self.alice], index=False)
        sf.reveal(self.db).to_csv(input_path[self.bob], index=False)

        self.spu.psi_csv(['c1', 'c2'], input_path, output_path)

        expected = pd.DataFrame({'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]})
        da = pd.read_csv(output_path[self.alice])
        db = pd.read_csv(output_path[self.bob])
        pd.testing.assert_frame_equal(da, expected)
        pd.testing.assert_frame_equal(db, expected)
        shutil.rmtree('.data', ignore_errors=True)


class TestDevicePSI3PC(DeviceTestCase):
    def setUp(self) -> None:
        self.spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

        da = pd.DataFrame(
            {
                'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
                'c2': ['A5', 'A1', 'A2', 'A6', 'B4', 'A3'],
                'c3': [5, 1, 2, 6, 4, 3],
            }
        )

        db = pd.DataFrame(
            {
                'c1': ['K3', 'K1', 'K9', 'K4'],
                'c2': ['B3', 'A1', 'A9', 'A4'],
                'c3': [3, 1, 9, 4],
            }
        )

        dc = pd.DataFrame(
            {
                'c1': ['K9', 'K4', 'K3', 'K1', 'k8'],
                'c2': ['A9', 'B4', 'B3', 'A1', 'k8'],
                'c3': [9, 4, 3, 1, 8],
            }
        )

        self.da = sf.to(self.alice, da)
        self.db = sf.to(self.bob, db)
        self.dc = sf.to(self.carol, dc)

    def test_single_col(self):
        da, db, dc = self.spu.psi_df('c1', [self.da, self.db, self.dc], protocol='ecdh')

        expected = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'A3', 'B4'], 'c3': [1, 3, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)

        expected = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'B3', 'A4'], 'c3': [1, 3, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)

        expected = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'B3', 'B4'], 'c3': [1, 3, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(dc).reset_index(drop=True), expected)

    def test_multiple_col(self):
        da, db, dc = self.spu.psi_df(
            ['c1', 'c2'], [self.da, self.db, self.dc], protocol='ecdh'
        )
        expected = pd.DataFrame({'c1': ['K1'], 'c2': ['A1'], 'c3': [1]})
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(sf.reveal(dc).reset_index(drop=True), expected)
