import os
import shutil

import pandas as pd
import uuid

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

        db4 = pd.DataFrame(
            {
                'c11': ['K3', 'K1', 'K9', 'K4'],
                'c21': ['B3', 'A1', 'A9', 'A4'],
                'c31': [3, 1, 9, 4],
            }
        )

        self.da = sf.to(self.alice, da)
        self.db = sf.to(self.bob, db)
        self.db2 = sf.to(self.bob, db2)
        self.db3 = sf.to(self.bob, db3)
        self.dc = sf.to(self.carol, db)
        self.db4 = sf.to(self.bob, db4)

    def test_single_col(self):

        da, db = self.spu.psi_df('c1', [self.da, self.db], 'alice')

        expected = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)

        expected = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'B3', 'A4'], 'c3': [1, 3, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)

    def test_multiple_col(self):
        da, db = self.spu.psi_df(['c1', 'c2'], [self.da, self.db], 'alice')

        expected = pd.DataFrame({'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]})
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)

    def test_different_cols(self):
        da, db = self.spu.psi_df(
            {self.alice: ['c1', 'c2'], self.bob: ['c11', 'c21']},
            [self.da, self.db4],
            'alice',
        )

        expected_a = pd.DataFrame(
            {'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]}
        )
        expected_b = pd.DataFrame(
            {'c11': ['K1', 'K4'], 'c21': ['A1', 'A4'], 'c31': [1, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected_a)
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected_b)

    def test_invalid_device(self):
        with self.assertRaisesRegex(AssertionError, 'not co-located'):
            da, dc = self.spu.psi_df(['c1', 'c2'], [self.da, self.dc], 'alice')
            sf.reveal([da, dc])

    def test_duplicate_col(self):
        with self.assertRaisesRegex(RuntimeError, 'found duplicated keys'):
            da, db = self.spu.psi_df('c1', [self.da, self.db2], 'alice')
            sf.reveal([da, db])

        # reset spu to clear corrupted state
        self.spu.reset()

    def test_missing_col(self):
        with self.assertRaisesRegex(RuntimeError, "can't find feature names 'c4'"):
            da, db = self.spu.psi_df(['c1', 'c4'], [self.da, self.db2], 'alice')
            sf.reveal([da, db])

        # reset spu to clear corrupted state
        self.spu.reset()

    def test_no_intersection(self):
        da, db = self.spu.psi_df('c1', [self.da, self.db3], 'alice')
        expected = pd.DataFrame({'c1': [], 'c2': [], 'c3': []}).astype('object')
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)

    def test_no_broadcast(self):
        # only alice can get result
        da, db = self.spu.psi_df(
            'c1', [self.da, self.db], 'alice', 'KKRT_PSI_2PC', False, True, False
        )
        expected = pd.DataFrame(
            {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]}
        )
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
        # bob can not get result
        self.assertIsNone(sf.reveal(db))

    def test_psi_csv(self):
        data_dir = f'.data/{uuid.uuid4()}'

        input_path = {
            self.alice: f'{data_dir}/alice.csv',
            self.bob: f'{data_dir}/bob.csv',
        }
        output_path = {
            self.alice: f'{data_dir}/alice_psi.csv',
            self.bob: f'{data_dir}/bob_psi.csv',
        }

        os.makedirs(data_dir, exist_ok=True)
        sf.reveal(self.da).to_csv(input_path[self.alice], index=False)
        sf.reveal(self.db).to_csv(input_path[self.bob], index=False)

        self.spu.psi_csv(['c1', 'c2'], input_path, output_path, 'alice')

        expected = pd.DataFrame({'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]})
        da = pd.read_csv(output_path[self.alice])
        db = pd.read_csv(output_path[self.bob])
        pd.testing.assert_frame_equal(da, expected)
        pd.testing.assert_frame_equal(db, expected)
        shutil.rmtree(data_dir, ignore_errors=True)


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
        da, db, dc = self.spu.psi_df(
            'c1', [self.da, self.db, self.dc], 'alice', protocol='ECDH_PSI_3PC'
        )

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
            ['c1', 'c2'],
            [self.da, self.db, self.dc],
            protocol='ECDH_PSI_3PC',
            receiver='alice',
        )
        expected = pd.DataFrame({'c1': ['K1'], 'c2': ['A1'], 'c3': [1]})
        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(sf.reveal(dc).reset_index(drop=True), expected)


class TestDevicePSIJoin(DeviceTestCase):
    def setUp(self) -> None:
        da = pd.DataFrame(
            {
                'id1': ['K100', 'K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
                'item': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                'feature1': ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'],
            }
        )

        db = pd.DataFrame(
            {
                'id2': ['K200', 'K300', 'K400', 'K500', 'K600', 'K700'],
                'feature2': ['AA', 'BB', 'CC', 'DD', 'EE', 'FF'],
            }
        )

        self.da = sf.to(self.alice, da)
        self.db = sf.to(self.bob, db)

    def test_psi_join_df(self):
        select_keys = {
            self.alice: ['id1'],
            self.bob: ['id2'],
        }

        da, db = self.spu.psi_join_df(select_keys, [self.da, self.db], 'bob', 'bob')

        result_a = pd.DataFrame(
            {
                'id1': ['K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
                'item': ['B', 'C', 'D', 'E', 'F', 'G'],
                'feature1': ['BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'],
            }
        )

        result_b = pd.DataFrame(
            {
                'id2': ['K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
                'feature2': ['AA', 'AA', 'BB', 'CC', 'CC', 'DD'],
            }
        )

        pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), result_a)
        pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), result_b)

    def test_psi_join_csv(self):
        data_dir = f'.data/{uuid.uuid4()}'

        input_path = {
            self.alice: f'{data_dir}/alice.csv',
            self.bob: f'{data_dir}/bob.csv',
        }
        output_path = {
            self.alice: f'{data_dir}/alice_psi.csv',
            self.bob: f'{data_dir}/bob_psi.csv',
        }

        os.makedirs(data_dir, exist_ok=True)
        sf.reveal(self.da).to_csv(input_path[self.alice], index=False)
        sf.reveal(self.db).to_csv(input_path[self.bob], index=False)

        select_keys = {
            self.alice: ['id1'],
            self.bob: ['id2'],
        }

        self.spu.psi_join_csv(select_keys, input_path, output_path, 'alice', 'alice')

        result_a = pd.DataFrame(
            {
                'id1': ['K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
                'item': ['B', 'C', 'D', 'E', 'F', 'G'],
                'feature1': ['BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'],
            }
        )

        result_b = pd.DataFrame(
            {
                'id2': ['K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
                'feature2': ['AA', 'AA', 'BB', 'CC', 'CC', 'DD'],
            }
        )

        da = pd.read_csv(output_path[self.alice])
        db = pd.read_csv(output_path[self.bob])
        pd.testing.assert_frame_equal(da, result_a)
        pd.testing.assert_frame_equal(db, result_b)
        shutil.rmtree(data_dir, ignore_errors=True)
