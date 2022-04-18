import numpy as np
import pandas as pd

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.io import util as io_util
from secretflow.data.ndarray import (load, shuffle,
                                     train_test_split)
from secretflow.data.vertical import VDataFrame
from secretflow.utils.errors import InvalidArgumentError
from tests.basecase import DeviceTestCase, array_equal


class TestFedNdarray(DeviceTestCase):
    def test_load_file_should_ok(self):
        # GIVEN
        file_alice = 'tests/datasets/adult/horizontal/adult.alice.npy'
        file_bob = 'tests/datasets/adult/horizontal/adult.bob.npy'

        # WHEN
        fed_arr = load(
            {self.alice: file_alice, self.bob: file_bob}, allow_pickle=True)
        # THEN
        alice_arr = np.load(io_util.open(file_alice), allow_pickle=True)
        bob_arr = np.load(io_util.open(file_bob), allow_pickle=True)
        self.assertTrue(array_equal(
            reveal(fed_arr.partitions[self.alice]), alice_arr))
        self.assertTrue(array_equal(
            reveal(fed_arr.partitions[self.bob]), bob_arr))

    def test_load_pyu_object_should_ok(self):
        # GIVEN
        alice_arr = self.alice(lambda: np.array([[1, 2, 3], [4, 5, 6]]))()
        bob_arr = self.bob(lambda: np.array([[11, 12, 13], [14, 15, 16]]))()

        # WHEN
        fed_arr = load({self.alice: alice_arr, self.bob: bob_arr})

        # THEN
        self.assertEqual(fed_arr.partitions[self.alice], alice_arr)
        self.assertEqual(fed_arr.partitions[self.bob], bob_arr)

    def test_load_should_error_with_wrong_pyu_object(self):
        # GIVEN
        alice_arr = self.alice(lambda: np.array([[1, 2, 3], [4, 5, 6]]))()
        bob_arr = self.bob(lambda: np.array([[11, 12, 13], [14, 15, 16]]))()

        # WHEN & THEN
        with self.assertRaisesRegex(InvalidArgumentError, 'Device of source differs with its key.'):
            load({self.alice: bob_arr, self.bob: alice_arr})

    def test_train_test_split_on_hdataframe_should_ok(self):
        # GIVEN
        file_alice = 'tests/datasets/adult/horizontal/adult.alice.npy'
        file_bob = 'tests/datasets/adult/horizontal/adult.bob.npy'
        fed_arr = load(
            {self.alice: file_alice, self.bob: file_bob}, allow_pickle=True)

        # WHEN
        fed_arr0, fed_arr1 = train_test_split(fed_arr, 0.6, shuffle=False)

        # THEN
        alice_arr = np.load(io_util.open(file_alice), allow_pickle=True)
        bob_arr = np.load(io_util.open(file_bob), allow_pickle=True)
        self.assertTrue(
            array_equal(
                np.concatenate(
                    [reveal(fed_arr0.partitions[self.alice]),
                     reveal(fed_arr1.partitions[self.alice])], axis=0),
                alice_arr))
        self.assertTrue(
            array_equal(
                np.concatenate(
                    [reveal(fed_arr0.partitions[self.bob]),
                     reveal(fed_arr1.partitions[self.bob])], axis=0),
                bob_arr))

    def test_train_test_split_on_vdataframe_should_ok(self):
        # GIVEN
        df_alice = pd.DataFrame({'id': [1, 2, 3, 4],
                                 'a1': ['K5', 'K1', None, 'K6'],
                                 'a2': ['A5', 'A1', 'A2', 'A6'],
                                 'a3': [5, 1, 2, 6]})

        df_bob = pd.DataFrame({'id': [1, 2, 3, 4],
                               'b4': [10.2, 20.5, None, -0.4],
                               'b5': ['B3', None, 'B9', 'B4'],
                               'b6': [3, 1, 9, 4]})
        df = VDataFrame(
            {self.alice: Partition(data=self.alice(lambda: df_alice)()),
             self.bob: Partition(data=self.bob(lambda: df_bob)())})
        fed_arr = df.values

        # WHEN
        fed_arr0, fed_arr1 = train_test_split(fed_arr, 0.6, shuffle=False)

        # THEN
        np.testing.assert_equal(
            reveal(fed_arr0.partitions[self.alice])[:, 0],
            reveal(fed_arr0.partitions[self.bob])[:, 0])
        np.testing.assert_equal(
            reveal(fed_arr1.partitions[self.alice])[:, 0],
            reveal(fed_arr1.partitions[self.bob])[:, 0])

    def test_shuffle_should_ok(self):
        # GIVEN
        df_alice = pd.DataFrame({'id': [1, 2, 3, 4],
                                 'a1': ['K5', 'K1', None, 'K6'],
                                 'a2': ['A5', 'A1', 'A2', 'A6'],
                                 'a3': [5, 1, 2, 6]})

        df_bob = pd.DataFrame({'id': [1, 2, 3, 4],
                               'b4': [10.2, 20.5, None, -0.4],
                               'b5': ['B3', None, 'B9', 'B4'],
                               'b6': [3, 1, 9, 4]})
        df = VDataFrame(
            {self.alice: Partition(data=self.alice(lambda: df_alice)()),
             self.bob: Partition(data=self.bob(lambda: df_bob)())})
        fed_arr = df.values

        # WHEN
        shuffle(fed_arr)

        # THEN
        np.testing.assert_equal(
            reveal(fed_arr.partitions[self.alice])[:, 0],
            reveal(fed_arr.partitions[self.bob])[:, 0])

    def test_load_npz(self):
        file_alice = 'tests/datasets/fraud/horizontal/fraud_detection_train_0_of_2.npz'
        file_bob = 'tests/datasets/fraud/horizontal/fraud_detection_train_1_of_2.npz'
        # WHEN
        data = load(
            {self.alice: file_alice, self.bob: file_bob}, allow_pickle=True)

        # THEN
        alice_arr = np.load(io_util.open(file_alice), allow_pickle=True)

        bob_arr = np.load(io_util.open(file_bob), allow_pickle=True)

        for k, v in data.items():
            self.assertTrue(array_equal(
                reveal(v.partitions[self.alice]), alice_arr[k]))
            self.assertTrue(array_equal(
                reveal(v.partitions[self.bob]), bob_arr[k]))
