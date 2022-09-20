import numpy as np
import pandas as pd
import os
import tempfile

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.ndarray import load, shuffle, train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.utils.errors import InvalidArgumentError

from tests.basecase import DeviceTestCase, array_equal


class TestFedNdarray(DeviceTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _, alice_path = tempfile.mkstemp()
        _, bob_path = tempfile.mkstemp()
        alice_arr = np.random.rand(20, 10)
        bob_arr = np.random.rand(10, 10)
        np.save(alice_path, alice_arr, allow_pickle=False)
        np.save(bob_path, bob_arr, allow_pickle=False)
        cls.path = {cls.alice: f'{alice_path}.npy', cls.bob: f'{bob_path}.npy'}
        cls.alice_arr = alice_arr
        cls.bob_arr = bob_arr

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        try:
            for filepath in cls.path.values():
                os.remove(filepath)
        except OSError:
            pass

    def test_load_file_should_ok(self):
        # WHEN
        fed_arr = load(self.path, allow_pickle=True)
        # THEN
        self.assertTrue(
            array_equal(reveal(fed_arr.partitions[self.alice]), self.alice_arr)
        )
        self.assertTrue(array_equal(reveal(fed_arr.partitions[self.bob]), self.bob_arr))

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
        with self.assertRaisesRegex(
            InvalidArgumentError, 'Device of source differs with its key.'
        ):
            load({self.alice: bob_arr, self.bob: alice_arr})

    def test_train_test_split_on_hdataframe_should_ok(self):
        # GIVEN
        fed_arr = load(self.path, allow_pickle=True)

        # WHEN
        fed_arr0, fed_arr1 = train_test_split(fed_arr, 0.6, shuffle=False)

        # THEN
        self.assertTrue(
            array_equal(
                np.concatenate(
                    [
                        reveal(fed_arr0.partitions[self.alice]),
                        reveal(fed_arr1.partitions[self.alice]),
                    ],
                    axis=0,
                ),
                self.alice_arr,
            )
        )
        self.assertTrue(
            array_equal(
                np.concatenate(
                    [
                        reveal(fed_arr0.partitions[self.bob]),
                        reveal(fed_arr1.partitions[self.bob]),
                    ],
                    axis=0,
                ),
                self.bob_arr,
            )
        )

    def test_train_test_split_on_vdataframe_should_ok(self):
        # GIVEN
        df_alice = pd.DataFrame(
            {
                'id': [1, 2, 3, 4],
                'a1': ['K5', 'K1', None, 'K6'],
                'a2': ['A5', 'A1', 'A2', 'A6'],
                'a3': [5, 1, 2, 6],
            }
        )

        df_bob = pd.DataFrame(
            {
                'id': [1, 2, 3, 4],
                'b4': [10.2, 20.5, None, -0.4],
                'b5': ['B3', None, 'B9', 'B4'],
                'b6': [3, 1, 9, 4],
            }
        )
        df = VDataFrame(
            {
                self.alice: Partition(data=self.alice(lambda: df_alice)()),
                self.bob: Partition(data=self.bob(lambda: df_bob)()),
            }
        )
        fed_arr = df.values

        # WHEN
        fed_arr0, fed_arr1 = train_test_split(fed_arr, 0.6, shuffle=False)

        # THEN
        np.testing.assert_equal(
            reveal(fed_arr0.partitions[self.alice])[:, 0],
            reveal(fed_arr0.partitions[self.bob])[:, 0],
        )
        np.testing.assert_equal(
            reveal(fed_arr1.partitions[self.alice])[:, 0],
            reveal(fed_arr1.partitions[self.bob])[:, 0],
        )

    def test_shuffle_should_ok(self):
        # GIVEN
        df_alice = pd.DataFrame(
            {
                'id': [1, 2, 3, 4],
                'a1': ['K5', 'K1', None, 'K6'],
                'a2': ['A5', 'A1', 'A2', 'A6'],
                'a3': [5, 1, 2, 6],
            }
        )

        df_bob = pd.DataFrame(
            {
                'id': [1, 2, 3, 4],
                'b4': [10.2, 20.5, None, -0.4],
                'b5': ['B3', None, 'B9', 'B4'],
                'b6': [3, 1, 9, 4],
            }
        )
        df = VDataFrame(
            {
                self.alice: Partition(data=self.alice(lambda: df_alice)()),
                self.bob: Partition(data=self.bob(lambda: df_bob)()),
            }
        )
        fed_arr = df.values

        # WHEN
        shuffle(fed_arr)

        # THEN
        np.testing.assert_equal(
            reveal(fed_arr.partitions[self.alice])[:, 0],
            reveal(fed_arr.partitions[self.bob])[:, 0],
        )

    def test_load_npz(self):
        # GIVEN
        _, alice_path = tempfile.mkstemp()
        _, bob_path = tempfile.mkstemp()
        alice_train = np.random.rand(20, 10)
        alice_test = np.random.rand(20, 1)
        bob_train = np.random.rand(10, 10)
        bob_test = np.random.rand(10, 1)
        np.savez(alice_path, train=alice_train, test=alice_test)
        np.savez(bob_path, train=bob_train, test=bob_test)
        # WHEN
        data = load(
            {self.alice: f'{alice_path}.npz', self.bob: f'{bob_path}.npz'},
            allow_pickle=True,
        )

        # THEN
        for k, v in data.items():
            if k == 'train':
                self.assertTrue(
                    array_equal(reveal(v.partitions[self.alice]), alice_train)
                )
                self.assertTrue(array_equal(reveal(v.partitions[self.bob]), bob_train))
            else:
                self.assertTrue(
                    array_equal(reveal(v.partitions[self.alice]), alice_test)
                )
                self.assertTrue(array_equal(reveal(v.partitions[self.bob]), bob_test))

    def test_astype_should_ok(self):
        # WHEN
        fed_arr = load(self.path)
        fed_arr = fed_arr.astype(str)

        # THEN
        self.assertTrue(
            array_equal(
                reveal(fed_arr.partitions[self.alice]), self.alice_arr.astype(str)
            )
        )
        self.assertTrue(
            array_equal(reveal(fed_arr.partitions[self.bob]), self.bob_arr.astype(str))
        )
