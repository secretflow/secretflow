import numpy as np
import pandas as pd
import os
import tempfile
import unittest

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.ndarray import (
    load,
    shuffle,
    train_test_split,
    tss,
    rss,
    r2_score,
    mean_abs_err,
    mean_abs_percent_err,
    subtract,
    histogram,
    residual_histogram,
)
from secretflow.data.vertical import VDataFrame
from secretflow.utils.errors import InvalidArgumentError

from tests.basecase import MultiDriverDeviceTestCase, array_equal
from secretflow.utils.simulation.datasets import create_ndarray
import sklearn.metrics


class TestFedNdarray(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def gen_arr():
            _, path = tempfile.mkstemp()
            arr = np.random.rand(20, 10)
            np.save(path, arr, allow_pickle=False)
            return path, arr

        alice_path, alice_arr = reveal(cls.alice(gen_arr, num_returns=2)())
        bob_path, bob_arr = reveal(cls.alice(gen_arr, num_returns=2)())
        cls.path = {cls.alice: f"{alice_path}.npy", cls.bob: f"{bob_path}.npy"}
        cls.alice_arr = alice_arr
        cls.bob_arr = bob_arr

        cls.y_true = reveal(cls.alice(lambda: np.random.rand(100, 100))())
        cls.y_pred = reveal(
            cls.alice(lambda arr: arr + np.random.rand(100, 100) / 20)(cls.y_true)
        )
        cls.y_true_fed_h = create_ndarray(
            cls.y_true, {cls.alice: 0.3, cls.bob: 0.7}, axis=0
        )
        cls.y_pred_fed_h = create_ndarray(
            cls.y_pred, {cls.alice: 0.3, cls.bob: 0.7}, axis=0
        )
        cls.y_true_fed_v = create_ndarray(cls.y_true, {cls.bob: 1.0}, axis=1)
        cls.y_pred_fed_v = create_ndarray(cls.y_pred, {cls.bob: 1.0}, axis=1)

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
            InvalidArgumentError, "Device of source differs with its key."
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
                "id": [1, 2, 3, 4],
                "a1": ["K5", "K1", None, "K6"],
                "a2": ["A5", "A1", "A2", "A6"],
                "a3": [5, 1, 2, 6],
            }
        )

        df_bob = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "b4": [10.2, 20.5, None, -0.4],
                "b5": ["B3", None, "B9", "B4"],
                "b6": [3, 1, 9, 4],
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
                "id": [1, 2, 3, 4],
                "a1": ["K5", "K1", None, "K6"],
                "a2": ["A5", "A1", "A2", "A6"],
                "a3": [5, 1, 2, 6],
            }
        )

        df_bob = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "b4": [10.2, 20.5, None, -0.4],
                "b5": ["B3", None, "B9", "B4"],
                "b6": [3, 1, 9, 4],
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
        def gen_arr():
            _, path = tempfile.mkstemp()
            train = np.random.rand(20, 10)
            test = np.random.rand(10, 1)
            np.savez(path, train=train, test=test)
            return path, train, test

        alice_path, alice_train, alice_test = reveal(self.alice(gen_arr)())
        bob_path, bob_train, bob_test = reveal(self.alice(gen_arr)())
        # _, bob_path = tempfile.mkstemp()
        # alice_train = np.random.rand(20, 10)
        # alice_test = np.random.rand(20, 1)
        # bob_train = np.random.rand(10, 10)
        # bob_test = np.random.rand(10, 1)
        # np.savez(alice_path, train=alice_train, test=alice_test)
        # np.savez(bob_path, train=bob_train, test=bob_test)
        # WHEN
        data = load(
            {self.alice: f"{alice_path}.npz", self.bob: f"{bob_path}.npz"},
            allow_pickle=True,
        )

        # THEN
        for k, v in data.items():
            if k == "train":
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

    def operator_h_v_cases_test(self, test_handle, true_val, binary=True):
        if not binary:
            a_h = reveal(test_handle(self.y_true_fed_h, spu_device=self.spu))
            a_v = reveal(test_handle(self.y_true_fed_v))
        else:
            a_h = reveal(test_handle(self.y_true_fed_h, self.y_pred_fed_h, self.spu))
            a_v = reveal(test_handle(self.y_true_fed_v, self.y_pred_fed_v))
            # Currently mixed case is not supported
        np.testing.assert_almost_equal(true_val, a_h, decimal=2)
        np.testing.assert_almost_equal(true_val, a_v, decimal=2)

    def test_tss(self):
        tss_val = np.sum(np.square(self.y_true - np.mean(self.y_true)))
        self.operator_h_v_cases_test(tss, tss_val, False)

    def test_rss(self):
        rss_val = np.sum(np.square(self.y_true - self.y_pred))
        self.operator_h_v_cases_test(rss, rss_val)

    def test_r2_score(self):
        r2score_val = sklearn.metrics.r2_score(self.y_true, self.y_pred)
        self.operator_h_v_cases_test(r2_score, r2score_val)

    def test_mean_abs_err(self):
        mae_val = sklearn.metrics.mean_absolute_error(self.y_true, self.y_pred)
        self.operator_h_v_cases_test(mean_abs_err, mae_val)

    def test_mean_abs_percent_err(self):
        mape_val = sklearn.metrics.mean_absolute_percentage_error(
            self.y_true, self.y_pred
        )
        self.operator_h_v_cases_test(mean_abs_percent_err, mape_val)

    def test_subtraction(self):
        residual_val = self.y_true - self.y_pred
        self.operator_h_v_cases_test(subtract, residual_val)

    def test_histogram(self):
        hist, edges = np.histogram(self.y_true)
        h_v, e_v = reveal(histogram(self.y_true_fed_v))
        np.testing.assert_almost_equal(hist, reveal(h_v), decimal=2)
        np.testing.assert_almost_equal(edges, reveal(e_v), decimal=2)
        # TODO(zoupeicheng.zpc): pending on spu support for the following.
        # h_h, e_h = reveal(histogram(self.y_true_fed_h, spu_device = self.spu))
        # np.testing.assert_almost_equal(hist, reveal(h_h), decimal=2)
        # np.testing.assert_almost_equal(edges, reveal(e_h), decimal=2)

    @unittest.skip('Not stable @jiuqi')
    def test_residual_histogram(self):
        hist, edges = np.histogram(self.y_true - self.y_pred)
        h_v, e_v = reveal(residual_histogram(self.y_true_fed_v, self.y_pred_fed_v))
        h_v_direct, e_v_direct = reveal(
            histogram(self.y_true_fed_v - self.y_pred_fed_v)
        )
        np.testing.assert_almost_equal(hist, reveal(h_v), decimal=2)
        np.testing.assert_almost_equal(edges, reveal(e_v), decimal=2)
        np.testing.assert_almost_equal(hist, reveal(h_v_direct), decimal=2)
        np.testing.assert_almost_equal(edges, reveal(e_v_direct), decimal=2)
