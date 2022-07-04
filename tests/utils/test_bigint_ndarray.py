import unittest
import numpy as np

from heu import phe
from secretflow.utils import ndarray_bigint


class BigintNdarrayCase(unittest.TestCase):
    def test_arange(self):
        a = ndarray_bigint.arange(120)
        a.resize((2, 3, 4, 5))
        self.assertTrue((np.arange(120).reshape((2, 3, 4, 5)) == a.to_numpy()).all())

    def test_randbits(self):
        a = ndarray_bigint.randbits((100000,), 8)
        res = {i: 0 for i in range(-128, 128)}
        for i in a.data:
            self.assertGreaterEqual(i, -128)
            self.assertLessEqual(i, 127)
            res[i] += 1
        for i, c in res.items():
            self.assertGreater(c, 0, f"cannot generate randint {i}")

    def test_randint(self):
        bound = 2**2048
        a = ndarray_bigint.randint((100,), -bound, bound)
        uint128_max = 2**128 - 1
        # a is much bigger than uint128
        self.assertTrue((a.to_numpy() > uint128_max).any())
        self.assertTrue((a.to_numpy() < -uint128_max).any())

    def test_add(self):
        arrays = [ndarray_bigint.randbits((3, 4), 16) for _ in range(10)]
        array_sum = sum(arrays, ndarray_bigint.zeros((3, 4)))

        np_array_sum = sum([a.to_numpy() for a in arrays])
        self.assertTrue((array_sum.to_numpy() == np_array_sum).all())

    def test_to_bytes(self):
        array = ndarray_bigint.arange(300)
        b1 = array.to_bytes(1)
        self.assertTrue(len(b1), 300)
        b2 = array.to_bytes(2)
        self.assertEqual(len(b2), 600)

    def test_to_numpy(self):
        # int128 case
        array = ndarray_bigint.randbits((500, 300), 64).to_numpy()
        self.assertEqual(array.dtype, np.int64)
        array = ndarray_bigint.randbits((500, 300), 65).to_numpy()
        self.assertEqual(array.dtype, object)
        array = ndarray_bigint.randbits((500, 300), 128).to_numpy()
        self.assertEqual(array.dtype, object)
        self.assertTrue(isinstance(array[0][0], int))

        # big int case
        array = ndarray_bigint.randbits((2, 3), 512)
        array_np = array.to_numpy()
        array_pt = np.vectorize(lambda x: phe.Plaintext(x))(array_np)
        self.assertTrue(isinstance(array_pt[0][0], phe.Plaintext))
        self.assertEqual(array.data[0], int(array_pt[0][0]))


if __name__ == '__main__':
    unittest.main()
