import numpy as np

from secretflow.utils.compressor import ForwardData, QuantizedData, QuantizedFP


def test_compressed(compressor):
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    c_a = compressor.compress(a)
    assert isinstance(c_a, QuantizedData)

    a_list = [a, a, a]
    c_a_list = compressor.compress(a_list)
    assert np.all([isinstance(x, QuantizedData) for x in c_a_list])

    a_forward = ForwardData(hidden=a)
    c_a_forward = compressor.compress(a_forward)
    assert isinstance(c_a_forward.hidden, QuantizedData)


def test_abs_max_equal(compressor):
    a = np.array([0, -100.0, 1, 2, 3, 4, -5, -6, -7, -8])
    c_a = compressor.compress(a)
    res = compressor.decompress(c_a)
    np.testing.assert_almost_equal(np.max(np.abs(res)), np.max(np.abs(a)))


def test_all_zeros(compressor):
    a = np.zeros([3, 3])
    c_a = compressor.compress(a)
    res = compressor.decompress(c_a)
    assert (res == a).all()


def test_qfp_main():
    compressor = QuantizedFP()
    test_compressed(compressor)
    test_abs_max_equal(compressor)
    test_all_zeros(compressor)


if __name__ == '__main__':
    test_qfp_main()
