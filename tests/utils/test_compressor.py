import numpy as np

from secretflow.utils.compressor import (
    ForwardData,
    QuantizedData,
    QuantizedFP,
    QuantizedKmeans,
)


def compressed_test(compressor):
    a = np.random.normal(0, 5, size=(128, 256))
    c_a = compressor.compress(a)
    assert isinstance(c_a, QuantizedData)

    a_list = [a, a, a]
    c_a_list = compressor.compress(a_list)
    assert np.all([isinstance(x, QuantizedData) for x in c_a_list])

    a_forward = ForwardData(hidden=a)
    c_a_forward = compressor.compress(a_forward)
    assert isinstance(c_a_forward.hidden, QuantizedData)


def abs_max_equal_test(compressor):
    a = np.random.normal(0, 5, size=(128, 256))
    c_a = compressor.compress(a)
    res = compressor.decompress(c_a)
    np.testing.assert_almost_equal(np.max(np.abs(res)), np.max(np.abs(a)))


def all_zeros_test(compressor):
    a = np.zeros((128, 256))
    c_a = compressor.compress(a)
    res = compressor.decompress(c_a)
    assert (res == a).all()


def test_qfp():
    compressor = QuantizedFP()
    compressed_test(compressor)
    abs_max_equal_test(compressor)
    all_zeros_test(compressor)


def test_qkm():
    compressor = QuantizedKmeans()
    compressed_test(compressor)
    all_zeros_test(compressor)
