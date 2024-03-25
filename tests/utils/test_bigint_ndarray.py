# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from heu import phe

from secretflow.utils import ndarray_bigint


def test_arange():
    a = ndarray_bigint.arange(120)
    a.resize((2, 3, 4, 5))
    assert (np.arange(120).reshape((2, 3, 4, 5)) == a.to_numpy()).all()


def test_randbits():
    a = ndarray_bigint.randbits((100000,), 8)
    res = {i: 0 for i in range(-128, 128)}
    for i in a.data:
        assert i >= -128
        assert i <= 127
        res[i] += 1
    for i, c in res.items():
        assert c > 0, f"cannot generate randint {i}"


def test_randint():
    bound = 2**2048
    a = ndarray_bigint.randint((100,), -bound, bound)
    uint128_max = 2**128 - 1
    # a is much bigger than uint128
    assert (a.to_numpy() > uint128_max).any()
    assert (a.to_numpy() < -uint128_max).any()


def test_add():
    arrays = [ndarray_bigint.randbits((3, 4), 16) for _ in range(10)]
    array_sum = sum(arrays, ndarray_bigint.zeros((3, 4)))

    np_array_sum = sum([a.to_numpy() for a in arrays])
    assert (array_sum.to_numpy() == np_array_sum).all()


def test_to_bytes():
    array = ndarray_bigint.arange(300)
    b1 = array.to_bytes(1)
    assert len(b1) == 300
    b2 = array.to_bytes(2)
    assert len(b2) == 600


def test_to_numpy():
    # int128 case
    array = ndarray_bigint.randbits((500, 300), 64).to_numpy()
    assert array.dtype == np.int64
    array = ndarray_bigint.randbits((500, 300), 65).to_numpy()
    assert array.dtype == object
    array = ndarray_bigint.randbits((500, 300), 128).to_numpy()
    assert array.dtype == object
    assert isinstance(array[0][0], int)

    # big int case
    array = ndarray_bigint.randbits((2, 3), 512)
    array_np = array.to_numpy()
    array_pt = np.vectorize(lambda x: phe.Plaintext(phe.SchemaType.ZPaillier, x))(
        array_np
    )
    assert isinstance(array_pt[0][0], phe.Plaintext)
    assert array.data[0] == int(array_pt[0][0])
