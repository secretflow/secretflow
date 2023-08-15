# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .errors import InvalidArgumentError
def LWE_encode(m: np.ndarray, is_float:bool,fxp_bits: int,prime:int) -> np.ndarray:
    """Encode float ndarray to uint64 finite field.
    Float will times 2**fxp_bits firstly.

    Args:
        m (np.ndarray): the ndarray to encode.
        is_float: the tpye of the ndarray is float.
        fraction_precesion (int): keep how many decimal digits after the dot.
            Must provide if ndarray dtype is float.          ##小数点后面的精度

    Returns:
        np.ndarray: the encoded FieldArray.
    """
    assert isinstance(m, np.ndarray), f'Support ndarray only but got {type(m)}'
    abs_max = (prime - 1) / 2
    if is_float:
        if m.dtype not in [np.float16, np.float32, np.float64]:
            raise InvalidArgumentError(f'Accept float ndarray only but got {m.dtype}')
        assert fxp_bits is not None, f'Fxp_bits must not be None.'
        max_value = abs(m).max()
        if abs(max_value * (1 << fxp_bits)) > abs_max:
            raise InvalidArgumentError(
                f'Float data {max_value} exceeds uint range ({-abs_max}, {abs_max}) after encoding.')
        # Convert to np.float64 for reducing overflow.% GF.order
        nparray_data = ((m.astype(np.float64) * (1 << fxp_bits)) % prime).astype(np.int64)
    else:
        max_value = abs(m).max()
        if abs(max_value) > abs_max:
            raise InvalidArgumentError(
                f'Float data {max_value} exceeds uint range ({-abs_max}, {abs_max}) after encoding.')
        nparray_data = (m % prime).astype(np.int64)
    return nparray_data

def LWE_decode(m: np.ndarray, is_float: bool,fxp_bits: int,prime:int) -> np.ndarray:
    """Decode ndarray from uint64 finite field to the float.
    Fraction precesion shall be corresponding to encoding fraction precesion.

    Args:
        m (np.ndarray): the ndarray to decode.
        fxp_bits (int): the decimal digits to keep when encoding float.
            Must provide if the original dtype is float.

    Returns:
        np.ndarray: the decoded float ndarray.
    """
    assert isinstance(m, np.ndarray), f'Support ndarray only but got {type(m)}'
    for i in range(len(m)):
        if m[i] > ((prime - 1) / 2):
            m[i] = m[i] - prime
    if is_float:
        assert fxp_bits is not None, f'Fraction precesion must not be None.'
        return m.astype(np.int64) / (1 << fxp_bits)
    else:
        return m.astype(np.int64)

def encode(m: np.ndarray, fxp_bits: int) -> np.ndarray:
    """Encode float ndarray to uint64 finite field.
    Float will times 2**fxp_bits firstly.

    Args:
        m (np.ndarray): the ndarray to encode.
        fraction_precision (int): keep how many decimal digits after the dot.
            Must provide if ndarray dtype is float.

    Returns:
        np.ndarray: the encoded ndarray.
    """
    assert isinstance(m, np.ndarray), f'Support ndarray only but got {type(m)}'
    if m.dtype not in [np.float16, np.float32, np.float64]:
        raise InvalidArgumentError(f'Accept float ndarray only but got {m.dtype}')

    uint64_max = 0xFFFFFFFFFFFFFFFF
    assert fxp_bits is not None, f'Fxp_bits must not be None.'
    max_value = m.max()
    if max_value * (1 << fxp_bits) > uint64_max:
        raise InvalidArgumentError(
            f'Float data {max_value} exceeds uint range (0, {uint64_max}) after encoding.'
        )
    # Convert to np.float64 for reducing overflow.
    return (m.astype(np.float64) * (1 << fxp_bits)).astype(np.uint64)


def decode(m: np.ndarray, fxp_bits: int) -> np.ndarray:
    """Decode ndarray from uint64 finite field to the float.
    Fraction precision shall be corresponding to encoding fraction precision.

    Args:
        m (np.ndarray): the ndarray to decode.
        fxp_bits (int): the decimal digits to keep when encoding float.
            Must provide if the original dtype is float.

    Returns:
        np.ndarray: the decoded float ndarray.
    """
    assert isinstance(m, np.ndarray), f'Support ndarray only but got {type(m)}'
    assert m.dtype == np.uint64, f'Ndarray dtype must be uint but got {m.dtype}'
    assert fxp_bits is not None, f'Fraction precision must not be None.'
    # Convert to int for restoring the negatives.
    return m.astype(np.int64) / (1 << fxp_bits)
