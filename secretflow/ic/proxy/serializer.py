# Copyright 2023 Ant Group Co., Ltd.
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

from typing import Union, Sequence, List, Any
import numpy as np
import jax
import heu.numpy as hnp
from heu.phe import PublicKey
from secretflow.ic.proto.runtime import data_exchange_pb2 as de


def serialize(data: Any) -> bytes:
    if isinstance(data, (bool, np.bool_)):
        return _serialize_bool(data)
    elif isinstance(data, int):
        return _serialize_int(data)
    elif isinstance(data, (np.ndarray, jax.numpy.ndarray)):
        return _serialize_ndarray(data)
    elif isinstance(data, (list, tuple)):
        return _serialize_list(data)
    elif isinstance(data, PublicKey):
        return _serialize_public_key(data)
    elif isinstance(data, hnp.CiphertextArray):
        return data.serialize(format=hnp.MatrixSerializeFormat.Interconnection)
    else:
        raise NotImplementedError(f'serialize not implemented for type: {type(data)}')


def deserialize(msg_bytes: bytes) -> Any:
    data_pb = de.DataExchangeProtocol()
    data_pb.ParseFromString(msg_bytes)
    if data_pb.scalar_type == de.SCALAR_TYPE_OBJECT:
        if data_pb.scalar_type_name == "public_key":
            return _public_key_from_bytes(data_pb.scalar.buf)
        else:
            return _deserialize_hnp_ndarray(msg_bytes)
    elif data_pb.HasField('scalar'):
        return _scalar_from_bytes(data_pb.scalar, data_pb.scalar_type)
    elif data_pb.HasField('f_ndarray'):
        return _ndarray_from_bytes(data_pb.f_ndarray, data_pb.scalar_type)
    elif data_pb.HasField('f_scalar_list'):
        return _scalar_list_from_bytes(data_pb.f_scalar_list, data_pb.scalar_type)
    elif data_pb.HasField('f_ndarray_list'):
        return _ndarray_list_from_bytes(data_pb.f_ndarray_list, data_pb.scalar_type)


def _serialize_public_key(data: PublicKey) -> bytes:
    data_pb = de.DataExchangeProtocol()
    data_pb.scalar_type = de.SCALAR_TYPE_OBJECT
    data_pb.scalar_type_name = "public_key"
    data_pb.scalar.buf = data.serialize()

    return data_pb.SerializeToString()


def _public_key_from_bytes(buf: bytes) -> PublicKey:
    return PublicKey.load_from(buf)


def _deserialize_hnp_ndarray(msg_bytes: bytes) -> hnp.CiphertextArray:
    return hnp.CiphertextArray.load_from(
        msg_bytes, format=hnp.MatrixSerializeFormat.Interconnection
    )


def _serialize_bool(data: Union[bool, np.bool_]) -> bytes:
    data_pb = de.DataExchangeProtocol()
    data_pb.scalar_type = de.SCALAR_TYPE_BOOL
    data_pb.scalar.buf = _bool_to_bytes(data)

    return data_pb.SerializeToString()


def _bool_to_bytes(data: Union[bool, np.bool_]) -> bytes:
    if isinstance(data, bool):
        return data.to_bytes(length=1, byteorder='little')
    elif isinstance(data, np.bool_):
        return data.tobytes()


def _bool_from_bytes(msg_bytes: bytes) -> bool:
    return bool.from_bytes(msg_bytes, byteorder='little')


def _serialize_int(data: int) -> bytes:
    data_pb = de.DataExchangeProtocol()
    signed = data < 0
    byte_size = _get_int_size(data, signed)
    data_pb.scalar_type = _get_scalar_type_from_size(signed, byte_size=byte_size)
    data_pb.scalar.buf = data.to_bytes(
        length=byte_size, byteorder='little', signed=signed
    )

    return data_pb.SerializeToString()


def _int_from_bytes(msg_bytes: bytes, int_type: de.ScalarType) -> int:
    signed = _get_scalar_type_signed(int_type)

    return int.from_bytes(msg_bytes, byteorder='little', signed=signed)


def _scalar_from_bytes(
    scalar: de.Scalar, scalar_type: de.ScalarType
) -> Union[bool, int]:
    if scalar_type == de.SCALAR_TYPE_BOOL:
        return _bool_from_bytes(scalar.buf)
    else:
        return _int_from_bytes(scalar.buf, scalar_type)


def _serialize_ndarray(data: Union[np.ndarray, jax.numpy.ndarray]) -> bytes:
    data_pb = de.DataExchangeProtocol()
    data_pb.scalar_type = _numpy_type_to_scalar_type(data.dtype)
    data_pb.f_ndarray.shape.extend(data.shape)
    data_pb.f_ndarray.item_buf = data.tobytes()
    return data_pb.SerializeToString()


def _ndarray_from_bytes(ndarray: de.FNdArray, scalar_type: de.ScalarType) -> np.ndarray:
    item_type = _scalar_type_to_numpy_type(scalar_type)
    data = np.frombuffer(ndarray.item_buf, dtype=item_type)
    data = data.reshape(list(ndarray.shape))
    return data


def _serialize_list(data: Sequence) -> bytes:
    if len(data) == 0:
        return _serialize_empty_list()
    else:
        if isinstance(data[0], (bool, np.bool_)):
            return _serialize_bool_list(data)
        elif isinstance(data[0], int):
            return _serialize_int_list(data)
        elif isinstance(data[0], (np.ndarray, jax.numpy.ndarray)):
            return _serialize_ndarray_list(data)
        else:
            raise NotImplementedError(f'type {type(data[0])} is not supported')


def _serialize_empty_list() -> bytes:
    data_pb = de.DataExchangeProtocol()
    data_pb.scalar_type = de.SCALAR_TYPE_INT8  # any valid type
    data_pb.f_scalar_list.item_count = 0
    return data_pb.SerializeToString()


def _serialize_bool_list(data: Union[Sequence[bool], Sequence[np.bool_]]) -> bytes:
    data_pb = de.DataExchangeProtocol()
    data_pb.scalar_type = de.SCALAR_TYPE_BOOL
    data_pb.f_scalar_list.item_count = len(data)
    for item in data:
        data_pb.f_scalar_list.item_buf += _bool_to_bytes(item)
    return data_pb.SerializeToString()


def _serialize_int_list(data: Sequence[int]) -> bytes:
    data_pb = de.DataExchangeProtocol()
    # signed = any(item < 0 for item in data)
    # item_byte_size = max(_get_int_size(item, signed) for item in data)
    signed = True
    item_byte_size = max(
        _get_int_size(min(data), signed),
        _get_int_size(max(data), signed),
    )
    data_pb.scalar_type = _get_scalar_type_from_size(signed, item_byte_size)
    data_pb.f_scalar_list.item_count = len(data)
    for item in data:
        data_pb.f_scalar_list.item_buf += item.to_bytes(
            length=item_byte_size, byteorder='little', signed=signed
        )
    return data_pb.SerializeToString()


def _serialize_ndarray_list(data: Sequence[np.ndarray]) -> bytes:
    data_pb = de.DataExchangeProtocol()

    assert all(ndarray.dtype == data[0].dtype for ndarray in data)
    data_pb.scalar_type = _numpy_type_to_scalar_type(data[0].dtype)

    for ndarray in data:
        ndarray_pb = de.FNdArray()
        ndarray_pb.shape.extend(ndarray.shape)
        ndarray_pb.item_buf = ndarray.tobytes()
        data_pb.f_ndarray_list.ndarrays.append(ndarray_pb)

    return data_pb.SerializeToString()


def _scalar_list_from_bytes(
    scalar_list: de.FScalarList, scalar_type: de.ScalarType
) -> Union[List[bool], List[int]]:
    if scalar_type == de.SCALAR_TYPE_BOOL:
        return _bool_list_from_bytes(scalar_list)
    else:
        return _int_list_from_bytes(scalar_list, scalar_type)


def _bool_list_from_bytes(scalar_list: de.FScalarList) -> List[bool]:
    return [
        _bool_from_bytes(scalar_list.item_buf[i : i + 1])
        for i in range(len(scalar_list.item_buf))
    ]


def _int_list_from_bytes(
    scalar_list: de.FScalarList, scalar_type: de.ScalarType
) -> List[int]:
    item_byte_size = _get_scalar_type_size(scalar_type)
    signed = _get_scalar_type_signed(scalar_type)
    data = list()
    begin = 0
    end = item_byte_size
    for i in range(scalar_list.item_count):
        data.append(
            int.from_bytes(
                scalar_list.item_buf[begin:end], byteorder='little', signed=signed
            )
        )
        begin = end
        end += item_byte_size

    return data


def _ndarray_list_from_bytes(
    ndarray_list: de.FNdArrayList, scalar_type: de.ScalarType
) -> List[np.ndarray]:
    return [
        _ndarray_from_bytes(ndarray, scalar_type) for ndarray in ndarray_list.ndarrays
    ]


def _get_int_size(data: int, signed: bool) -> int:
    if signed:
        return (8 + (data + (data < 0)).bit_length()) // 8
    else:
        return (data.bit_length() + 7) // 8


_SCALAR_TYPE_TO_SIZE_DICT = {
    de.SCALAR_TYPE_INT8: 1,
    de.SCALAR_TYPE_UINT8: 1,
    de.SCALAR_TYPE_INT16: 2,
    de.SCALAR_TYPE_UINT16: 2,
    de.SCALAR_TYPE_INT32: 4,
    de.SCALAR_TYPE_UINT32: 4,
    de.SCALAR_TYPE_INT64: 8,
    de.SCALAR_TYPE_UINT64: 8,
}


def _get_scalar_type_size(scalar_type: de.ScalarType) -> int:
    assert (
        scalar_type in _SCALAR_TYPE_TO_SIZE_DICT
    ), f'_get_scalar_type_size not implemented for scalar type {scalar_type}'

    return _SCALAR_TYPE_TO_SIZE_DICT[scalar_type]


_SCALAR_TYPE_TO_SIGNED_DICT = {
    de.SCALAR_TYPE_INT8: True,
    de.SCALAR_TYPE_UINT8: False,
    de.SCALAR_TYPE_INT16: True,
    de.SCALAR_TYPE_UINT16: False,
    de.SCALAR_TYPE_INT32: True,
    de.SCALAR_TYPE_UINT32: False,
    de.SCALAR_TYPE_INT64: True,
    de.SCALAR_TYPE_UINT64: False,
}


def _get_scalar_type_signed(scalar_type: de.ScalarType) -> bool:
    assert (
        scalar_type in _SCALAR_TYPE_TO_SIGNED_DICT
    ), f'{scalar_type} not in SCALAR_TYPE_TO_SIGNED_DICT'

    return _SCALAR_TYPE_TO_SIGNED_DICT[scalar_type]


_SIZE_TO_SIGNED_SCALAR_TYPE_DICT = {
    1: de.SCALAR_TYPE_INT8,
    2: de.SCALAR_TYPE_INT16,
    4: de.SCALAR_TYPE_INT32,
    8: de.SCALAR_TYPE_INT64,
}

_SIZE_TO_UNSIGNED_SCALAR_TYPE_DICT = {
    0: de.SCALAR_TYPE_UINT8,  # size of integer zero may be 0
    1: de.SCALAR_TYPE_UINT8,
    2: de.SCALAR_TYPE_UINT16,
    4: de.SCALAR_TYPE_UINT32,
    8: de.SCALAR_TYPE_UINT64,
}


def _get_scalar_type_from_size(signed: bool, byte_size: int) -> de.ScalarType:
    if signed:
        assert (
            byte_size in _SIZE_TO_SIGNED_SCALAR_TYPE_DICT
        ), f'{byte_size} not in SIZE_TO_SIGNED_SCALAR_TYPE_DICT'

        return _SIZE_TO_SIGNED_SCALAR_TYPE_DICT[byte_size]
    else:
        assert (
            byte_size in _SIZE_TO_UNSIGNED_SCALAR_TYPE_DICT
        ), f'{byte_size} not in SIZE_TO_UNSIGNED_SCALAR_TYPE_DICT'

        return _SIZE_TO_UNSIGNED_SCALAR_TYPE_DICT[byte_size]


_NUMPY_TYPE_TO_SCALAR_TYPE_DICT = {
    np.bool_: de.SCALAR_TYPE_BOOL,
    np.int8: de.SCALAR_TYPE_INT8,
    np.uint8: de.SCALAR_TYPE_UINT8,
    np.int16: de.SCALAR_TYPE_INT16,
    np.uint16: de.SCALAR_TYPE_UINT16,
    np.int32: de.SCALAR_TYPE_INT32,
    np.uint32: de.SCALAR_TYPE_UINT32,
    np.int64: de.SCALAR_TYPE_INT64,
    np.uint64: de.SCALAR_TYPE_UINT64,
    np.float16: de.SCALAR_TYPE_FLOAT16,
    np.float32: de.SCALAR_TYPE_FLOAT32,
    np.float64: de.SCALAR_TYPE_FLOAT64,
}


def _numpy_type_to_scalar_type(np_type: np.dtype) -> de.ScalarType:
    assert (
        np_type.type in _NUMPY_TYPE_TO_SCALAR_TYPE_DICT
    ), f'{np_type.type} not in NUMPY_TYPE_TO_SCALAR_TYPE_DICT'

    return _NUMPY_TYPE_TO_SCALAR_TYPE_DICT[np_type.type]


_SCALAR_TYPE_TO_NUMPY_TYPE_DICT = {
    de.SCALAR_TYPE_BOOL: np.bool_,
    de.SCALAR_TYPE_INT8: np.int8,
    de.SCALAR_TYPE_UINT8: np.uint8,
    de.SCALAR_TYPE_INT16: np.int16,
    de.SCALAR_TYPE_UINT16: np.uint16,
    de.SCALAR_TYPE_INT32: np.int32,
    de.SCALAR_TYPE_UINT32: np.uint32,
    de.SCALAR_TYPE_INT64: np.int64,
    de.SCALAR_TYPE_UINT64: np.uint64,
    de.SCALAR_TYPE_FLOAT16: np.float16,
    de.SCALAR_TYPE_FLOAT32: np.float32,
    de.SCALAR_TYPE_FLOAT64: np.float64,
}


def _scalar_type_to_numpy_type(scalar_type: de.ScalarType) -> np.dtype:
    assert (
        scalar_type in _SCALAR_TYPE_TO_NUMPY_TYPE_DICT
    ), f'{scalar_type} not in SCALAR_TYPE_TO_NUMPY_TYPE_DICT'

    return _SCALAR_TYPE_TO_NUMPY_TYPE_DICT[scalar_type]


if __name__ == '__main__':
    msg_bytes = serialize(-135)
    print(deserialize(msg_bytes))

    msg_bytes = serialize(True)
    print(deserialize(msg_bytes))

    arr = np.array([[1, 2], [3, -4]])
    msg_bytes = serialize(arr)
    print(deserialize(msg_bytes))

    msg_bytes = serialize([1, 2, -3, 4])
    print(deserialize(msg_bytes))

    msg_bytes = serialize([True, False, True, True])
    print(deserialize(msg_bytes))
