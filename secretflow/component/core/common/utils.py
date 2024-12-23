# Copyright 2024 Ant Group Co., Ltd.
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


import jax
import jaxlib
import numpy as np
import pyarrow as pa

from secretflow.spec.v1.component_pb2 import Attribute


def pa_dtype_to_type(dt: pa.DataType) -> type:
    if pa.types.is_integer(dt):
        return int
    elif pa.types.is_floating(dt):
        return float
    elif pa.types.is_boolean(dt):
        return bool
    elif pa.types.is_string(dt):
        return str
    else:
        raise ValueError(f"unsupport type {dt}")


def np_dtype_to_type(dt: np.dtype) -> type:
    if np.issubdtype(dt, np.bool_):
        return bool
    elif np.issubdtype(dt, np.integer):
        return int
    elif np.issubdtype(dt, np.floating):
        return float
    elif np.issubdtype(dt, np.str_) or np.issubdtype(dt, np.object_):
        return str
    else:
        raise ValueError(f"unsupport type {dt}")


def to_type(dt) -> type:
    if isinstance(dt, pa.DataType):
        return pa_dtype_to_type(dt)
    elif isinstance(dt, np.dtype):
        return np_dtype_to_type(dt)
    elif isinstance(dt, np.generic):
        return np_dtype_to_type(dt.dtype)
    elif isinstance(dt, jaxlib.xla_extension.ArrayImpl):
        ja = jax.device_get(dt)
        if ja.ndim == 0:
            return type(ja.item())
        else:
            raise TypeError(f"unsupported type {type(dt)} {dt}")
    elif isinstance(dt, (bool, int, float, str)):
        return type(dt)
    elif isinstance(dt, type) and dt in [bool, int, float, str]:
        return dt
    else:
        raise TypeError(f"unsupported type {type(dt)} {dt}")


def to_type_str(dt) -> str:
    return to_type(dt).__name__


def to_attribute(v) -> Attribute:  # type: ignore
    if isinstance(v, Attribute):
        return v
    elif isinstance(v, pa.Scalar):
        v = v.as_py()
    elif isinstance(v, np.generic):
        v = v.item()
    elif isinstance(v, jaxlib.xla_extension.ArrayImpl):
        ja = jax.device_get(v)
        if ja.ndim == 0:
            v = ja.item()
        else:
            v = ja.tolist()

    is_list = isinstance(v, list)
    if is_list:
        assert len(v) > 0
        prim_type = type(v[0])
    else:
        prim_type = type(v)

    if prim_type == bool:
        return Attribute(bs=v) if is_list else Attribute(b=v)
    elif prim_type == int:
        return Attribute(i64s=v) if is_list else Attribute(i64=v)
    elif prim_type == float:
        return Attribute(fs=v) if is_list else Attribute(f=v)
    elif prim_type == str:
        return Attribute(ss=v) if is_list else Attribute(s=v)
    else:
        raise ValueError(f"unsupported primitive type {prim_type}")


TRUE_STRINGS = ["true", "yes", "y", "enable", "enabled", "1"]
FALSE_STRINGS = ["false", "no", "n", "disable", "disabled", "0"]


def to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in TRUE_STRINGS:
        return True
    if value.lower() in FALSE_STRINGS:
        return False
    raise ValueError(f"Unknown boolean value '{value}'")
