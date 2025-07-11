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
import spu


def spu_fxp_precision(field_type):
    """Fixed point integer default precision bits"""
    field_precision_dict = {
        spu.FieldType.FM32.name: 8,
        spu.FieldType.FM64.name: 18,
        spu.FieldType.FM128.name: 26,
    }
    if isinstance(field_type, spu.FieldType):
        field_type = field_type.name

    if field_type in field_precision_dict:
        return field_precision_dict[field_type]

    raise ValueError(f'unsupported field type {field_type}')


def spu_fxp_size(field_type):
    """Fixed point integer size in bytes"""
    field_fxp_size_dict = {
        spu.FieldType.FM32.name: 4,
        spu.FieldType.FM64.name: 8,
        spu.FieldType.FM128.name: 16,
    }
    if isinstance(field_type, spu.FieldType):
        field_type = field_type.name

    if field_type in field_fxp_size_dict:
        return field_fxp_size_dict[field_type]

    raise ValueError(f'unsupported field type {field_type}')


HEU_SPU_DT_SWITCHER = {
    "DT_I1": spu.DataType.DT_I1,
    "DT_I8": spu.DataType.DT_I8,
    "DT_U8": spu.DataType.DT_U8,
    "DT_I16": spu.DataType.DT_I16,
    "DT_U16": spu.DataType.DT_U16,
    "DT_I32": spu.DataType.DT_I32,
    "DT_U32": spu.DataType.DT_U32,
    "DT_I64": spu.DataType.DT_I64,
    "DT_U64": spu.DataType.DT_U64,
    "DT_F32": spu.DataType.DT_F32,
    "DT_F64": spu.DataType.DT_F64,
}


def heu_datatype_to_spu(heu_dt):
    assert heu_dt in HEU_SPU_DT_SWITCHER, f"Unsupported heu datatype {heu_dt}"
    return HEU_SPU_DT_SWITCHER.get(heu_dt)


SPU_HEU_DT_SWITCHER = {
    spu.DataType.DT_I1: "DT_I1",
    spu.DataType.DT_I8: "DT_I8",
    spu.DataType.DT_U8: "DT_U8",
    spu.DataType.DT_I16: "DT_I16",
    spu.DataType.DT_U16: "DT_U16",
    spu.DataType.DT_I32: "DT_I32",
    spu.DataType.DT_U32: "DT_U32",
    spu.DataType.DT_I64: "DT_I64",
    spu.DataType.DT_U64: "DT_U64",
    spu.DataType.DT_F32: "DT_F32",
    spu.DataType.DT_F64: "DT_F64",
}


def spu_datatype_to_heu(spu_dt):
    assert spu_dt in SPU_HEU_DT_SWITCHER, f"Unsupported spu datatype {spu_dt}"
    return SPU_HEU_DT_SWITCHER.get(spu_dt)


HEU_NP_DT_SWITCHER = {
    "DT_I1": np.bool_,
    "DT_I8": np.int8,
    "DT_U8": np.uint8,
    "DT_I16": np.int16,
    "DT_U16": np.uint16,
    "DT_I32": np.int32,
    "DT_U32": np.uint32,
    "DT_I64": np.int64,
    "DT_U64": np.uint64,
    "DT_F32": np.float32,
    "DT_F64": np.float64,
}


def heu_datatype_to_numpy(heu_dt) -> np.dtype:
    assert heu_dt in HEU_NP_DT_SWITCHER, f"Unsupported heu datatype {heu_dt}"
    return HEU_NP_DT_SWITCHER.get(heu_dt)
