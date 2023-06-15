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

import enum
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from secretflow.data.vertical import read_csv
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.device.spu import SPU, SPUObject
from secretflow.device.driver import DeviceObject, wait
from secretflow.protos.component.comp_pb2 import Attribute, AttrType, IoDef
from secretflow.protos.component.data_pb2 import (
    DeviceObjectCollection,
    DistData,
    IndividualTable,
    SystemInfo,
    VerticalTable,
)
from secretflow.protos.component.report_pb2 import Descriptions, Div, Report, Tab, Table


class MetaEnum(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(enum.Enum, metaclass=MetaEnum):
    def __repr__(self):
        return self.value

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return str(self) == str(other)


@enum.unique
class DistDataType(BaseEnum):
    VERTICAL_TABLE = "sf.table.vertical_table"
    INDIVIDUAL_TABLE = "sf.table.individual"
    SS_SGD_MODEL = "sf.model.ss_sgd"
    SGB_MODEL = "sf.model.sgb"
    WOE_RUNNING_RULE = "sf.rule.woe_binning"
    SS_XGB_MODEL = "sf.model.ss_xgb"
    REPORT = "sf.report"


@enum.unique
class DataSetFormatSupported(BaseEnum):
    CSV = "csv"


SUPPORTED_VTABLE_DATA_TYPE = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "str": object,
}


def check_io_def(io_def: IoDef):
    for t in io_def.types:
        if t not in DistDataType:
            raise ValueError(
                f"IoDef {io_def.name}: {t} is not a supported DistData types"
            )


def check_dist_data(data: DistData, io_def: IoDef = None):
    if io_def is not None:
        check_io_def(io_def)
        if data.type not in list(io_def.types):
            raise ValueError(
                f"DistData {data.name}: type {data.type} is not allowed according to io def {io_def.types}."
            )

    if data.type == DistDataType.INDIVIDUAL_TABLE:
        if len(data.data_refs) > 1:
            raise f"DistData {data.name}: data_refs is greater than 1 for {data.type}"


@dataclass
class DistdataInfo:
    uri: str
    format: str


def extract_distdata_info(
    db: DistData,
) -> Dict[str, DistdataInfo]:
    ret = {}

    for dr in db.data_refs:
        ret[dr.party] = DistdataInfo(dr.uri, dr.format)

    return ret


def merge_individuals_to_vtable(srcs: List[DistData], dest: DistData) -> DistData:
    # copy srcs' schema into dist
    # use for union individual tables into vtable
    vmeta = VerticalTable()
    for s in srcs:
        assert s.type == DistDataType.INDIVIDUAL_TABLE
        imeta = IndividualTable()
        assert s.meta.Unpack(imeta)
        vmeta.schemas.append(imeta.schema)
        vmeta.num_lines = imeta.num_lines

    dest.meta.Pack(vmeta)

    return dest


def extract_table_header(
    db: DistData,
    load_features: bool = False,
    load_labels: bool = False,
    load_ids: bool = False,
    feature_selects: List[str] = None,
    col_selects: List[str] = None,
) -> Dict[str, Dict[str, np.dtype]]:
    """
    Args:
        db (DistData): input DistData.
        load_features (bool, optional): Whether to load feature cols. Defaults to False.
        load_labels (bool, optional): Whether to load label cols. Defaults to False.
        load_ids (bool, optional): Whether to load id cols. Defaults to False.
        feature_selects (List[str], optional): Load part of feature cols. Only in effect if load_features is True. Defaults to None.
        col_selects (List[str], optional): Load part of cols. Applies to all cols. Defaults to None.
    """
    meta = (
        IndividualTable()
        if db.type.lower() == DistDataType.INDIVIDUAL_TABLE
        else VerticalTable()
    )
    db.meta.Unpack(meta)
    schemas = (
        [meta.schema]
        if db.type.lower() == DistDataType.INDIVIDUAL_TABLE
        else meta.schemas
    )

    if feature_selects is not None:
        feature_selects = set(feature_selects)

    if col_selects is not None:
        col_selects = set(col_selects)

    ret = dict()
    for slice, dr in zip(schemas, db.data_refs):
        smeta = dict()
        if load_features:
            for t, h in zip(slice.types, slice.features):
                if feature_selects is not None:
                    if h not in feature_selects:
                        # feature not selected, skip
                        continue
                    feature_selects.remove(h)

                if col_selects is not None:
                    if h not in col_selects:
                        # feature not selected, skip
                        continue
                    col_selects.remove(h)

                t = t.lower()
                assert t in SUPPORTED_VTABLE_DATA_TYPE
                smeta[h] = SUPPORTED_VTABLE_DATA_TYPE[t]
        if load_labels:
            for label in slice.labels:
                if col_selects is not None:
                    if label not in col_selects:
                        # label not selected, skip
                        continue
                    col_selects.remove(label)
                smeta[label] = SUPPORTED_VTABLE_DATA_TYPE["f32"]
        if load_ids:
            for id in slice.ids:
                if col_selects is not None:
                    if id not in col_selects:
                        # id not selected, skip
                        continue
                    col_selects.remove(id)
                smeta[id] = SUPPORTED_VTABLE_DATA_TYPE["str"]

        if len(smeta):
            ret[dr.party] = smeta

    if feature_selects is not None and len(feature_selects) > 0:
        raise AttributeError(f"unknown features {feature_selects} in feature_selects")

    if col_selects is not None and len(col_selects) > 0:
        raise AttributeError(f"unknown cols {col_selects} in col_selects")

    return ret


def load_table(
    ctx,
    db: DistData,
    load_features: bool = False,
    load_labels: bool = False,
    load_ids: bool = False,
    feature_selects: List[str] = None,  # if None, load all features
    col_selects: List[str] = None,  # if None, load all cols
) -> VDataFrame:
    assert load_features or load_labels or load_ids, "At least one flag should be true"
    assert (
        db.type.lower() == DistDataType.INDIVIDUAL_TABLE
        or db.type.lower() == DistDataType.VERTICAL_TABLE
    ), f"path format {db.type.lower()} should be sf.table.individual or sf.table.vertical_table"

    v_headers = extract_table_header(
        db,
        load_features=load_features,
        load_labels=load_labels,
        load_ids=load_ids,
        feature_selects=feature_selects,
        col_selects=col_selects,
    )
    parties_path_format = extract_distdata_info(db)
    for p in v_headers:
        assert (
            p in parties_path_format
        ), f"schema party {p} is not in dataref parties {v_headers.keys()}"
        # only support csv for now, skip type distribute
        assert (
            parties_path_format[p].format.lower() in DataSetFormatSupported
        ), f"Illegal path format: {parties_path_format[p].format.lower()}, path format of party {p} should be in DataSetFormatSupported"
    # TODO: assert sys_info

    with ctx.tracer.trace_io():
        pyus = {p: PYU(p) for p in v_headers}
        filepaths = {
            pyus[p]: os.path.join(ctx.local_fs_wd, parties_path_format[p].uri)
            for p in v_headers
        }
        dtypes = {pyus[p]: v_headers[p] for p in v_headers}
        vdf = read_csv(filepaths, dtypes=dtypes)
        wait(vdf)

    return vdf


def dump_vertical_table(
    ctx,
    v_data: VDataFrame,
    uri: str,
    meta: VerticalTable,
    sys_info: SystemInfo,
) -> DistData:
    assert isinstance(v_data, VDataFrame)
    with ctx.tracer.trace_io():
        output_uri = {p: f"{uri}_{p.party}" for p in v_data.partitions}
        output_path = {
            p: os.path.join(ctx.local_fs_wd, output_uri[p]) for p in output_uri
        }
        wait(v_data.to_csv(output_path, index=False))

    ret = DistData(
        name=uri,
        type=str(DistDataType.VERTICAL_TABLE),
        sys_info=sys_info,
        data_refs=[
            DistData.DataRef(uri=output_uri[p], party=p.party, format="csv")
            for p in output_uri
        ],
    )
    ret.meta.Pack(meta)

    return ret


def model_dumps(
    model_name: str,
    model_type: str,
    major_version: int,
    minor_version: int,
    objs: List[DeviceObject],
    public_info: str,
    storage_root: str,
    dist_data_uri: str,
    sys_info: SystemInfo,
) -> DistData:
    objs_uri = []
    objs_party = []
    saved_objs = []
    for i, obj in enumerate(objs):
        if isinstance(obj, PYUObject):
            device: PYU = obj.device
            uri = f"{dist_data_uri}_{i}"
            path = os.path.join(storage_root, uri)

            def dumps(path: str, obj: Any):
                import pickle

                with open(path, "wb") as f:
                    f.write(pickle.dumps(obj))

            wait(device(dumps)(path, obj))

            saved_obj = DeviceObjectCollection.DeviceObject(
                type="pyu", data_ref_idxs=[len(objs_uri)]
            )
            saved_objs.append(saved_obj)
            objs_uri.append(uri)
            objs_party.append(device.party)
        elif isinstance(obj, SPUObject):
            device: SPU = obj.device
            uris = [f"{dist_data_uri}_{i}_{party}" for party in device.actors.keys()]
            spu_paths = [os.path.join(storage_root, uri) for uri in uris]

            wait(device.dump(obj, spu_paths))

            saved_obj = DeviceObjectCollection.DeviceObject(
                type="spu", data_ref_idxs=[len(objs_uri) + p for p in range(len(uris))]
            )
            saved_objs.append(saved_obj)
            objs_uri.extend(uris)
            objs_party.extend(list(device.actors.keys()))
        else:
            raise RuntimeError(f"not supported objs type {type(obj)}")

    model_info = {
        "major_version": major_version,
        "minor_version": minor_version,
        "public_info": public_info,
    }

    model_meta = DeviceObjectCollection(
        objs=saved_objs,
        public_info=json.dumps(model_info),
    )

    dist_data = DistData(
        name=model_name,
        type=str(model_type),
        sys_info=sys_info,
        data_refs=[
            DistData.DataRef(uri=uri, party=p, format="pickle")
            for uri, p in zip(objs_uri, objs_party)
        ],
    )
    dist_data.meta.Pack(model_meta)

    return dist_data


def model_loads(
    dist_data: DistData,
    max_major_version: int,
    max_minor_version: int,
    model_type: str,
    storage_root: str,
    pyus: Dict[str, PYU] = None,
    spu: SPU = None,
    # TODO: assert sys_info
    # sys_info: SystemInfo = None,
) -> Tuple[List[DeviceObject], str]:
    assert dist_data.type == model_type
    model_meta = DeviceObjectCollection()
    assert dist_data.meta.Unpack(model_meta)

    model_info = json.loads(model_meta.public_info)

    assert (
        isinstance(model_info, dict)
        and "major_version" in model_info
        and "minor_version" in model_info
        and "public_info" in model_info
    )

    assert (
        max_major_version >= model_info["major_version"]
        and max_minor_version >= model_info["minor_version"]
    ), "not support model version"

    objs = []
    for save_obj in model_meta.objs:
        if save_obj.type == "pyu":
            assert pyus is not None
            assert len(save_obj.data_ref_idxs) == 1
            data_ref = dist_data.data_refs[save_obj.data_ref_idxs[0]]
            party = data_ref.party
            assert party in pyus
            assert data_ref.format == "pickle"

            def loads(path: str) -> Any:
                import pickle

                with open(path, "rb") as f:
                    # TODO: not secure, may change to json loads/dumps?
                    return pickle.loads(f.read())

            objs.append(pyus[party](loads)(os.path.join(storage_root, data_ref.uri)))
        elif save_obj.type == "spu":
            # TODO: only support one spu for now
            assert spu is not None
            assert len(save_obj.data_ref_idxs) > 1
            full_paths = {}
            for data_ref_idx in save_obj.data_ref_idxs:
                data_ref = dist_data.data_refs[data_ref_idx]
                assert data_ref.format == "pickle"
                party = data_ref.party
                assert party not in full_paths
                full_paths[party] = os.path.join(storage_root, data_ref.uri)
            assert set(full_paths.keys()) == set(spu.actors.keys())
            spu_paths = [full_paths[party] for party in spu.actors.keys()]
            objs.append(spu.load(spu_paths))
        else:
            raise RuntimeError(f"not supported objs type {save_obj.type}")

    return objs, model_info["public_info"]


def gen_table_statistic_report(df: pd.DataFrame) -> Report:
    headers, rows = [], []
    headers.append(
        Table.HeaderItem(name="table_column_name", desc="", type=AttrType.AT_STRING)
    )
    rows.append(
        Table.Row(
            name="table_column_name",
            desc="",
            items=[Attribute(s=i) for i in df.index],
        ),
    )
    for col in df.columns:
        headers.append(Table.HeaderItem(name=col, desc="", type=AttrType.AT_STRING))
        rows.append(
            Table.Row(name=col, desc="", items=[Attribute(s=str(i)) for i in df[col]])
        )

    r_table = Table(headers=headers, rows=rows)

    return Report(
        name="table statistics",
        desc="",
        tabs=[
            Tab(
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="table",
                                table=r_table,
                            )
                        ],
                    )
                ],
            )
        ],
    )


def dump_table_statistics(name, sys_info, df: pd.DataFrame) -> DistData:
    report_mate = gen_table_statistic_report(df)
    res = DistData(
        name=name,
        sys_info=sys_info,
        type=str(DistDataType.REPORT),
        data_refs=[],
    )
    res.meta.Pack(report_mate)
    return res


def dump_pva_eval_result(name, sys_info, value: float) -> DistData:
    r_desc = Descriptions(
        items=[
            Descriptions.Item(
                name="pva", type=AttrType.AT_FLOAT, value=Attribute(f=value)
            )
        ]
    )

    report_mate = Report(
        name="report",
        desc="pva",
        tabs=[
            Tab(
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="descriptions",
                                descriptions=r_desc,
                            )
                        ],
                    )
                ],
            )
        ],
    )
    report_dd = DistData(
        name=name,
        type=str(DistDataType.REPORT),
        sys_info=sys_info,
    )
    report_dd.meta.Pack(report_mate)
    return report_dd
