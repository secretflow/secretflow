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
from secretflow.spec.extend.data_pb2 import DeviceObjectCollection
from secretflow.spec.v1.component_pb2 import IoDef
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    SystemInfo,
    TableSchema,
    VerticalTable,
)


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
    SS_GLM_MODEL = "sf.model.ss_glm"
    SGB_MODEL = "sf.model.sgb"
    BIN_RUNNING_RULE = "sf.rule.binning"
    SS_XGB_MODEL = "sf.model.ss_xgb"
    ONEHOT_RULE = "sf.rule.onehot_encode"
    PREPROCESSING_RULE = "sf.rule.proprocessing"
    REPORT = "sf.report"


@enum.unique
class DataSetFormatSupported(BaseEnum):
    CSV = "csv"


SUPPORTED_VTABLE_DATA_TYPE = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bool": bool,
    "int": int,
    "float": float,
    "str": object,
}

REVERSE_DATA_TYPE_MAP = dict((v, k) for k, v in SUPPORTED_VTABLE_DATA_TYPE.items())


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
        vmeta.line_count = imeta.line_count

    dest.meta.Pack(vmeta)

    return dest


def extract_table_header(
    db: DistData,
    load_features: bool = False,
    load_labels: bool = False,
    load_ids: bool = False,
    feature_selects: List[str] = None,
    col_selects: List[str] = None,
    col_excludes: List[str] = None,
    return_schema_names: bool = False,
) -> Dict[str, Dict[str, np.dtype]]:
    """
    Args:
        db (DistData): input DistData.
        load_features (bool, optional): Whether to load feature cols. Defaults to False.
        load_labels (bool, optional): Whether to load label cols. Defaults to False.
        load_ids (bool, optional): Whether to load id cols. Defaults to False.
        feature_selects (List[str], optional): Load part of feature cols. Only in effect if load_features is True. Defaults to None.
        col_selects (List[str], optional): Load part of cols. Applies to all cols. Defaults to None. Couldn't use with col_excludes.
        col_excludes (List[str], optional): Load all cols exclude these. Applies to all cols. Defaults to None. Couldn't use with col_selects.
        return_schema_names (bool, optional): if True, also return schema names Dict[str, List[str]]
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

    if col_selects is not None and col_excludes is not None:
        raise AttributeError("col_selects and col_excludes couldn't use together.")

    if col_selects is not None:
        col_selects = set(col_selects)

    if col_excludes is not None:
        col_excludes = set(col_excludes)

    ret = dict()
    schema_names = {}
    labels = {}
    features = {}
    ids = {}
    for slice, dr in zip(schemas, db.data_refs):
        smeta = dict()
        party_labels = []
        party_features = []
        party_ids = []
        if load_features:
            for t, h in zip(slice.feature_types, slice.features):
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

                if col_excludes is not None:
                    if h in col_excludes:
                        continue

                t = t.lower()
                assert (
                    t in SUPPORTED_VTABLE_DATA_TYPE
                ), f"The feature type {t} is not supported"
                if return_schema_names:
                    party_features.append(h)
                smeta[h] = SUPPORTED_VTABLE_DATA_TYPE[t]
        if load_labels:
            for t, h in zip(slice.label_types, slice.labels):
                if col_selects is not None:
                    if h not in col_selects:
                        # label not selected, skip
                        continue
                    col_selects.remove(h)

                if col_excludes is not None:
                    if h in col_excludes:
                        continue
                if return_schema_names:
                    party_labels.append(h)
                smeta[h] = SUPPORTED_VTABLE_DATA_TYPE[t]
        if load_ids:
            for t, h in zip(slice.id_types, slice.ids):
                if col_selects is not None:
                    if h not in col_selects:
                        # id not selected, skip
                        continue
                    col_selects.remove(h)

                if col_excludes is not None:
                    if h in col_excludes:
                        continue

                if return_schema_names:
                    party_ids.append(h)

                smeta[h] = SUPPORTED_VTABLE_DATA_TYPE[t]

        if len(smeta):
            ret[dr.party] = smeta
            labels[dr.party] = party_labels
            features[dr.party] = party_features
            ids[dr.party] = party_ids

    schema_names["labels"] = labels
    schema_names["features"] = features
    schema_names["ids"] = ids

    if feature_selects is not None and len(feature_selects) > 0:
        raise AttributeError(f"unknown features {feature_selects} in feature_selects")

    if col_selects is not None and len(col_selects) > 0:
        raise AttributeError(f"unknown cols {col_selects} in col_selects")
    if return_schema_names:
        return ret, schema_names
    return ret


def load_table(
    ctx,
    db: DistData,
    load_features: bool = False,
    load_labels: bool = False,
    load_ids: bool = False,
    feature_selects: List[str] = None,  # if None, load all features
    col_selects: List[str] = None,  # if None, load all cols
    col_excludes: List[str] = None,
    return_schema_names: bool = False,
    nrows: int = None,
) -> VDataFrame:
    assert load_features or load_labels or load_ids, "At least one flag should be true"
    assert (
        db.type.lower() == DistDataType.INDIVIDUAL_TABLE
        or db.type.lower() == DistDataType.VERTICAL_TABLE
    ), f"path format {db.type.lower()} should be sf.table.individual or sf.table.vertical_table"
    if return_schema_names:
        v_headers, schema_names = extract_table_header(
            db,
            load_features=load_features,
            load_labels=load_labels,
            load_ids=load_ids,
            feature_selects=feature_selects,
            col_selects=col_selects,
            col_excludes=col_excludes,
            return_schema_names=True,
        )
    else:
        v_headers = extract_table_header(
            db,
            load_features=load_features,
            load_labels=load_labels,
            load_ids=load_ids,
            feature_selects=feature_selects,
            col_selects=col_selects,
            col_excludes=col_excludes,
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
    # TODO: assert system_info

    with ctx.tracer.trace_io():
        pyus = {p: PYU(p) for p in v_headers}
        filepaths = {
            pyus[p]: os.path.join(ctx.local_fs_wd, parties_path_format[p].uri)
            for p in v_headers
        }
        dtypes = {pyus[p]: v_headers[p] for p in v_headers}
        vdf = read_csv(filepaths, dtypes=dtypes, nrows=nrows)
        wait(vdf)
    if return_schema_names:
        return vdf, schema_names
    return vdf


def load_table_select_and_exclude_pair(
    ctx,
    db: DistData,
    load_features: bool = False,
    load_labels: bool = False,
    load_ids: bool = False,
    col_selects: List[str] = None,  # if None, load all cols
    to_pandas: bool = True,
    nrows: int = None,
):
    """
    Load two tables, one is the selected, another is the complement.
    """
    trans_x = load_table(
        ctx,
        db,
        load_features,
        load_labels,
        load_ids,
        col_selects=col_selects,
        nrows=nrows,
    )

    remain_x = load_table(
        ctx,
        db,
        load_features=True,
        load_ids=True,
        load_labels=True,
        col_excludes=col_selects,
        nrows=nrows,
    )
    if to_pandas:
        trans_x = trans_x.to_pandas()
        remain_x = remain_x.to_pandas()
    return trans_x, remain_x


def move_feature_to_label(schema: TableSchema, label: str) -> TableSchema:
    new_schema = TableSchema()
    new_schema.CopyFrom(schema)
    if label in list(schema.features) and label not in list(schema.labels):
        new_schema.ClearField('features')
        new_schema.ClearField('feature_types')
        for k, v in zip(list(schema.features), list(schema.feature_types)):
            if k != label:
                new_schema.features.append(k)
                new_schema.feature_types.append(v)
            else:
                label_type = v
        new_schema.labels.append(label)
        new_schema.label_types.append(label_type)
    return new_schema


@dataclass
class VerticalTableWrapper:
    line_count: int
    schema_map: Dict[str, TableSchema]

    def to_vertical_table(self, order: List[str] = None):
        if order is None:
            schemas = list(self.schema_map.values())
        else:
            schemas = [self.schema_map[k] for k in order]

        return VerticalTable(schemas=schemas, line_count=self.line_count)

    @classmethod
    def from_dist_data(cls, data: DistData, line_count: int = None):
        meta = VerticalTable()
        assert data.meta.Unpack(meta)

        return cls(
            line_count=meta.line_count if line_count is None else line_count,
            schema_map={
                data_ref.party: schema
                for data_ref, schema in zip(list(data.data_refs), list(meta.schemas))
            },
        )


def dump_vertical_table(
    ctx,
    v_data: VDataFrame,
    uri: str,
    meta: VerticalTableWrapper,
    system_info: SystemInfo,
) -> DistData:
    assert isinstance(v_data, VDataFrame)

    with ctx.tracer.trace_io():
        output_uri = {p: uri for p in v_data.partitions}
        output_path = {
            p: os.path.join(ctx.local_fs_wd, output_uri[p]) for p in output_uri
        }
        wait(v_data.to_csv(output_path, index=False))
        order = [p.party for p in v_data.partitions]

    ret = DistData(
        name=uri,
        type=str(DistDataType.VERTICAL_TABLE),
        system_info=system_info,
        data_refs=[
            DistData.DataRef(uri=output_uri[p], party=p.party, format="csv")
            for p in output_uri
        ],
    )
    ret.meta.Pack(meta.to_vertical_table(order))

    return ret


def model_dumps(
    model_name: str,
    model_type: str,
    major_version: int,
    minor_version: int,
    objs: List[DeviceObject],
    public_info: Any,
    storage_root: str,
    dist_data_uri: str,
    system_info: SystemInfo,
) -> DistData:
    objs_uri = []
    objs_party = []
    saved_objs = []
    for i, obj in enumerate(objs):
        if isinstance(obj, PYUObject):
            device: PYU = obj.device
            uri = f"{dist_data_uri}/{i}"
            path = os.path.join(storage_root, uri)

            def dumps(path: str, obj: Any):
                import pickle
                from pathlib import Path

                # create parent folders.
                file = Path(path)
                file.parent.mkdir(parents=True, exist_ok=True)

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
            uris = [f"{dist_data_uri}/{i}" for party in device.actors.keys()]
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
        system_info=system_info,
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
    # TODO: assert system_info
    # system_info: SystemInfo = None,
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


def save_prediction_csv(
    pred_df: pd.DataFrame,
    pred_key: str,
    path: str,
    label_df: pd.DataFrame = None,
    label_keys: List[str] = None,
    id_df: pd.DataFrame = None,
    id_keys: List[str] = None,
) -> None:
    x = pd.DataFrame(pred_df, columns=[pred_key])

    if label_df is not None:
        label = pd.DataFrame(label_df, columns=label_keys)
        x = pd.concat([x, label], axis=1)
    if id_df is not None:
        id = pd.DataFrame(id_df, columns=id_keys)
        x = pd.concat([x, id], axis=1)

    x.to_csv(path, index=False)


def gen_prediction_csv_meta(
    id_header: Dict[str, Dict[str, np.dtype]],
    label_header: Dict[str, Dict[str, np.dtype]],
    party: str,
    pred_name: str,
    line_count: int = None,
    id_keys: List[str] = None,
    label_keys: List[str] = None,
) -> IndividualTable:
    return IndividualTable(
        schema=TableSchema(
            ids=id_keys if id_keys is not None else [],
            id_types=[REVERSE_DATA_TYPE_MAP[id_header[party][k]] for k in id_keys]
            if id_keys is not None
            else [],
            labels=(label_keys if label_keys is not None else []) + [pred_name],
            label_types=(
                [REVERSE_DATA_TYPE_MAP[label_header[party][k]] for k in label_keys]
                if label_keys is not None
                else []
            )
            + ["float"],
        ),
        line_count=line_count if line_count is not None else -1,
    )
