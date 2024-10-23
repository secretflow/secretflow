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
import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import duckdb
import duckdb.typing
import numpy as np
import pandas as pd

from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.device.spu import SPU, SPUObject
from secretflow.device.driver import DeviceObject, reveal, wait
from secretflow.error_system.exceptions import (
    CompEvalError,
    DataFormatError,
    InvalidArgumentError,
    NotSupportedError,
    PartyNotFoundError,
    SFModelError,
    SFTrainingHyperparameterError,
)
from secretflow.spec.extend.data_pb2 import DeviceObjectCollection
from secretflow.spec.v1.component_pb2 import IoDef
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    SystemInfo,
    TableSchema,
    VerticalTable,
)
from secretflow.utils import secure_pickle as pickle


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
    # tables
    VERTICAL_TABLE = "sf.table.vertical_table"
    INDIVIDUAL_TABLE = "sf.table.individual"
    # models
    SS_SGD_MODEL = "sf.model.ss_sgd"
    SS_GLM_MODEL = "sf.model.ss_glm"
    SGB_MODEL = "sf.model.sgb"
    SS_XGB_MODEL = "sf.model.ss_xgb"
    SL_NN_MODEL = "sf.model.sl_nn"
    # binning rule
    BIN_RUNNING_RULE = "sf.rule.binning"
    # others preprocessing rules
    PREPROCESSING_RULE = "sf.rule.preprocessing"
    # report
    REPORT = "sf.report"
    # read data
    READ_DATA = "sf.read_data"
    # serving model file
    SERVING_MODEL = "sf.serving.model"
    # checkpoints
    SS_GLM_CHECKPOINT = "sf.checkpoint.ss_glm"
    SGB_CHECKPOINT = "sf.checkpoint.sgb"
    SS_XGB_CHECKPOINT = "sf.checkpoint.ss_xgb"
    SS_SGD_CHECKPOINT = "sf.checkpoint.ss_sgd"
    # if input of component is optional, then the corresponding type can be NULL
    NULL = "sf.null"


@enum.unique
class DataSetFormatSupported(BaseEnum):
    CSV = "csv"
    ORC = "orc"


SUPPORTED_VTABLE_DATA_TYPE = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    # remove float16
    # "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
    "int": int,
    "float": float,
    "str": np.object_,
}

_same_np_type = {
    float: np.float64,
    int: np.int64,
    np.int_: np.int64,
    bool: np.bool_,
    np.object_: np.str_,
    str: np.str_,
}


def is_same_np_type(t1: np.dtype, t2: np.dtype):
    t1 = _same_np_type.get(t1, t1)
    t2 = _same_np_type.get(t2, t2)
    return t1 == t2


NP_DTYPE_TO_DUCKDB_DTYPE = {
    np.int8: duckdb.typing.TINYINT,
    np.int16: duckdb.typing.SMALLINT,
    np.int32: duckdb.typing.INTEGER,
    np.int64: duckdb.typing.BIGINT,
    np.uint8: duckdb.typing.UTINYINT,
    np.uint16: duckdb.typing.USMALLINT,
    np.uint32: duckdb.typing.UINTEGER,
    np.uint64: duckdb.typing.UBIGINT,
    np.float32: duckdb.typing.FLOAT,
    np.float64: duckdb.typing.DOUBLE,
    np.bool_: duckdb.typing.BOOLEAN,
    int: duckdb.typing.INTEGER,
    float: duckdb.typing.DOUBLE,
    np.object_: duckdb.typing.VARCHAR,
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
            raise ValueError(
                f"DistData {data.name}: data_refs is greater than 1 for {data.type}"
            )


@dataclass
class DistDataInfo:
    uri: str
    format: str
    null_strs: List[str]
    dtypes: Dict[str, np.dtype]
    id_cols: List[str]
    feature_cols: List[str]
    label_cols: List[str]

    def mate_equal(self, others: "DistDataInfo"):
        return (
            self.dtypes == others.dtypes
            and set(self.label_cols) == set(self.label_cols)
            and set(self.feature_cols) == set(others.feature_cols)
            and set(self.id_cols) == set(others.id_cols)
        )


def extract_data_infos(
    db: DistData,
    partitions_order: List[str] = None,
    load_features: bool = False,
    load_labels: bool = False,
    load_ids: bool = False,
    col_selects: List[str] = None,
    col_excludes: List[str] = None,
) -> Dict[str, DistDataInfo]:
    """
    Args:
        db (DistData): input DistData.
        load_features (bool, optional): Whether to load feature cols. Defaults to False.
        load_labels (bool, optional): Whether to load label cols. Defaults to False.
        load_ids (bool, optional): Whether to load id cols. Defaults to False.
        col_selects (List[str], optional): Load part of cols. Applies to all cols. Defaults to None.
        col_excludes (List[str], optional): Load all cols exclude these. Applies to all cols. Defaults to None. Couldn't use with col_selects.
    """
    assert (
        load_features or load_labels or load_ids
    ), "At least one load flag should be true"

    assert (
        db.type.lower() == DistDataType.INDIVIDUAL_TABLE
        or db.type.lower() == DistDataType.VERTICAL_TABLE
    ), f"path format {db.type.lower()} should be sf.table.individual or sf.table.vertical_table"

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
    col_selects_set = None
    if col_selects is not None:
        col_selects_set = set(col_selects)
        if len(col_selects) != len(col_selects_set):
            raise DataFormatError.duplicate(
                f"no repetition allowed in col_selects, got {col_selects}"
            )

    if col_excludes is not None:
        col_excludes = set(col_excludes)

    if col_selects_set is not None and col_excludes is not None:
        intersection = set.intersection(col_selects_set, col_excludes)
        if len(col_selects_set & col_excludes) != 0:
            raise DataFormatError.col_selects_intersected_with_col_excludes(
                col_selects=col_selects_set, col_excludes=col_excludes
            )

    ret = dict()
    label_party_name = None
    for slice, dr in zip(schemas, db.data_refs):
        dtype = dict()
        feature_cols = []
        if load_features:
            for t, h in zip(slice.feature_types, slice.features):
                if col_selects_set is not None:
                    if h not in col_selects_set:
                        # feature not selected, skip
                        continue
                    col_selects_set.remove(h)

                if col_excludes is not None:
                    if h in col_excludes:
                        continue

                t = t.lower()
                if t not in SUPPORTED_VTABLE_DATA_TYPE:
                    raise NotSupportedError.not_supported_feature_type(feature_type=t)
                dtype[h] = SUPPORTED_VTABLE_DATA_TYPE[t]
                feature_cols.append(h)
        label_cols = []
        if load_labels:
            for t, h in zip(slice.label_types, slice.labels):
                if col_selects_set is not None:
                    if h not in col_selects_set:
                        # label not selected, skip
                        continue
                    col_selects_set.remove(h)

                if col_excludes is not None:
                    if h in col_excludes:
                        continue
                dtype[h] = SUPPORTED_VTABLE_DATA_TYPE[t]
                label_cols.append(h)
            if len(label_cols) > 0:
                label_party_name = dr.party
        id_cols = []
        if load_ids:
            for t, h in zip(slice.id_types, slice.ids):
                if col_selects_set is not None:
                    if h not in col_selects_set:
                        # id not selected, skip
                        continue
                    col_selects_set.remove(h)

                if col_excludes is not None:
                    if h in col_excludes:
                        continue
                dtype[h] = SUPPORTED_VTABLE_DATA_TYPE[t]
                id_cols.append(h)

        if len(dtype):
            # reorder items according to col selects
            if col_selects is not None and len(col_selects) > 0:
                dtype = {i: dtype[i] for i in col_selects if i in dtype}

            if dr.format.lower() not in DataSetFormatSupported:
                raise NotSupportedError.not_supported_file_format(
                    format={dr.format}, supported_format=DataSetFormatSupported
                )
            ret[dr.party] = DistDataInfo(
                dr.uri,
                dr.format.lower(),
                list(dr.null_strs),
                dtype,
                id_cols,
                feature_cols,
                label_cols,
            )

    if col_selects_set is not None and len(col_selects_set) > 0:
        raise DataFormatError.unknown_header_field(
            column_names=col_selects_set,
        )

    def reorder_partitions(d: Dict[str, Any]):
        if partitions_order is None:
            return d
        # Assumed label from one party
        # partitions order may not contain label holder party
        # this is because label party has no features
        # in this case, add the label holder party to the end
        set_partitions = set(partitions_order)
        set_d_keys = set(d.keys())
        if (label_party_name in set_d_keys) and (
            label_party_name not in set_partitions
        ):
            partitions_order.append(label_party_name)
            set_partitions = set(partitions_order)
        assert (
            set_partitions == set_d_keys
        ), f"{set_partitions} <> {set_d_keys}, {label_party_name}, {(len(set_partitions) == len(set_d_keys) - 1)}, { (label_party_name not in set_partitions)}"
        return {k: d[k] for k in partitions_order}

    return reorder_partitions(ret)


@dataclass
class TableMetaWrapper:
    line_count: int
    schema_map: Dict[str, TableSchema]

    def to_vertical_table(self, order: List[str] = None):
        assert len(self.schema_map) > 1
        if order is None:
            schemas = list(self.schema_map.values())
        else:
            schemas = [self.schema_map[k] for k in order]

        return VerticalTable(schemas=schemas, line_count=self.line_count)

    def to_individual_table(self):
        assert len(self.schema_map) == 1
        return IndividualTable(
            schema=list(self.schema_map.values())[0], line_count=self.line_count
        )

    def dist_table_type(self) -> str:
        assert len(self.schema_map) >= 1
        if len(self.schema_map) > 1:
            return str(DistDataType.VERTICAL_TABLE)
        return str(DistDataType.INDIVIDUAL_TABLE)

    def to_dist_data(self, order: List[str]):
        assert len(self.schema_map) >= 1
        if len(self.schema_map) > 1:
            return self.to_vertical_table(order)
        return self.to_individual_table()

    @classmethod
    def from_dist_info(cls, dist_info: Dict[PYU, DistDataInfo]):
        return cls(
            line_count=0,
            schema_map={
                pyu.party: TableSchema(
                    id_types=[
                        REVERSE_DATA_TYPE_MAP[info.dtypes[c]] for c in info.id_cols
                    ],
                    feature_types=[
                        REVERSE_DATA_TYPE_MAP[info.dtypes[c]] for c in info.feature_cols
                    ],
                    label_types=[
                        REVERSE_DATA_TYPE_MAP[info.dtypes[c]] for c in info.label_cols
                    ],
                    features=info.feature_cols,
                    labels=info.label_cols,
                    ids=info.id_cols,
                )
                for pyu, info in dist_info.items()
            },
        )

    @classmethod
    def from_dist_data(
        cls, data: DistData, line_count: int = None, feature_selects: List[str] = None
    ):
        if data.type == DistDataType.VERTICAL_TABLE:
            meta = VerticalTable()
            if not data.meta.Unpack(meta):
                raise DataFormatError.unpack_distdata_error(unpack_type="VerticalTable")
            if len(data.data_refs) != len(meta.schemas):
                raise DataFormatError.dataset_not_aligned(
                    "the length of schemas is not equal to the length of data_refs"
                )

            return cls(
                line_count=meta.line_count if line_count is None else line_count,
                schema_map={
                    data_ref.party: schema
                    for data_ref, schema in zip(
                        list(data.data_refs), list(meta.schemas)
                    )
                },
            )
        elif data.type == DistDataType.INDIVIDUAL_TABLE:
            meta = IndividualTable()
            if not data.meta.Unpack(meta):
                raise DataFormatError.unpack_distdata_error(
                    unpack_type="IndividualTable"
                )
            if len(data.data_refs) != 1:
                raise DataFormatError.dataset_not_aligned(
                    "the length of data_refs should be 1, got {len(data.data_refs)}"
                )

            party = data.data_refs[0].party
            return cls(
                line_count=meta.line_count if line_count is None else line_count,
                schema_map={party: meta.schema},
            )
        else:
            raise NotSupportedError.not_supported_data_type(
                f"TableMetaWrapper unsupported data format type: {data.type}"
            )


def model_dumps(
    ctx,
    model_name: str,
    model_type: str,
    major_version: int,
    minor_version: int,
    objs: List[DeviceObject],
    public_info: Any,
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

            def dumps(comp_storage, uri: str, obj: Any):
                with comp_storage.get_writer(uri) as w:
                    pickle.dump(obj, w)

            wait(device(dumps)(ctx.comp_storage, uri, obj))

            saved_obj = DeviceObjectCollection.DeviceObject(
                type="pyu", data_ref_idxs=[len(objs_uri)]
            )
            saved_objs.append(saved_obj)
            objs_uri.append(uri)
            objs_party.append(device.party)
        elif isinstance(obj, SPUObject):
            device: SPU = obj.device
            uris = [f"{dist_data_uri}/{i}" for _ in device.actors]

            device.dump(
                obj,
                [lambda uri=uri: ctx.comp_storage.get_writer(uri) for uri in uris],
            )

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


def get_model_info(dist_data: DistData):
    model_meta = DeviceObjectCollection()
    if not dist_data.meta.Unpack(model_meta):
        raise DataFormatError.unpack_distdata_error(
            unpack_type="DeviceObjectCollection"
        )
    model_info = json.loads(model_meta.public_info)
    if not isinstance(model_info, dict):
        raise SFModelError.model_info_error(reason="model_info is not a dict")
    return model_info, model_meta


def get_model_public_info(dist_data: DistData):
    model_info, _ = get_model_info(dist_data)
    return json.loads(model_info["public_info"])


def get_model_info_check_version(
    dist_data: DistData, max_major_version: int, max_minor_version: int
):
    model_info, model_meta = get_model_info(dist_data)

    if not {"major_version", "minor_version", "public_info"}.issubset(model_info):
        raise SFModelError.model_info_error(
            reason="model_info should contain keys: major_version, minor_version, public_info"
        )

    if not (
        max_major_version >= model_info["major_version"]
        and max_minor_version >= model_info["minor_version"]
    ):
        raise SFModelError.model_info_error(
            reason=f"not support model version, support {max_major_version}.{max_minor_version}, \
                input {model_info['major_version']}.{model_info['minor_version']}"
        )

    return model_info, model_meta


def model_meta_info(
    dist_data: DistData,
    max_major_version: int,
    max_minor_version: int,
    model_type: str,
    # TODO: assert system_info
    # system_info: SystemInfo = None,
) -> Tuple[List[DeviceObject], str]:
    '''
    load model from dist_data, return public_info
    '''
    if dist_data.type != model_type:
        raise SFModelError.model_meta_error(
            f"model type of dist_data '{dist_data.type}' does not match to model_type '{model_type}'"
        )

    model_info, _ = get_model_info_check_version(
        dist_data, max_major_version, max_minor_version
    )

    return model_info["public_info"]


def model_loads(
    ctx,
    dist_data: DistData,
    major_version: int,
    max_minor_version: int,
    model_type: str,
    pyus: Dict[str, PYU] = None,
    spu: SPU = None,
    # TODO: assert system_info
    # system_info: SystemInfo = None,
) -> Tuple[List[DeviceObject], str]:
    if dist_data.type != model_type:
        raise SFModelError.model_meta_error(
            f"model type of dist_data '{dist_data.type}' does not match to model_type '{model_type}'"
        )

    model_info, model_meta = get_model_info_check_version(
        dist_data, major_version, max_minor_version
    )

    objs = []
    for save_obj in model_meta.objs:
        if save_obj.type == "pyu":
            if len(save_obj.data_ref_idxs) != 1:
                raise SFModelError.model_meta_error(
                    "only support one data_ref_idxs for pyu"
                )

            data_ref = dist_data.data_refs[save_obj.data_ref_idxs[0]]
            party = data_ref.party
            if pyus is not None:
                if party not in pyus:
                    raise PartyNotFoundError(
                        f"party {party} not in '{','.join(pyus.keys())}'"
                    )
                pyu = pyus[party]
            else:
                pyu = PYU(party)

            if data_ref.format != "pickle":
                raise DataFormatError.wrong_data_type(
                    data_name=dist_data.data_refs, data_type='pickle'
                )

            def loads(comp_storage, path: str) -> Any:
                with comp_storage.get_reader(path) as r:
                    return pickle.load(r)

            objs.append(pyu(loads)(ctx.comp_storage, data_ref.uri))
        elif save_obj.type == "spu":
            # TODO: only support one spu for now
            if spu is None:
                raise SFModelError.model_meta_error(
                    "input spu is None when loading model of type spu"
                )
            if len(save_obj.data_ref_idxs) <= 1:
                raise SFModelError.model_meta_error(
                    "len of data_ref_idxs of spu should be larger than 1"
                )
            full_paths = {}
            for data_ref_idx in save_obj.data_ref_idxs:
                data_ref = dist_data.data_refs[data_ref_idx]
                if data_ref.format != "pickle":
                    raise DataFormatError.wrong_data_type(
                        data_name=dist_data.data_refs, data_type='pickle'
                    )
                party = data_ref.party
                if party in full_paths:
                    raise SFModelError.model_party_incorrect(
                        f"found duplicated party {party} in dist_data.data_refs"
                    )
                uri = data_ref.uri
                full_paths[party] = lambda uri=uri: ctx.comp_storage.get_reader(uri)
            if set(full_paths.keys()) != set(spu.actors.keys()):
                raise SFModelError.model_party_incorrect(
                    f"party of dist_data.data_refs not match with spu.actors, "
                    f"dist_data.data_refs: {set(full_paths.keys())}, spu.actors: {set(spu.actors.keys())}"
                )
            spu_paths = [full_paths[party] for party in spu.actors.keys()]
            objs.append(spu.load(spu_paths))
        else:
            raise NotSupportedError.not_supported_data_type(
                f"not supported objs type {save_obj.type}"
            )

    return objs, model_info["public_info"]


def save_prediction_csv(
    pred_df: pd.DataFrame,
    pred_key: str,
    path: str,
    addition_df: Union[List[pd.DataFrame], List[np.array]] = None,
    addition_keys: List[str] = None,
    try_append: bool = False,
) -> None:
    x = pd.DataFrame(pred_df, columns=[pred_key])

    addition_df = [
        df if isinstance(df, pd.DataFrame) else pd.DataFrame(df) for df in addition_df
    ]

    if addition_df:
        if not addition_keys:
            raise InvalidArgumentError(
                "addition_keys is None while addition_df is not empty"
            )
        addition_data = pd.concat(addition_df, axis=1)
        addition_data.columns = addition_keys
        x = pd.concat([x, addition_data], axis=1)

    if try_append:
        if not os.path.isfile(path):
            x.to_csv(path, index=False)
        else:
            x.to_csv(path, mode='a', header=False, index=False)
    else:
        x.to_csv(path, index=False)


def download_files(
    ctx,
    remote_fns: Dict[Union[str, PYU], str],
    local_fns: Dict[Union[str, PYU], Union[str, DistDataInfo]],
    overwrite: bool = True,
):
    pyu_remotes = {
        p.party if isinstance(p, PYU) else p: remote_fns[p] for p in remote_fns
    }
    pyu_locals = {}
    for p in local_fns:
        k = p.party if isinstance(p, PYU) else p
        v = local_fns[p] if isinstance(local_fns[p], str) else local_fns[p].uri
        pyu_locals[k] = v

    if set(pyu_remotes.keys()) != set(pyu_locals.keys()):
        raise CompEvalError.party_check_failed(f"{pyu_remotes} <> {pyu_locals}")

    def download_file(comp_storage, uri, output_path):
        if not overwrite and os.path.exists(output_path):
            # skip download
            if not os.path.isfile(output_path):
                raise SFTrainingHyperparameterError.not_a_file(
                    f"In download_file, when choosing not overwrite, the output_path {output_path} should be a file"
                )
        else:
            comp_storage.download_file(uri, output_path)

    waits = []
    for p in pyu_remotes:
        remote_fn = pyu_remotes[p]
        local_fn = pyu_locals[p]
        waits.append(PYU(p)(download_file)(ctx.comp_storage, remote_fn, local_fn))

    wait(waits)


def any_pyu_from_spu_config(config: dict):
    return PYU(config["nodes"][0]["party"])


def generate_random_string(pyu: PYU):
    return reveal(pyu(lambda: str(uuid.uuid4()))())
