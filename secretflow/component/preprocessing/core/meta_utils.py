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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.∏
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from typing import Dict, List

from google.protobuf.json_format import MessageToJson, Parse

from secretflow.component.data_utils import (
    SUPPORTED_VTABLE_DATA_TYPE,
    VerticalTableWrapper,
)
from secretflow.spec.extend import preprocessing_meta_pb2
from secretflow.spec.v1.data_pb2 import TableSchema


def result_pandas_type(ref_dtypes, key):
    type_str = str(ref_dtypes[key])
    if type_str == 'object':
        return "str"
    else:
        assert type_str in SUPPORTED_VTABLE_DATA_TYPE
        return type_str


def create_one_info(append_key, fixed_type, ref_dtypes):
    info = preprocessing_meta_pb2.AddColumnInfo()
    info.name = append_key
    info.type = fixed_type if fixed_type else result_pandas_type(ref_dtypes, append_key)
    return info


# Function to create an InfoChange message for a specific field
def create_add_columns(append_keys, fixed_type, ref_dtypes):
    return [
        create_one_info(append_key, fixed_type, ref_dtypes)
        for append_key in append_keys
    ]


# Produce meta change for a field
def produce_meta_change(
    meta: VerticalTableWrapper,
    drop_cols: Dict[str, List],
    new_features: Dict[str, List] = None,
    new_labels: Dict[str, List] = None,
    fixed_type: str = None,
    ref_dtypes: Dict[str, type] = None,
) -> Dict[str, str]:
    assert (fixed_type is not None) or (ref_dtypes is not None)
    meta_change_dict = {}
    for party in meta.schema_map.keys():
        if party in drop_cols:
            try_drop_ids = set(drop_cols[party]).intersection(
                set(meta.schema_map[party].ids)
            )
            assert len(try_drop_ids) == 0, "do not modify id columns"
        party_meta_change = preprocessing_meta_pb2.TableMetaChange()
        if party in drop_cols:
            party_meta_change.drop_columns.extend(drop_cols[party])
        if new_features is not None and party in new_features:
            append_keys = new_features[party]
            party_meta_change.new_features.extend(
                create_add_columns(append_keys, fixed_type, ref_dtypes)
            )
        if new_labels is not None and party in new_labels:
            append_keys = new_labels[party]
            party_meta_change.new_labels.extend(
                create_add_columns(append_keys, fixed_type, ref_dtypes)
            )
        meta_change_dict[party] = MessageToJson(party_meta_change)
    return meta_change_dict


def dict_to_str(meta_change_dict: Dict[str, str]) -> str:
    return json.dumps(meta_change_dict)


def str_to_dict(meta_change_json_str: str) -> Dict[str, str]:
    return json.loads(meta_change_json_str)


# field maybe id, label or feature
def apply_method_for_field(
    new_schema_fields,
    new_schema_field_types,
    schema_fields,
    schema_field_types,
    meta_new_fields: List[preprocessing_meta_pb2.AddColumnInfo],
    drop_key: List[str],
):
    new_schema_fields[:] = []
    new_schema_field_types[:] = []
    for idx in range(len(schema_fields)):
        if schema_fields[idx] not in drop_key:
            new_schema_fields.append(schema_fields[idx])
            new_schema_field_types.append(schema_field_types[idx])

    for info in meta_new_fields:
        new_schema_fields.append(info.name)
        new_schema_field_types.append(info.type)


def apply_meta_change(
    meta: VerticalTableWrapper,
    meta_change_dict: Dict[str, str],
) -> VerticalTableWrapper:
    for party, meta_change_serialized in meta_change_dict.items():
        table_schema = meta.schema_map[party]
        new_schema = TableSchema()
        new_schema.CopyFrom(table_schema)
        meta_change = preprocessing_meta_pb2.TableMetaChange()
        Parse(meta_change_serialized, meta_change)

        apply_method_for_field(
            new_schema.features,
            new_schema.feature_types,
            table_schema.features,
            table_schema.feature_types,
            meta_change.new_features,
            meta_change.drop_columns,
        )
        apply_method_for_field(
            new_schema.labels,
            new_schema.label_types,
            table_schema.labels,
            table_schema.label_types,
            meta_change.new_labels,
            meta_change.drop_columns,
        )
        # save to meta
        meta.schema_map[party] = new_schema
    return meta
