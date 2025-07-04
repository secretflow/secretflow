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

import json
import logging
from typing import List

from google.protobuf.json_format import MessageToJson
from kuscia.proto.api.v1alpha1.common_pb2 import DataColumn
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2 import DomainData
from secretflow_spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)

from secretflow.kuscia.task_config import TableAttr


def convert_domain_data_to_individual_table(
    domain_data: DomainData, table_attr: TableAttr = None
) -> IndividualTable:
    assert domain_data.type == 'table'
    dist_data = DistData(
        name=domain_data.name,
        type="sf.table.individual",
        data_refs=[
            DistData.DataRef(
                uri=domain_data.relative_uri,
                party=domain_data.author,
                format='csv',
                null_strs=(
                    json.loads(domain_data.attributes["NullStrs"])
                    if "NullStrs" in domain_data.attributes
                    else []
                ),
            )
        ],
    )

    attr_dict = (
        {col.col_name: col.col_type for col in table_attr.column_attrs}
        if table_attr
        else {}
    )

    ids, features, labels = [], [], []
    for col in domain_data.columns:
        if col.name in attr_dict:
            attr_type = attr_dict[col.name]
            assert attr_type in (
                'id',
                'feature',
                'label',
                "binned",
            ), f"{col.name} is not in [id, feature, binned, label]"
            if attr_type == 'id':
                ids.append((col.name, col.type))
            elif attr_type == 'feature' or attr_type == 'binned':
                # binned feature is treated as feature
                features.append((col.name, col.type))
            elif attr_type == 'label':
                labels.append((col.name, col.type))
        else:
            logging.info(f"{col.name} is not in table attr, treat it as feature")
            features.append((col.name, col.type))

    meta = IndividualTable(
        schema=TableSchema(
            ids=[col[0] for col in ids],
            id_types=[col[1] for col in ids],
            features=[col[0] for col in features],
            feature_types=[col[1] for col in features],
            labels=[col[0] for col in labels],
            label_types=[col[1] for col in labels],
        ),
        line_count=-1,
    )

    dist_data.meta.Pack(meta)

    return dist_data


def convert_dist_data_to_domain_data(
    domaindata_id: str,
    datasource_id: str,
    x: DistData,
    output_uri: str,
    party: str,
    partition_spec: str,
) -> DomainData:
    def convert_data_type(dist_data_type: str) -> str:
        if dist_data_type.startswith("sf.table"):
            return "table"
        elif dist_data_type.startswith("sf.model"):
            return "model"
        elif dist_data_type.startswith("sf.rule"):
            return "rule"
        elif dist_data_type == "sf.report":
            return "report"
        elif dist_data_type == "sf.read_data":
            return "read_data"
        elif dist_data_type == "sf.serving.model":
            return "serving_model"
        return "unknown"

    def get_data_columns(x: DistData, party: str) -> List[DataColumn]:
        ret = []
        if (
            x.type == "sf.table.individual"
            or x.type == "sf.table.vertical"
            or x.type == "sf.table.vertical_table"
        ):
            meta = (
                IndividualTable()
                if x.type.lower() == "sf.table.individual"
                else VerticalTable()
            )

            assert x.meta.Unpack(meta)

            schemas = (
                [meta.schema]
                if x.type.lower() == "sf.table.individual"
                else meta.schemas
            )

            for schema, data_ref in zip(schemas, list(x.data_refs)):
                if data_ref.party != party:
                    continue
                for id, type in zip(list(schema.ids), list(schema.id_types)):
                    ret.append(DataColumn(name=id, type=type, comment="id"))

                for feature, type in zip(
                    list(schema.features), list(schema.feature_types)
                ):
                    ret.append(DataColumn(name=feature, type=type, comment="feature"))

                for label, type in zip(list(schema.labels), list(schema.label_types)):
                    ret.append(DataColumn(name=label, type=type, comment="label"))

        return ret

    domain_data = DomainData(
        domaindata_id=domaindata_id,
        name=x.name,
        type=convert_data_type(x.type),
        relative_uri=output_uri,
        datasource_id=datasource_id,
        vendor="secretflow",
    )

    domain_data.attributes["dist_data"] = MessageToJson(
        x, including_default_value_fields=True, indent=0
    )
    if partition_spec:
        domain_data.attributes["partition_spec"] = partition_spec

    domain_data.columns.extend(get_data_columns(x, party))

    return domain_data
