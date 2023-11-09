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

from typing import List

from google.protobuf.json_format import MessageToJson
from kuscia.proto.api.v1alpha1.common_pb2 import DataColumn
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2 import DomainData

from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, VerticalTable


def convert_domain_data_to_individual_table(
    domain_data: DomainData,
) -> IndividualTable:
    import logging

    logging.warning(
        'kuscia adapter has to deduce dist data from domain data at this moment.'
    )
    assert domain_data.type == 'table'
    dist_data = DistData(name=domain_data.name, type="sf.table.individual")

    meta = IndividualTable()
    for col in domain_data.columns:
        if not col.comment or col.comment == 'feature':
            meta.schema.features.append(col.name)
            meta.schema.feature_types.append(col.type)
        elif col.comment == 'id':
            meta.schema.ids.append(col.name)
            meta.schema.id_types.append(col.type)
        elif col.comment == 'label':
            meta.schema.labels.append(col.name)
            meta.schema.label_types.append(col.type)
    meta.line_count = -1
    dist_data.meta.Pack(meta)

    data_ref = DistData.DataRef()
    data_ref.uri = domain_data.relative_uri
    data_ref.party = domain_data.author
    data_ref.format = 'csv'
    dist_data.data_refs.append(data_ref)

    return dist_data


def convert_dist_data_to_domain_data(
    domaindata_id: str, datasource_id: str, x: DistData, output_uri: str, party: str
) -> DomainData:
    def convert_data_type(dist_data_type: str) -> str:
        if "table" in dist_data_type:
            return "table"
        elif "model" in dist_data_type:
            return "model"
        elif "rule" in dist_data_type:
            return "rule"
        elif "report" in dist_data_type:
            return "report"
        return "unknown"

    def get_data_columns(x: DistData, party: str) -> List[DataColumn]:
        ret = []
        if x.type == "sf.table.individual" or x.type == "sf.table.vertical_table":
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
        x, including_default_value_fields=True
    )
    domain_data.columns.extend(get_data_columns(x, party))

    return domain_data
