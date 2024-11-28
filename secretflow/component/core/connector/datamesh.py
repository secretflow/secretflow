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

import os

from dataproxy.sdk import (
    DataProxyConfig,
    DataProxyFileAdapter,
    FileFormat,
    TlSConfig,
    UploadInfo,
)
from kuscia.proto.api.v1alpha1.common_pb2 import DataColumn

from ..common.types import BaseEnum
from ..dist_data.vtable import VTableFormat, VTableSchema
from ..storage import Storage
from .connector import IConnector


def new_dataproxy_client():
    dp_config = DataProxyConfig(
        data_proxy_addr=os.environ.get("DATAMESH_ADDRESS", ""),
        tls_config=TlSConfig(
            certificate_path=os.environ.get("CLIENT_CERT_FILE", ''),
            private_key_path=os.environ.get("CLIENT_PRIVATE_KEY_FILE", ''),
            ca_file_path=os.environ.get("TRUSTED_CA_FILE", ''),
        ),
    )

    return DataProxyFileAdapter(dp_config)


class FileType(BaseEnum):
    TABLE = "table"
    MODEL = "model"


def _to_file_format(ff: VTableFormat) -> FileFormat:
    if ff == VTableFormat.CSV:
        return FileFormat.CSV
    elif ff == VTableFormat.ORC:
        return FileFormat.ORC
    else:
        raise ValueError(f"unknown file format {ff}")


class DataMesh(IConnector):
    def upload_table(
        self,
        storage: Storage,
        data_dir: str,
        input_uri: str,
        input_format: VTableFormat,
        input_schema: VTableSchema,
        output_path: str,
        output_params: dict,
    ):
        local_path = input_uri
        if not storage.is_local_fs():
            local_path = os.path.join(data_dir, input_uri)
            storage.download_file(input_uri, local_path)

        domaindata_id = output_params.pop("domaindata_id", "")
        datasource_id = output_params.pop("datasource_id", "")
        attributes = output_params

        columns = [
            DataColumn(name=f.name, type=f.ftype, comment=str(f.kind))
            for f in input_schema.fields.values()
        ]

        self.upload_file(
            local_path,
            output_path,
            FileType.TABLE,
            _to_file_format(input_format),
            attributes,
            columns,
            datasource_id,
            domaindata_id,
        )

        if not storage.is_local_fs():
            os.remove(local_path)
            local_dir = os.path.dirname(local_path)
            while local_dir != data_dir:
                os.rmdir(local_dir)
                local_dir = os.path.dirname(local_dir)

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        file_type: FileType,
        format: FileFormat,
        attributes: dict,
        columns: list[DataColumn],
        datasource_id: str,
        domaindata_id: str,
    ):
        if domaindata_id == "" or datasource_id == "":
            raise ValueError(f"invalid params {datasource_id} {domaindata_id}")

        upload_info = UploadInfo(
            domaindata_id=domaindata_id,
            name="",
            type=str(file_type),
            datasource_id=datasource_id,
            relative_uri=remote_path,
            attributes=attributes,
            columns=columns,
            vendor="secretflow",
        )
        client = new_dataproxy_client()
        client.upload_file(upload_info, local_path, format)
        client.close()
