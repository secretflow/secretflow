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
import uuid

import grpc
from dataproxy import (
    DataProxyConfig,
    DataProxyFileAdapter,
    DownloadInfo,
    FileFormat,
    TlSConfig,
    UploadInfo,
)
from kuscia.proto.api.v1alpha1.common_pb2 import DataColumn
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2 import (
    DomainData,
    QueryDomainDataRequest,
)
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2_grpc import DomainDataServiceStub
from secretflow_spec import (
    Storage,
    StorageType,
    StrEnum,
    VTableField,
    VTableFieldKind,
    VTableFormat,
    VTableSchema,
)

from .connector import IConnector, TableInfo


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


def create_channel():
    address = os.environ.get("DATAMESH_ADDRESS", "")
    env_client_cert_file = os.environ.get("CLIENT_CERT_FILE", '')
    env_client_key_file = os.environ.get("CLIENT_PRIVATE_KEY_FILE", '')
    env_trusted_ca_file = os.environ.get("TRUSTED_CA_FILE", '')

    if env_client_cert_file:
        # mTLS enabled.
        with open(env_client_cert_file, 'rb') as client_cert, open(
            env_client_key_file, 'rb'
        ) as client_key, open(env_trusted_ca_file, 'rb') as trusted_ca:
            credentials = grpc.ssl_channel_credentials(
                trusted_ca.read(), client_key.read(), client_cert.read()
            )
            channel = grpc.secure_channel(address, credentials)
    else:
        channel = grpc.insecure_channel(address)

    return channel


def create_domain_data_service_stub(channel):
    return DomainDataServiceStub(channel)


def get_domain_data(stub: DomainDataServiceStub, id: str) -> DomainData:
    ret = stub.QueryDomainData(QueryDomainDataRequest(domaindata_id=id))
    if ret.status.code != 0:
        raise RuntimeError(f"get_dist_data failed for {id}: status = {ret.status}")

    return ret.data


class FileType(StrEnum):
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
    def download_table(
        self,
        storage: Storage,
        data_dir: str,
        input_path: str,
        input_params: dict,
        output_uri: str,
        output_format: VTableFormat = VTableFormat.ORC,
    ) -> TableInfo:
        domaindata_id = input_params.pop("domaindata_id", "")
        partition_spec = input_params.pop("partition_spec", "")
        if domaindata_id == "":
            raise ValueError(f"empty domaindata id")

        channel = create_channel()
        stub = create_domain_data_service_stub(channel)
        domain_data = get_domain_data(stub, domaindata_id)

        columns = []
        for col in domain_data.columns:
            kind = VTableFieldKind.from_str(col.comment)
            if kind == VTableFieldKind.UNKNOWN:
                kind = VTableFieldKind.FEATURE
            columns.append(VTableField(col.name, col.type, kind))

        download_info = DownloadInfo(
            domaindata_id=domaindata_id,
            partition_spec=partition_spec,
        )

        is_local_fs = storage.get_type() == StorageType.LOCAL_FS

        local_path = (
            storage.get_full_path(output_uri)
            if is_local_fs
            else os.path.join(data_dir, str(uuid.uuid4()))
        )

        client = new_dataproxy_client()
        client.download_file(download_info, local_path, _to_file_format(output_format))
        client.close()

        if not is_local_fs:
            chunk_size = 64 * 1014 * 1024
            with open(local_path, "rb") as f, storage.get_writer(output_uri) as w:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    w.write(chunk)

        return TableInfo(VTableSchema(columns), -1)

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
        is_local_fs = storage.get_type() == StorageType.LOCAL_FS
        if not is_local_fs:
            local_path = os.path.join(data_dir, input_uri)
            storage.download_file(input_uri, local_path)

        domaindata_id = output_params.pop("domaindata_id", "")
        datasource_id = output_params.pop("datasource_id", "")
        attributes = output_params

        columns = [
            DataColumn(name=f.name, type=f.type, comment=str(f.kind))
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

        if not is_local_fs:
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
