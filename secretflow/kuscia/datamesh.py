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


import os

import grpc
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2 import (
    CreateDomainDataRequest,
    DomainData,
    QueryDomainDataRequest,
)
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2_grpc import DomainDataServiceStub
from kuscia.proto.api.v1alpha1.datamesh.domaindatasource_pb2 import (
    DomainDataSource,
    QueryDomainDataSourceRequest,
)
from kuscia.proto.api.v1alpha1.datamesh.domaindatasource_pb2_grpc import (
    DomainDataSourceServiceStub,
)


def create_channel(address: str):
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


def create_domain_data_service_stub(address: str):
    return DomainDataServiceStub(create_channel(address))


def get_domain_data(stub: DomainDataServiceStub, id: str) -> DomainData:
    ret = stub.QueryDomainData(QueryDomainDataRequest(domaindata_id=id))
    if ret.status.code != 0:
        raise RuntimeError(f"get_dist_data failed for {id}: status = {ret.status}")

    return ret.data


def create_domain_data(stub: DomainDataServiceStub, data: DomainData):
    ret = stub.CreateDomainData(
        CreateDomainDataRequest(
            domaindata_id=data.domaindata_id,
            name=data.name,
            type=data.type,
            datasource_id=data.datasource_id,
            relative_uri=data.relative_uri,
            attributes=data.attributes,
            # partition=data.partition,
            columns=data.columns,
            vendor=data.vendor,
        )
    )

    if ret.status.code != 0:
        raise RuntimeError(
            f"create_domain_data failed for {data.domaindata_id}: status = {ret.status}"
        )


def create_domain_data_source_service_stub(address: str):
    return DomainDataSourceServiceStub(create_channel(address))


def get_domain_data_source(
    stub: DomainDataSourceServiceStub, id: str
) -> DomainDataSource:
    ret = stub.QueryDomainDataSource(QueryDomainDataSourceRequest(datasource_id=id))

    if ret.status.code != 0:
        raise RuntimeError(
            f"get_domain_data_source failed for {id}: status = {ret.status}"
        )

    return ret.data
