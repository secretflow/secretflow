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
import pyarrow.csv as csv
import pyarrow.flight as flight
from google.protobuf.any_pb2 import Any
from kuscia.proto.api.v1alpha1.common_pb2 import FileFormat
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
from kuscia.proto.api.v1alpha1.datamesh.flightdm_pb2 import (
    ActionCreateDomainDataRequest,
    ActionCreateDomainDataResponse,
    CommandDomainDataQuery,
    CommandDomainDataUpdate,
    CSVWriteOptions,
    FileWriteOptions,
)

DEFAULT_GENERIC_OPTIONS = [("GRPC_ARG_KEEPALIVE_TIME_MS", 60000)]

DEFAULT_FLIGHT_CALL_OPTIONS = flight.FlightCallOptions(
    timeout=10
)  # timeout unit is second.


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


def create_domain_data_service_stub(channel):
    return DomainDataServiceStub(channel)


def get_domain_data(stub: DomainDataServiceStub, id: str) -> DomainData:
    ret = stub.QueryDomainData(QueryDomainDataRequest(domaindata_id=id))
    if ret.status.code != 0:
        raise RuntimeError(f"get_dist_data failed for {id}: status = {ret.status}")

    return ret.data


def create_domain_data_in_dm(stub: DomainDataServiceStub, data: DomainData):
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


def create_domain_data_source_service_stub(channel):
    return DomainDataSourceServiceStub(channel)


def get_domain_data_source(
    stub: DomainDataSourceServiceStub, id: str
) -> DomainDataSource:
    ret = stub.QueryDomainDataSource(QueryDomainDataSourceRequest(datasource_id=id))

    if ret.status.code != 0:
        raise RuntimeError(
            f"get_domain_data_source failed for {id}: status = {ret.status}"
        )

    return ret.data


def create_dm_flight_client(dm_address: str):
    client_cert_path = os.environ.get("CLIENT_CERT_FILE", '')
    client_key_path = os.environ.get("CLIENT_PRIVATE_KEY_FILE", '')
    trusted_ca_path = os.environ.get("TRUSTED_CA_FILE", '')

    with open(client_cert_path, 'rb') as client_cert, open(
        client_key_path, 'rb'
    ) as client_key, open(trusted_ca_path, 'rb') as trusted_ca:
        dm_flight_client = flight.connect(
            "grpc+tls://" + dm_address,
            tls_root_certs=trusted_ca.read(),
            cert_chain=client_cert.read(),
            private_key=client_key.read(),
            generic_options=DEFAULT_GENERIC_OPTIONS,
        )

        return dm_flight_client


def get_csv_from_dp(
    dm_flight_client,
    domain_data_id: str,
    output_file_path: str,
):
    domain_data_query = CommandDomainDataQuery(
        domaindata_id=domain_data_id,
    )

    any = Any()
    any.Pack(domain_data_query)

    descriptor = flight.FlightDescriptor.for_command(any.SerializeToString())

    flight_info = dm_flight_client.get_flight_info(
        descriptor=descriptor, options=DEFAULT_FLIGHT_CALL_OPTIONS
    )

    dp_uri = flight_info.endpoints[0].locations[0]
    ticket = flight_info.endpoints[0].ticket

    dp_flight_client = flight.connect(dp_uri, generic_options=DEFAULT_GENERIC_OPTIONS)
    flight_reader = dp_flight_client.do_get(ticket=ticket).to_reader()

    # NOTE(junfeng): use pandas to write csv since pyarrow will add quotes in headers.
    for batch in flight_reader:
        batch_pd = batch.to_pandas()
        batch_pd.to_csv(
            output_file_path,
            index=False,
            mode='a',
            header=not os.path.exists(output_file_path),
        )

    dp_flight_client.close()


def create_domain_data_in_dp(
    dm_flight_client,
    datasource_id: str,
    domain_data: DomainData,
):
    create_domain_data_request = CreateDomainDataRequest(
        # NOTE: rm
        domaindata_id=domain_data.domaindata_id,
        name=domain_data.name,
        type=domain_data.type,
        datasource_id=domain_data.datasource_id,
        relative_uri=domain_data.relative_uri,
        attributes=domain_data.attributes,
        # partition=data.partition,
        columns=domain_data.columns,
        vendor=domain_data.vendor,
        file_format=FileFormat.CSV,
        # file_format=FileFormat.UNKNOWN,
    )

    action_create_domain_data_request = ActionCreateDomainDataRequest(
        request=create_domain_data_request
    )

    action = flight.Action(
        "ActionCreateDomainDataRequest",
        action_create_domain_data_request.SerializeToString(),
    )

    results = dm_flight_client.do_action(
        action=action, options=DEFAULT_FLIGHT_CALL_OPTIONS
    )

    for res in results:
        action_response = ActionCreateDomainDataResponse()
        action_response.ParseFromString(res.body)
        assert action_response.response.status.message == "success"


def put_data_to_dp(dm_flight_client, domaindata_id: str, file_local_path: str):
    command_domain_data_update = CommandDomainDataUpdate(
        domaindata_id=domaindata_id,
        file_write_options=FileWriteOptions(
            csv_options=CSVWriteOptions(field_delimiter=",")
        ),
    )

    any = Any()
    any.Pack(command_domain_data_update)

    descriptor = flight.FlightDescriptor.for_command(any.SerializeToString())

    flight_info = dm_flight_client.get_flight_info(
        descriptor=descriptor, options=DEFAULT_FLIGHT_CALL_OPTIONS
    )

    dp_uri = flight_info.endpoints[0].locations[0]
    ticket = flight_info.endpoints[0].ticket

    reader = csv.open_csv(file_local_path)
    schema = reader.schema

    dp_flight_client = flight.connect(dp_uri, generic_options=DEFAULT_GENERIC_OPTIONS)

    descriptor = flight.FlightDescriptor.for_command(ticket.ticket)
    flight_writer, _ = dp_flight_client.do_put(descriptor=descriptor, schema=schema)

    for batch in reader:
        flight_writer.write(batch)

    flight_writer.close()
    reader.close()
    dp_flight_client.close()
