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
from enum import Enum

import grpc
import pyarrow as pa
import pyarrow.csv as csv
from pyarrow import orc
import pyarrow.flight as flight
from google.protobuf.any_pb2 import Any
from kuscia.proto.api.v1alpha1.common_pb2 import DataColumn, FileFormat
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2 import (
    CreateDomainDataRequest,
    CreateDomainDataResponse,
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
    CommandDomainDataQuery,
    CommandDomainDataUpdate,
    ContentType,
    CSVWriteOptions,
    FileWriteOptions,
)

DEFAULT_GENERIC_OPTIONS = [("GRPC_ARG_KEEPALIVE_TIME_MS", 60000)]

DEFAULT_FLIGHT_CALL_OPTIONS = flight.FlightCallOptions(
    timeout=10
)  # timeout unit is second.


class DataFileFormat(Enum):
    CSV = 1
    ORC = 2
    BINARY = 3


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
        raise RuntimeError(f"create_domain_data failed for {data} : ret = {ret}")


def create_domain_data_source_service_stub(channel):
    return DomainDataSourceServiceStub(channel)


def get_domain_data_source(
    stub: DomainDataSourceServiceStub, id: str
) -> DomainDataSource:
    ret = stub.QueryDomainDataSource(QueryDomainDataSourceRequest(datasource_id=id))

    if ret.status.code != 0:
        raise RuntimeError(f"get_domain_data_source failed for {id}: ret = {ret}")

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


def get_file_from_dp(
    dm_flight_client,
    domain_data_id: str,
    output_file_path: str,
    file_format: DataFileFormat,
    partition_spec: str = "",
):
    domain_data_query = CommandDomainDataQuery(
        domaindata_id=domain_data_id,
        content_type=ContentType.Table,
        partition_spec=partition_spec,
    )
    if file_format == DataFileFormat.CSV or file_format == DataFileFormat.ORC:
        domain_data_query.content_type = ContentType.Table
    elif file_format == DataFileFormat.BINARY:
        domain_data_query.content_type = ContentType.RAW
    else:
        raise AttributeError(f"unknown file_format {file_format}")

    any = Any()
    any.Pack(domain_data_query)

    descriptor = flight.FlightDescriptor.for_command(any.SerializeToString())

    flight_info = dm_flight_client.get_flight_info(
        descriptor=descriptor, options=DEFAULT_FLIGHT_CALL_OPTIONS
    )

    location = flight_info.endpoints[0].locations[0]
    ticket = flight_info.endpoints[0].ticket
    dp_uri = location.uri.decode('utf-8')

    if dp_uri.startswith("kuscia://"):
        dp_flight_client = dm_flight_client
    else:
        dp_flight_client = flight.connect(
            dp_uri, generic_options=DEFAULT_GENERIC_OPTIONS
        )

    flight_reader = dp_flight_client.do_get(ticket=ticket).to_reader()

    if file_format == DataFileFormat.CSV:
        # NOTE(junfeng): use pandas to write csv since pyarrow will add quotes in headers.
        # FIXME: BUG io should running in pyu device context, not in driver context.
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        for batch in flight_reader:
            batch_pd = batch.to_pandas()
            batch_pd.to_csv(
                output_file_path,
                index=False,
                mode='a',
                header=not os.path.exists(output_file_path),
            )
    elif file_format == DataFileFormat.ORC:
        with open(output_file_path, 'wb') as ofile:
            with orc.ORCWriter(
                ofile,
                compression="ZSTD",
                compression_block_size=256 * 1024,
                stripe_size=64 * 1024 * 1024,
            ) as writer:
                for batch in flight_reader:
                    table = pa.Table.from_batches([batch])
                    writer.write(table)

    elif file_format == DataFileFormat.BINARY:
        with open(output_file_path, "wb") as f:
            for batch in flight_reader:
                assert batch.num_columns == 1
                array = batch.column(0)
                assert array.type == pa.binary()
                for r in array:
                    f.write(r.as_py())
    else:
        raise AttributeError(f"unknown file_format {file_format}")

    # close flight client if not kuscia builtin dataserver
    if not dp_uri.startswith("kuscia://"):
        dp_flight_client.close()


def data_file_format2file_format(data_file_format: DataFileFormat) -> FileFormat:
    if data_file_format == DataFileFormat.CSV:
        return FileFormat.CSV
    elif data_file_format == DataFileFormat.ORC:
        return FileFormat.CSV
    elif data_file_format == DataFileFormat.BINARY:
        return FileFormat.BINARY
    else:
        raise AttributeError(f"unknown file_format {data_file_format}")


def create_domain_data_in_dp(
    dm_flight_client, domain_data: DomainData, data_file_format: DataFileFormat
):
    create_domain_data_request = CreateDomainDataRequest(
        # NOTE: rm
        domaindata_id=domain_data.domaindata_id,
        name=domain_data.name,
        type=domain_data.type,
        datasource_id=domain_data.datasource_id,
        relative_uri=domain_data.relative_uri,
        attributes=domain_data.attributes,
        file_format=data_file_format2file_format(data_file_format),
        # partition=data.partition,
        columns=domain_data.columns,
        vendor=domain_data.vendor,
    )

    action = flight.Action(
        "ActionCreateDomainDataRequest",
        create_domain_data_request.SerializeToString(),
    )

    results = dm_flight_client.do_action(
        action=action, options=DEFAULT_FLIGHT_CALL_OPTIONS
    )

    for res in results:
        action_response = CreateDomainDataResponse()
        action_response.ParseFromString(res.body.to_pybytes())
        assert action_response.status.code == 0


def columns_to_schema(columns: list[DataColumn]) -> dict:
    """Converts a list of DataColumn instances into a PyArrow Schema."""
    type_mapping = {
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "float16": pa.float16(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
        "int": pa.int64(),
        "float": pa.float64(),
        "str": pa.string(),
        "string": pa.string(),
    }

    def type_to_arrowtype(type_str: str) -> pa.Field:
        arrow_type = type_mapping.get(type_str.lower(), None)
        if arrow_type is None:
            raise ValueError(f"Unsupported type: {type_str}")
        return arrow_type

    col_dict = {col.name: type_to_arrowtype(col.type) for col in columns}
    return col_dict


def put_file_to_dp(
    dm_flight_client,
    domaindata_id: str,
    file_local_path: str,
    file_format: DataFileFormat,
    data: DomainData,
):
    if file_format == DataFileFormat.CSV:
        command_domain_data_update = CommandDomainDataUpdate(
            domaindata_id=domaindata_id,
            file_write_options=FileWriteOptions(
                csv_options=CSVWriteOptions(field_delimiter=",")
            ),
        )
        # FIXME: BUG io should running in pyu device context, not in driver context.
        col_dict = columns_to_schema(data.columns)
        reader = csv.open_csv(
            file_local_path, convert_options=csv.ConvertOptions(column_types=col_dict)
        )
        schema = reader.schema
    elif file_format == DataFileFormat.ORC:
        command_domain_data_update = CommandDomainDataUpdate(
            domaindata_id=domaindata_id,
            file_write_options=FileWriteOptions(
                csv_options=CSVWriteOptions(field_delimiter=",")
            ),
        )

        def _orc_reader():
            with open(file_local_path, 'rb') as input_file:
                orc_file = orc.ORCFile(input_file)
                nstripes = orc_file.nstripes
                current_stripe = 0
                while current_stripe < nstripes:
                    yield orc_file.read_stripe(current_stripe)
                    current_stripe += 1

        reader = _orc_reader()

        with open(file_local_path, 'rb') as input_file:
            schema = orc.ORCFile(input_file).schema

    elif file_format == DataFileFormat.BINARY:
        command_domain_data_update = CommandDomainDataUpdate(
            domaindata_id=domaindata_id,
            content_type=ContentType.RAW,
        )
        bin_col_name = "binary_data"

        # FIXME: BUG io should running in pyu device context, not in driver context.
        def _bin_reader():
            # 1MB
            read_chunks = 8
            chunks = []
            with open(file_local_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 128), b''):
                    chunks.append(chunk)
                    if len(chunks) >= read_chunks:
                        yield pa.record_batch([pa.array(chunks)], names=[bin_col_name])
                        chunks = []
                if len(chunks):
                    yield pa.record_batch([pa.array(chunks)], names=[bin_col_name])

        reader = _bin_reader()
        schema = pa.schema([(bin_col_name, pa.binary())])
    else:
        raise AttributeError(f"unknown file_format {file_format}")

    any = Any()
    any.Pack(command_domain_data_update)

    descriptor = flight.FlightDescriptor.for_command(any.SerializeToString())

    flight_info = dm_flight_client.get_flight_info(
        descriptor=descriptor, options=DEFAULT_FLIGHT_CALL_OPTIONS
    )

    location = flight_info.endpoints[0].locations[0]
    ticket = flight_info.endpoints[0].ticket
    dp_uri = location.uri.decode('utf-8')
    if dp_uri.startswith("kuscia://"):
        dp_flight_client = dm_flight_client
    else:
        dp_flight_client = flight.connect(
            dp_uri, generic_options=DEFAULT_GENERIC_OPTIONS
        )

    descriptor = flight.FlightDescriptor.for_command(ticket.ticket)
    flight_writer, _ = dp_flight_client.do_put(descriptor=descriptor, schema=schema)

	
    max_transfer_size = 64 * 1024 * 1024
    for batch in reader:
        if batch.nbytes > max_transfer_size:
            rows = batch.num_rows
            slice_cnt = (batch.nbytes + max_transfer_size - 1) // max_transfer_size
            slice_size = (rows + slice_cnt - 1) // slice_cnt
            assert (
                slice_size > 0
            ), f"row size if too big for transfer: {batch.nbytes}, rows: {rows}"
            for offset in range(0, rows, slice_size):
                flight_writer.write(
                    batch.slice(offset, length=min(slice_size, rows - offset))
                )
        else:
            flight_writer.write(batch)

    flight_writer.close()
    reader.close()
    # close flight client if not kuscia builtin dataserver
    if not dp_uri.startswith("kuscia://"):
        dp_flight_client.close()
