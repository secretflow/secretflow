# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import secrets
from concurrent import futures

import grpc
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from secretflowapis.v1 import status_pb2
from secretflowapis.v1.sdc import core_pb2
from secretflowapis.v1.sdc.authmanager import auth_manager_pb2, auth_manager_pb2_grpc
from secretflowapis.v1.sdc.dataagent import data_agent_pb2
from secretflowapis.v1.sdc.teeapps import tee_task_params_pb2


class MockAuthServer(auth_manager_pb2_grpc.AuthManagerServicer):
    def __init__(self):
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._pri_key = self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        self._pub_key = self._key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self._auths = {}

    def CreateDataWithAuth(
        self, request: data_agent_pb2.CreateDataWithAuthRequest, context
    ) -> data_agent_pb2.CreateDataWithAuthResponse:
        data_auth: core_pb2.DataAuth = request.data_auth
        data_meta: core_pb2.DataMeta = request.data_info
        data_part_meta = data_meta.partition_data[0]
        segment_data = data_part_meta.segment_data[0]

        data_key = self._key.decrypt(
            segment_data.encrypted_data_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA1()),
                algorithm=hashes.SHA1(),
                label=None,
            ),
        )

        self._auths[data_auth.data_uuid] = data_key

        return data_agent_pb2.CreateDataWithAuthResponse(
            status=status_pb2.Status(code=0)
        )

    def GetRaCertPems(
        self, request: data_agent_pb2.GetRaCertPemsRequest, context
    ) -> data_agent_pb2.GetRaCertPemsResponse:
        return data_agent_pb2.GetRaCertPemsResponse(
            status=status_pb2.Status(code=0),
            report_with_certs=[
                data_agent_pb2.ReportWithCertPem(cert_pem=self._pub_key)
            ],
        )

    def GetComputeMeta(self, request: auth_manager_pb2.GetComputeMetaRequest, context):
        tee_task_params: tee_task_params_pb2.TeeTaskParams = request.tee_task_params
        data_uuids = [input.data_uuid for input in tee_task_params.inputs]
        data_keys = [self._auths[data_uuid] for data_uuid in data_uuids]
        pub_key_pem = request.public_key.public_key
        input_metas = {}
        for data_uuid, data_key in zip(data_uuids, data_keys):
            part_data_uri = core_pb2.PartitionDataUri(
                partition_id="0",
                seg_data_uris=[core_pb2.SegmentDataUri(data_key=data_key)],
            )
            data_uri_with_dks = core_pb2.DataUri(
                data_uuid=data_uuid, part_data_uris=[part_data_uri]
            )
            input_meta = auth_manager_pb2.ComputeMeta.InputMeta(
                data_uri_with_dks=data_uri_with_dks
            )
            input_metas[data_uuid] = input_meta
        compute_meta = auth_manager_pb2.ComputeMeta(input_metas=input_metas)
        key = secrets.token_bytes(32)
        iv = secrets.token_bytes(12)

        from sdc.crypto.asymm import RsaEncryptor

        compute_meta_secret = RsaEncryptor(pub_key_pem).seal_asymm_secret(
            compute_meta.SerializeToString(), symm_key=key, iv=iv, aad=b"test"
        )

        return auth_manager_pb2.GetComputeMetaResponse(
            status=status_pb2.Status(code=0), encrypted_response=compute_meta_secret
        )

    def RegisterInsPubKey(
        self, request: data_agent_pb2.RegisterInsPubKeyRequest, context
    ):
        return data_agent_pb2.RegisterInsPubKeyResponse(
            status=status_pb2.Status(code=0)
        )


def start_auth_server(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    auth_manager_pb2_grpc.add_AuthManagerServicer_to_server(MockAuthServer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server
