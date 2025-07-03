# Copyright 2025 Ant Group Co., Ltd.
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

import json
import os
import socket

from secretflow_spec.v1.data_pb2 import StorageConfig

from secretflow.spec.extend.cluster_pb2 import SFClusterConfig, SFClusterDesc


def is_available_port(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except socket.error:
            return False


def get_available_port(start_port: int) -> int:
    if start_port < 1024:
        start_port = 1024

    port = start_port
    while port < 65535:
        if is_available_port(port):
            return port
        port += 1

    return 0


def generate_port(start_port: int):
    port = start_port
    while True:
        port = get_available_port(port)
        if port == 0:
            raise ValueError(f"cannot find available port from {start_port}.")
        yield port
        port += 1


def generate_port_by_node_index(node_index: int):
    node_index = node_index % 200
    start_port = 61000 + node_index * 20
    return generate_port(start_port)


ALL_PARTIES = ["alice", "bob", "carol", "davy"]


def get_parties(parties: list[str] | int | None) -> list[str]:
    if parties is None:
        return ["alice", "bob"]

    if isinstance(parties, int):
        party_count = parties
        return ALL_PARTIES[:party_count]
    elif isinstance(parties, list):
        assert len(parties) > 0
        return parties
    else:
        raise ValueError(f"invalid parties config: {parties}")


_default_comp_spu_conf = {
    "runtime_config": {
        "protocol": "REF2K",
        "field": "FM64",
    },
    "link_desc": {
        "connect_retry_times": 60,
        "connect_retry_interval_ms": 1000,
        "brpc_channel_protocol": "http",
        "brpc_channel_connection_type": "pooled",
        "recv_timeout_ms": 1200 * 1000,
        "http_timeout_ms": 1200 * 1000,
    },
}
_default_comp_heu_conf = {
    "mode": "PHEU",
    "schema": "ou",
    "key_size": 2048,
}


def build_comp_cluster_config(
    self_party: str,
    parties: list[str],
    fed_addrs: dict,
    spu_addrs: dict,
    inf_addrs: dict,
    *,
    spu_conf: dict = _default_comp_spu_conf,
    heu_conf: dict = _default_comp_heu_conf,
):
    assert parties and fed_addrs and spu_addrs and inf_addrs
    desc = SFClusterDesc(
        parties=parties,
        devices=[
            SFClusterDesc.DeviceDesc(
                name="spu",
                type="spu",
                parties=parties,
                config=json.dumps(spu_conf),
            ),
            SFClusterDesc.DeviceDesc(
                name="heu",
                type="heu",
                parties=[],
                config=json.dumps(heu_conf),
            ),
        ],
    )

    return SFClusterConfig(
        desc=desc,
        public_config=SFClusterConfig.PublicConfig(
            ray_fed_config=SFClusterConfig.RayFedConfig(
                parties=list(fed_addrs.keys()),
                addresses=list(fed_addrs.values()),
            ),
            spu_configs=[
                SFClusterConfig.SPUConfig(
                    name="spu",
                    parties=list(spu_addrs.keys()),
                    addresses=list(spu_addrs.values()),
                )
            ],
            barrier_on_shutdown=True,
            inference_config=SFClusterConfig.InferenceConfig(
                parties=list(inf_addrs.keys()),
                addresses=list(inf_addrs.values()),
            ),
            webhook_config=SFClusterConfig.WebhookConfig(
                progress_url="mock://xxxx",
            ),
        ),
        private_config=SFClusterConfig.PrivateConfig(
            self_party=self_party, ray_head_addr="local"
        ),
    )


def get_storage_root():
    import getpass
    import os
    import tempfile

    return os.path.join(tempfile.gettempdir(), getpass.getuser())


def build_comp_storage_config(
    self_party: str, test_id: str, minio_port: int, storage_type: str
):
    if storage_type == "local_fs":
        storage_root = get_storage_root()
        storage_path = os.path.join(storage_root, self_party, test_id)
        return StorageConfig(
            type="local_fs", local_fs=StorageConfig.LocalFSConfig(wd=storage_path)
        )

    s3_conf = build_s3_config(minio_port)

    bucket = s3_conf["bucket"]
    prefix = s3_conf["prefix"]
    access_key = s3_conf["access_key"]
    access_secret = s3_conf["access_secret"]
    version = s3_conf.get("version", "s3v4")
    endpoint_url = s3_conf["endpoint_url"]
    virtual_host = s3_conf.get("virtual_host", False)
    return StorageConfig(
        type="s3",
        s3=StorageConfig.S3Config(
            endpoint=endpoint_url,
            bucket=bucket,
            prefix=f"{prefix}/{self_party}/{test_id}",
            access_key_id=access_key,
            access_key_secret=access_secret,
            virtual_host=virtual_host,
            version=version,
        ),
    )


def build_s3_config(port: int) -> dict:
    return {
        "endpoint_url": f"http://127.0.0.1:{port}",
        "bucket": "sf-test",
        "prefix": "test-prefix",
        "access_key": "sf_test_aaa",
        "access_secret": "sf_test_sss",
        "virtual_host": False,
        "version": "s3v4",
    }


SPU_PROTOCOL_SEMI2K = "SEMI2K"
SPU_PROTOCOL_ABY3 = "ABY3"
SPU_PROTOCOL_CHEETAH = "CHEETAH"

SPU_FIELD_FM64 = "FM64"
SPU_FIELD_FM128 = "FM128"

_spu_semi2k_conf = {
    "protocol": SPU_PROTOCOL_SEMI2K,
    "field": SPU_FIELD_FM128,
    "share_max_chunk_size": 10251,
    "enable_pphlo_profile": False,
    "enable_hal_profile": False,
    "enable_pphlo_trace": False,
    "enable_action_trace": False,
}
_spu_aby3_conf = {
    "protocol": SPU_PROTOCOL_ABY3,
    "field": SPU_FIELD_FM64,
    "enable_pphlo_profile": False,
    "enable_hal_profile": False,
    "enable_pphlo_trace": False,
    "enable_action_trace": False,
}
_spu_cheetah_conf = {
    "protocol": SPU_PROTOCOL_CHEETAH,
    "field": SPU_FIELD_FM128,
    "share_max_chunk_size": 1025,
    "enable_pphlo_profile": False,
    "enable_hal_profile": False,
    "enable_pphlo_trace": False,
    "enable_action_trace": False,
}


def get_spu_runtime_config(protocol: str) -> dict:
    match protocol:
        case "semi2k":
            return _spu_semi2k_conf
        case "aby3":
            return _spu_aby3_conf
        case "cheetah":
            return _spu_cheetah_conf
        case _:
            raise ValueError(f"unsupported spu protocol. {protocol}")


def build_spu_cluster_config(protocol: str, addrs: dict):
    return {
        "nodes": [{"party": party, "address": addr} for party, addr in addrs.items()],
        "runtime_config": get_spu_runtime_config(protocol),
    }


def build_heu_config(sk_keeper: str, evaluators: list[str], heu_conf: dict) -> dict:
    """
    default config:
    {"mode": "PHEU", "he_parameters": {"schema": "ou", "key_pair": {"generate": {"bit_size": 2048}}}}
    """
    if heu_conf is None:
        heu_conf = {}
    return {
        "sk_keeper": {"party": sk_keeper},
        "evaluators": [{"party": party} for party in evaluators],
        "mode": heu_conf.get("mode", "PHEU"),
        "he_parameters": {
            "schema": heu_conf.get("schema", "ou"),
            "key_pair": heu_conf.get("key_pair", {"generate": {"bit_size": 2048}}),
        },
    }


def build_prod_cluster_config(self_party: str, fed_addrs: dict) -> dict:
    return {
        "self_party": self_party,
        "parties": {party: {"address": addr} for party, addr in fed_addrs.items()},
    }
