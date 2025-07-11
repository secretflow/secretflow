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

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import multiprocess
import spu

import secretflow as sf
from tests.sf_services import SERVICE_MINIO

from .sf_config import (
    SPU_FIELD_FM64,
    SPU_FIELD_FM128,
    build_comp_cluster_config,
    build_comp_storage_config,
    build_heu_config,
    build_prod_cluster_config,
    build_spu_cluster_config,
    generate_port_by_node_index,
    get_parties,
)


class MPCFixture:
    def __init__(self, func: Callable, services: list = None):
        self.func = func
        self.services = services


# Support isolation at the module level
_mpc_fixtures: dict[str, dict[str, MPCFixture]] = defaultdict(dict)


def is_mpc_fixture(name: str):
    assert isinstance(name, str), f"invalid mpc type, {type(name)},{name}"
    return name in _mpc_fixtures


def get_mpc_fixture(name: str, module: str = "") -> MPCFixture:
    module_fixs = _mpc_fixtures.get(name)
    if module_fixs is None:
        return None
    if len(module_fixs) == 1:
        return next(iter(module_fixs.values()))

    return module_fixs.get(module)


def mpc_fixture(fn: Callable = None, services: list[str] = None, alias: str = ""):
    """
    register mpc fixture
    """

    def wrapper(func: Callable):
        _mpc_fixtures[func.__name__][func.__module__] = MPCFixture(func, services)

        use_name = alias if alias else func.__name__
        func.__alias__ = use_name
        if alias != func.__name__:
            # None as placeholder, only used to support polymorphism
            assert _mpc_fixtures.get(alias, None) is None
            _mpc_fixtures[alias] = None

        return func

    if fn is None:
        return wrapper

    return wrapper(fn)


@dataclass
class ClusterConfig:
    fed_addrs: dict[str, str]
    spu_addrs: dict[str, str]
    inf_addrs: dict[str, str]  # inference addrs


def build_cluster_config(parties: list[str], node_index: int) -> ClusterConfig:
    port_gen = generate_port_by_node_index(node_index)

    fed_addrs, spu_addrs, inf_addrs = {}, {}, {}
    for idx, party in enumerate(parties):
        fed_addrs[party] = f"127.0.0.1:{next(port_gen)}"
        spu_addrs[party] = f"127.0.0.1:{next(port_gen)}"
        inf_addrs[party] = f"127.0.0.1:{next(port_gen)}"

    return ClusterConfig(fed_addrs=fed_addrs, spu_addrs=spu_addrs, inf_addrs=inf_addrs)


@dataclass
class DeviceInventory:
    alice: sf.PYU = None
    bob: sf.PYU = None
    carol: sf.PYU = None
    davy: sf.PYU = None
    spu: sf.SPU = None
    heu: sf.HEU = None

    def build_pyus(self, parties: list[str]):
        pyus = {p: sf.PYU(p) for p in parties}
        self.alice = pyus.get("alice")
        self.bob = pyus.get("bob")
        self.carol = pyus.get("carol")
        self.davy = pyus.get("davy")

    def build_spu(self, protocol: str, addrs: dict[str, str]):
        cluster_conf = build_spu_cluster_config(protocol, addrs)
        lind_desc = {"connect_retry_times": 60, "connect_retry_interval_ms": 1000}
        spu = sf.SPU(cluster_conf, link_desc=lind_desc, id="spu")
        self.spu = spu

    def build_heu(
        self,
        parties: list[str],
        heu_sk_keeper: str,
        heu_config: dict = None,
        heu_field=None,
    ):
        if heu_field is None:
            heu_field = spu.FieldType.FM128
        heu_evaluators = [party for party in parties if party != heu_sk_keeper]
        heu_config = build_heu_config(heu_sk_keeper, heu_evaluators, heu_config)
        self.heu = sf.HEU(heu_config, heu_field)


def build_devices(
    parties: list[str],
    spu_protocol: str,
    spu_addrs: dict,
    heu_sk_keeper: str,
    heu_config: dict = None,
    heu_field=None,
) -> DeviceInventory:
    devices = DeviceInventory()
    devices.build_pyus(parties)

    if spu_protocol:
        devices.build_spu(spu_protocol, spu_addrs)

    if heu_sk_keeper:
        devices.build_heu(parties, heu_sk_keeper, heu_config, heu_field)

    return devices


_default_brpc_options = {
    'proxy_max_restarts': 3,
    'timeout_in_ms': 300 * 1000,
    'recv_timeout_ms': 3600 * 1000,
    'connect_retry_times': 3600,
    'connect_retry_interval_ms': 1000,
    'brpc_channel_protocol': 'http',
    'brpc_channel_connection_type': 'pooled',
}


class SFProdParams:
    """
    sf_production_setup_devices params template
    """

    ABY3 = {"spu_protocol": "aby3"}
    GRPC_RAY = {"cross_silo_comm_backend": "grpc", "ray_mode": True}


@mpc_fixture()
def sf_production_setup_devices(
    self_party: str,
    parties: list[str],
    cluster: ClusterConfig,
    cross_silo_comm_backend: str = "grpc",
    cross_silo_comm_options: dict = None,
    spu_protocol: str = "semi2k",  # semi2k/aby3/cheetah
    spu_parties: list[str] | int = None,  # By default equal to parties
    heu_sk_keeper: str = "alice",
    heu_config: dict = None,
    heu_field=None,
    ray_mode=False,
):
    cluster_config = build_prod_cluster_config(self_party, cluster.fed_addrs)

    if cross_silo_comm_options is None:
        if cross_silo_comm_backend == "brpc_link":
            cross_silo_comm_options = _default_brpc_options

    sf.init(
        address="local",
        num_cpus=32,
        log_to_driver=True,
        logging_level='info',
        cluster_config=cluster_config,
        enable_waiting_for_other_parties_ready=False,
        cross_silo_comm_backend=cross_silo_comm_backend,
        cross_silo_comm_options=cross_silo_comm_options,
        ray_mode=ray_mode,
    )

    if spu_parties is None:
        spu_parties = get_parties(3) if spu_protocol == "aby3" else parties
    else:
        spu_parties = get_parties(spu_parties)

    spu_addrs = {k: v for k, v in cluster.spu_addrs.items() if k in spu_parties}

    if heu_field is None:
        heu_field = SPU_FIELD_FM64 if spu_protocol == "aby3" else SPU_FIELD_FM128
    devices = build_devices(
        parties, spu_protocol, spu_addrs, heu_sk_keeper, heu_config, heu_field
    )
    yield devices
    del devices

    sf.shutdown()


@mpc_fixture(services=[SERVICE_MINIO])
def sf_production_setup_comp(
    self_party: str,
    parties: list[str],
    cluster: ClusterConfig,
    testid: str,
    minio_port: int,
    # custom param
    mode: str = "prod",
):
    import os

    assert (
        testid and self_party and parties and minio_port
    ), f"testid={testid}, self_party={self_party}, parties={parties}, minio_port={minio_port}"

    os.environ["SF_UT_DO_NOT_EXIT_ENV_FLAG"] = "1"

    if mode == "simulation":
        storage_type = "local_fs"
        simulation = True
    else:
        storage_type = "s3"
        simulation = False

    storage_conf = build_comp_storage_config(
        self_party, testid, minio_port, storage_type
    )

    if simulation:
        sf.init(
            parties,
            address="local",
            num_cpus=32,
            log_to_driver=True,
            omp_num_threads=multiprocess.cpu_count(),
        )
        yield storage_conf, None
    else:
        cluster_conf = build_comp_cluster_config(
            self_party,
            parties,
            fed_addrs=cluster.fed_addrs,
            spu_addrs=cluster.spu_addrs,
            inf_addrs=cluster.inf_addrs,
        )
        yield storage_conf, cluster_conf

    if simulation:
        sf.shutdown()

    del os.environ["SF_UT_DO_NOT_EXIT_ENV_FLAG"]
