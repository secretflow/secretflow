import getpass
import json
import logging
import os
import tempfile
from dataclasses import dataclass

import multiprocess
import pytest
import spu
from xdist.scheduler import LoadScheduling

import secretflow as sf
import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig, SFClusterDesc
from secretflow.spec.v1.data_pb2 import StorageConfig
from secretflow.utils.testing import unused_tcp_port
from tests.cluster import cluster, set_self_party
from tests.load import SF_PARTIES, SF_PARTY_PREFIX, SFLoadPartyScheduling


def pytest_addoption(parser):
    parser.addoption(
        "--env",
        action="store",
        default="sim",
        help="env option: simulation or production",
        choices=("sim", "prod"),
    )


FIXTURES_FOR_PROD = ["sf_party_for_4pc"]


# if tests are using any fixtures from FIXTURES_FOR_PROD,
# mark them as 'prod' tests
def pytest_collection_modifyitems(items):
    for item in items:
        print(f"Fixture names for {item.name}: {item.fixturenames}")
    for item in items:
        find_fixtures_for_prod = False
        for name in item.fixturenames:
            if name in FIXTURES_FOR_PROD:
                find_fixtures_for_prod = True
                break

        if find_fixtures_for_prod:
            item.add_marker(pytest.mark.prod)


# Run prod tests when env is prod only.
# And run non-prod tests when env is sim only.
def pytest_runtest_setup(item):
    if item.get_closest_marker("prod"):
        if item.config.getoption("--env") == "sim":
            pytest.skip("test requires env in prod")
    else:
        if item.config.getoption("--env") == "prod":
            pytest.skip("test requires env in sim")


# Fix xdist number to be number of parties for prod.
def pytest_xdist_auto_num_workers(config):
    if config.getoption("--env") == "prod":
        return len(SF_PARTIES)
    else:
        return 2


# When env is prod, use the customized SFLoadPartyScheduling.
def pytest_xdist_make_scheduler(config, log):
    if config.getoption("--env") == "prod":
        return SFLoadPartyScheduling(config, log)
    else:
        return LoadScheduling(config, log)


heu_config = {
    "sk_keeper": {"party": "alice"},
    "evaluators": [
        {"party": "bob"},
        {"party": "carol"},
        {"party": "davy"},
    ],
    # The HEU working mode, choose from PHEU / LHEU / FHEU_ROUGH / FHEU
    "mode": "PHEU",
    "he_parameters": {
        "schema": "paillier",
        "key_pair": {"generate": {"bit_size": 2048}},
    },
}


def semi2k_cluster():
    return {
        "nodes": [
            {
                "party": "alice",
                "address": f"127.0.0.1:{unused_tcp_port()}",
            },
            {
                "party": "bob",
                "address": f"127.0.0.1:{unused_tcp_port()}",
            },
        ],
        "runtime_config": {
            "protocol": spu.spu_pb2.SEMI2K,
            "field": spu.spu_pb2.FM128,
            "share_max_chunk_size": 1025,
            "enable_pphlo_profile": False,
            "enable_hal_profile": False,
            "enable_pphlo_trace": False,
            "enable_action_trace": False,
        },
    }


def aby3_cluster():
    return {
        "nodes": [
            {
                "party": "alice",
                "address": f"127.0.0.1:{unused_tcp_port()}",
            },
            {
                "party": "bob",
                "address": f"127.0.0.1:{unused_tcp_port()}",
            },
            {
                "party": "carol",
                "address": f"127.0.0.1:{unused_tcp_port()}",
            },
        ],
        "runtime_config": {
            "protocol": spu.spu_pb2.ABY3,
            "field": spu.spu_pb2.FM64,
            "enable_pphlo_profile": False,
            "enable_hal_profile": False,
            "enable_pphlo_trace": False,
            "enable_action_trace": False,
        },
    }


@dataclass
class DeviceInventory:
    alice: sf.PYU = None
    bob: sf.PYU = None
    carol: sf.PYU = None
    davy: sf.PYU = None
    spu: sf.SPU = None  # 2pc
    spu2: sf.SPU = None  # 3pc


@pytest.fixture(scope="module", params=[semi2k_cluster])
def sf_memory_setup_devices(request):
    devices = DeviceInventory()
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.DEBUG)
    sf.shutdown()
    sf.init(
        ["alice", "bob", "carol", "davy", "spu"],
        debug_mode=True,
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")
    devices.carol = sf.PYU("carol")
    devices.davy = sf.PYU("davy")

    logging.warning(f"WARNING:The spu device is actually the pyu device in debug mode")
    devices.spu = sf.PYU("spu")
    cluster_def = sf.reveal(devices.alice(request.param)())

    devices.heu = sf.HEU(heu_config, cluster_def["runtime_config"]["field"])

    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module", params=[semi2k_cluster])
def sf_simulation_setup_devices(request):
    devices = DeviceInventory()
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.SIMULATION)
    sf.shutdown()
    sf.init(
        ["alice", "bob", "carol", "davy"],
        address="local",
        num_cpus=32,
        log_to_driver=True,
        omp_num_threads=multiprocess.cpu_count(),
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")
    devices.carol = sf.PYU("carol")
    devices.davy = sf.PYU("davy")

    cluster_def = sf.reveal(devices.alice(request.param)())

    devices.spu = sf.SPU(
        cluster_def,
        link_desc={
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
        },
        id='spu',
    )
    devices.heu = sf.HEU(heu_config, cluster_def["runtime_config"]["field"])

    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="session", params=SF_PARTIES)
def sf_party_for_4pc(request):
    yield request.param[len(SF_PARTY_PREFIX) :]


@pytest.fixture(scope="module")
def sf_production_setup_devices_grpc(request, sf_party_for_4pc):
    devices = DeviceInventory()
    sfd.set_distribution_mode(DISTRIBUTION_MODE.PRODUCTION)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address="local",
        num_cpus=32,
        log_to_driver=True,
        logging_level='info',
        cluster_config=cluster(),
        enable_waiting_for_other_parties_ready=False,
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")
    devices.carol = sf.PYU("carol")
    devices.davy = sf.PYU("davy")

    cluster_def = sf.reveal(devices.alice(semi2k_cluster)())

    devices.spu = sf.SPU(
        cluster_def,
        link_desc={
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
        },
        id='spu1',
    )

    cluster_def_2 = sf.reveal(devices.alice(aby3_cluster)())

    devices.spu2 = sf.SPU(
        cluster_def_2,
        link_desc={
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
        },
        id='spu2',
    )

    devices.heu = sf.HEU(heu_config, cluster_def["runtime_config"]["field"])

    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module")
def sf_production_setup_devices(request, sf_party_for_4pc):
    devices = DeviceInventory()
    sfd.set_distribution_mode(DISTRIBUTION_MODE.PRODUCTION)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address="local",
        num_cpus=32,
        log_to_driver=True,
        logging_level='info',
        cluster_config=cluster(),
        enable_waiting_for_other_parties_ready=False,
        cross_silo_comm_backend="brpc_link",
        cross_silo_comm_options={
            'proxy_max_restarts': 3,
            'timeout_in_ms': 300 * 1000,
            'recv_timeout_ms': 3600 * 1000,
            'connect_retry_times': 3600,
            'connect_retry_interval_ms': 1000,
            'brpc_channel_protocol': 'http',
            'brpc_channel_connection_type': 'pooled',
        },
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")
    devices.carol = sf.PYU("carol")
    devices.davy = sf.PYU("davy")

    cluster_def = sf.reveal(devices.alice(semi2k_cluster)())

    devices.spu = sf.SPU(
        cluster_def,
        link_desc={
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
            'recv_timeout_ms': 600 * 1000,
        },
        id='spu1',
    )

    cluster_def_2 = sf.reveal(devices.alice(aby3_cluster)())

    devices.spu2 = sf.SPU(
        cluster_def_2,
        link_desc={
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
            "recv_timeout_ms": 120000,
        },
        id='spu2',
    )

    devices.heu = sf.HEU(heu_config, cluster_def["runtime_config"]["field"])

    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module")
def sf_production_setup_devices_aby3(request, sf_party_for_4pc):
    devices = DeviceInventory()
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.PRODUCTION)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address="local",
        num_cpus=32,
        log_to_driver=True,
        logging_level='info',
        cluster_config=cluster(),
        enable_waiting_for_other_parties_ready=False,
        cross_silo_comm_backend="brpc_link",
        cross_silo_comm_options={
            'proxy_max_restarts': 3,
            'timeout_in_ms': 300 * 1000,
            'recv_timeout_ms': 3600 * 1000,
            'connect_retry_times': 3600,
            'connect_retry_interval_ms': 1000,
            'brpc_channel_protocol': 'http',
            'brpc_channel_connection_type': 'pooled',
        },
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")
    devices.carol = sf.PYU("carol")
    devices.davy = sf.PYU("davy")

    cluster_def = sf.reveal(devices.alice(aby3_cluster)())

    devices.spu = sf.SPU(
        cluster_def,
        link_desc={
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
            "brpc_channel_protocol": "http",
            "brpc_channel_connection_type": "pooled",
            "recv_timeout_ms": 2000 * 1000,
            "http_timeout_ms": 2000 * 1000,
        },
        id='spu1',
    )

    cluster_def_2 = sf.reveal(devices.alice(aby3_cluster)())

    devices.spu2 = sf.SPU(
        cluster_def_2,
        link_desc={
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
        },
        id='spu2',
    )

    devices.heu = sf.HEU(heu_config, cluster_def["runtime_config"]["field"])

    yield devices
    del devices
    sf.shutdown()


TEST_STORAGE_ROOT = os.path.join(tempfile.gettempdir(), getpass.getuser())


def prepare_storage_path(party):
    storage_path = os.path.join(TEST_STORAGE_ROOT, party)
    os.makedirs(storage_path, exist_ok=True)
    return storage_path


@pytest.fixture(scope="module")
def comp_prod_sf_cluster_config(request, sf_party_for_4pc):
    desc = SFClusterDesc(
        parties=["alice", "bob", "carol", "davy"],
        devices=[
            SFClusterDesc.DeviceDesc(
                name="spu",
                type="spu",
                parties=["alice", "bob"],
                config=json.dumps(
                    {
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
                ),
            ),
            SFClusterDesc.DeviceDesc(
                name="heu",
                type="heu",
                parties=[],
                config=json.dumps(
                    {
                        "mode": "PHEU",
                        "schema": "ou",
                        "key_size": 2048,
                    }
                ),
            ),
        ],
    )

    storage_path = prepare_storage_path(sf_party_for_4pc)
    sf_config = SFClusterConfig(
        desc=desc,
        public_config=SFClusterConfig.PublicConfig(
            ray_fed_config=SFClusterConfig.RayFedConfig(
                parties=["alice", "bob", "carol", "davy"],
                addresses=[
                    "127.0.0.1:61041",
                    "127.0.0.1:61042",
                    "127.0.0.1:61043",
                    "127.0.0.1:61044",
                ],
            ),
            spu_configs=[
                SFClusterConfig.SPUConfig(
                    name="spu",
                    parties=["alice", "bob"],
                    addresses=[
                        "127.0.0.1:61045",
                        "127.0.0.1:61046",
                    ],
                )
            ],
            barrier_on_shutdown=True,
        ),
        private_config=SFClusterConfig.PrivateConfig(
            self_party=sf_party_for_4pc,
            ray_head_addr="local",
        ),
    )

    storage_config = StorageConfig(
        type="local_fs",
        local_fs=StorageConfig.LocalFSConfig(wd=storage_path),
    )

    yield storage_config, sf_config
