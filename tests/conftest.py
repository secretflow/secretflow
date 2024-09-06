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

import getpass
import json
import logging
import os
import platform
import stat
import subprocess
import tempfile
import threading
import time
import urllib
import uuid
from dataclasses import dataclass

import multiprocess
import psutil
import pytest
import s3fs
import spu
from xdist.scheduler import LoadScheduling

import secretflow as sf
import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig, SFClusterDesc
from secretflow.spec.v1.data_pb2 import StorageConfig
from secretflow.utils.testing import unused_tcp_port
from tests.cluster import PYTEST_CLUSTER, cluster, set_self_party
from tests.load import SF_PARTIES, SF_PARTY_PREFIX, SFLoadPartyScheduling


def pytest_addoption(parser):
    parser.addoption(
        "--env",
        action="store",
        default="sim",
        help="env option: simulation or production",
        choices=("sim", "prod"),
    )


min_available_memory = float('inf')
pytest_start_time = float(0)
pytest_end = False


def log_memory_usage():
    '''
    log memory usage during pytest running every 5 seconds
    '''
    global min_available_memory, pytest_end
    count = 0
    while not pytest_end:
        if count % 5 == 0:
            mem = psutil.virtual_memory()
            min_available_memory = min(min_available_memory, mem.available)
            print(
                f"****** pytest runner: {PYTEST_CLUSTER + 1},",
                f"total memory: {mem.total / (1024 ** 3):.2f} GB,",
                f"available memory: {mem.available / (1024 ** 3):.2f} GB,",
                f"used percentage: {mem.percent}%",
                f"******",
            )
        count += 1
        time.sleep(1)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    '''
    start log_memory_usage at the beginning of pytest
    '''
    global pytest_start_time
    pytest_start_time = time.perf_counter()
    print(f"****** pytest runner: {PYTEST_CLUSTER + 1} started pytest session ******")
    threading.Thread(target=log_memory_usage).start()


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    '''
    log minimal available memory at the end of pytest
    '''
    global pytest_end
    global pytest_start_time
    pytest_end = True
    time.sleep(1)

    pytest_end_time = time.perf_counter()
    elapsed_time = pytest_end_time - pytest_start_time
    print(
        f"****** pytest runner: {PYTEST_CLUSTER + 1} end pytest session,",
        f"ExitCode: {exitstatus}, time elapsed: {elapsed_time:.2f} seconds,",
        f"min available memory during testing: {min_available_memory / (1024 ** 3):.2f} GB ******",
    )
    return exitstatus


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
        "schema": "ou",
        "key_pair": {"generate": {"bit_size": 2048}},
    },
}


def cheetah_cluster():
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
            "protocol": spu.spu_pb2.CHEETAH,
            "field": spu.spu_pb2.FM128,
            "share_max_chunk_size": 1025,
            "enable_pphlo_profile": False,
            "enable_hal_profile": False,
            "enable_pphlo_trace": False,
            "enable_action_trace": False,
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
            "share_max_chunk_size": 10251,
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


def _sf_production_setup_devices_grpc(request, sf_party_for_4pc, ray_mode):
    devices = DeviceInventory()
    set_self_party(sf_party_for_4pc)
    sf.init(
        address="local",
        num_cpus=32,
        log_to_driver=True,
        logging_level='info',
        cluster_config=cluster(),
        enable_waiting_for_other_parties_ready=False,
        ray_mode=ray_mode,
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

    return devices


@pytest.fixture(scope="module")
def sf_production_setup_devices_grpc(request, sf_party_for_4pc):
    devices = _sf_production_setup_devices_grpc(
        request, sf_party_for_4pc, ray_mode=False
    )
    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module")
def sf_production_setup_devices_grpc_ray(request, sf_party_for_4pc):
    devices = _sf_production_setup_devices_grpc(
        request, sf_party_for_4pc, ray_mode=True
    )
    yield devices
    del devices
    sf.shutdown()


def _sf_production_setup_devices(request, sf_party_for_4pc, ray_mode):
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
        ray_mode=ray_mode,
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

    return devices


@pytest.fixture(scope="module")
def sf_production_setup_devices_ray(request, sf_party_for_4pc):
    devices = _sf_production_setup_devices(request, sf_party_for_4pc, ray_mode=True)
    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module")
def sf_production_setup_devices(request, sf_party_for_4pc):
    devices = _sf_production_setup_devices(request, sf_party_for_4pc, ray_mode=False)
    yield devices
    del devices
    sf.shutdown()


def _sf_production_setup_devices_cheetah(request, sf_party_for_4pc, ray_mode):
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
        ray_mode=ray_mode,
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")
    devices.carol = sf.PYU("carol")
    devices.davy = sf.PYU("davy")

    cluster_def = sf.reveal(devices.alice(cheetah_cluster)())

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

    return devices


@pytest.fixture(scope="module")
def sf_production_setup_devices_cheetah_ray(request, sf_party_for_4pc):
    devices = _sf_production_setup_devices_cheetah(
        request, sf_party_for_4pc, ray_mode=True
    )
    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module")
def sf_production_setup_devices_cheetah(request, sf_party_for_4pc):
    devices = _sf_production_setup_devices_cheetah(
        request, sf_party_for_4pc, ray_mode=False
    )
    yield devices
    del devices
    sf.shutdown()


def _sf_production_setup_devices_aby3(request, sf_party_for_4pc, ray_mode):
    devices = DeviceInventory()
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
        ray_mode=ray_mode,
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

    return devices


@pytest.fixture(scope="module")
def sf_production_setup_devices_aby3_ray(request, sf_party_for_4pc):
    devices = _sf_production_setup_devices_aby3(
        request, sf_party_for_4pc, ray_mode=True
    )
    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module")
def sf_production_setup_devices_aby3(request, sf_party_for_4pc):
    devices = _sf_production_setup_devices_aby3(
        request, sf_party_for_4pc, ray_mode=False
    )
    yield devices
    del devices
    sf.shutdown()


TEST_STORAGE_ROOT = os.path.join(tempfile.gettempdir(), getpass.getuser())


def prepare_storage_path(party):
    storage_path = os.path.join(TEST_STORAGE_ROOT, party, str(uuid.uuid4()))
    os.makedirs(storage_path, exist_ok=True)
    return storage_path


def minio_server_is_ready(config):
    s3_client = s3fs.S3FileSystem(
        anon=False,
        key=config.access_key_id,
        secret=config.access_key_secret,
        client_kwargs={'endpoint_url': config.endpoint},
        config_kwargs={
            'signature_version': config.version,
            's3': {"addressing_style": "virtual" if config.virtual_host else "path"},
        },
    )
    try:
        s3_client.ls(config.bucket, detail=False)
        return True
    except Exception as e:
        return False


def setup_minio_server(storage_path, self_party):
    minio_path = os.path.join(storage_path, f"minio_{self_party}")
    os.makedirs(minio_path, exist_ok=True)

    minio_server = os.path.join(minio_path, "minio")
    if not os.path.exists(minio_server) or not os.path.isfile(minio_server):
        system = "linux"
        arch = "amd64"
        if platform.system() == "Darwin":
            system = "darwin"
        if platform.machine() == "arm64" or platform.machine() == "aarch64":
            arch = "arm64"
        urllib.request.urlretrieve(
            f"https://dl.min.io/server/minio/release/{system}-{arch}/minio",
            minio_server,
        )
        st = os.stat(minio_server)
        os.chmod(minio_server, st.st_mode | stat.S_IEXEC)

    minio_data_path = os.path.join(minio_path, "data")
    os.makedirs(
        os.path.join(minio_data_path, "sf-test"),
        exist_ok=True,
    )

    ms_env = os.environ.copy()
    ms_env["MINIO_BROWSER"] = "off"
    ms_env["MINIO_ACCESS_KEY"] = "sf_test_aaa"
    ms_env["MINIO_SECRET_KEY"] = "sf_test_sss"

    ports = {
        "alice": 64122 + PYTEST_CLUSTER,
        "bob": 64244 + PYTEST_CLUSTER,
        "carol": 64366 + PYTEST_CLUSTER,
        "davy": 64488 + PYTEST_CLUSTER,
    }
    endpoint = f"127.0.0.1:{ports[self_party]}"
    ms = subprocess.Popen(
        [minio_server, "server", minio_data_path, "--address", endpoint],
        env=ms_env,
    )

    storage_config = StorageConfig(
        type="s3",
        s3=StorageConfig.S3Config(
            endpoint=f"http://{endpoint}",
            bucket="sf-test",
            prefix="test-prefix",
            access_key_id="sf_test_aaa",
            access_key_secret="sf_test_sss",
            virtual_host=False,
            version="s3v4",
        ),
    )

    while True:
        time.sleep(0.4)
        if minio_server_is_ready(storage_config.s3):
            break

    return ms, storage_config


@pytest.fixture(scope="package")
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
                    f"127.0.0.1:{62000 + PYTEST_CLUSTER}",
                    f"127.0.0.1:{62500 + PYTEST_CLUSTER}",
                    f"127.0.0.1:{63000 + PYTEST_CLUSTER}",
                    f"127.0.0.1:{63500 + PYTEST_CLUSTER}",
                ],
            ),
            spu_configs=[
                SFClusterConfig.SPUConfig(
                    name="spu",
                    parties=["alice", "bob"],
                    addresses=[
                        f"127.0.0.1:{64000 + PYTEST_CLUSTER}",
                        f"127.0.0.1:{64500 + PYTEST_CLUSTER}",
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

    ms, storage_config = setup_minio_server(storage_path, sf_party_for_4pc)

    yield storage_config, sf_config

    ms.kill()
