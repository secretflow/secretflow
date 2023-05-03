from dataclasses import dataclass

import multiprocess
import pytest
import spu
from xdist.scheduler import LoadScheduling

import secretflow as sf
import secretflow.distributed as sfd
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
    if item.get_closest_marker('prod'):
        if item.config.getoption("--env") == 'sim':
            pytest.skip("test requires env in prod")
    else:
        if item.config.getoption("--env") == 'prod':
            pytest.skip("test requires env in sim")


# Fix xdist number to be number of parties for prod.
def pytest_xdist_auto_num_workers(config):
    if config.getoption("--env") == 'prod':
        return len(SF_PARTIES)
    else:
        return 2


# When env is prod, use the customized SFLoadPartyScheduling.
def pytest_xdist_make_scheduler(config, log):
    if config.getoption("--env") == 'prod':
        return SFLoadPartyScheduling(config, log)
    else:
        return LoadScheduling(config, log)


heu_config = {
    'sk_keeper': {'party': 'alice'},
    'evaluators': [
        {'party': 'bob'},
        {'party': 'carol'},
        {'party': 'davy'},
    ],
    # The HEU working mode, choose from PHEU / LHEU / FHEU_ROUGH / FHEU
    'mode': 'PHEU',
    'he_parameters': {
        'schema': 'paillier',
        'key_pair': {'generate': {'bit_size': 2048}},
    },
}


def semi2k_cluster():
    return {
        'nodes': [
            {
                'party': 'alice',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
            {
                'party': 'bob',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
        ],
        'runtime_config': {
            'protocol': spu.spu_pb2.SEMI2K,
            'field': spu.spu_pb2.FM128,
            'enable_pphlo_profile': False,
            'enable_hal_profile': False,
            'enable_pphlo_trace': False,
            'enable_action_trace': False,
        },
    }


def aby3_cluster():
    return {
        'nodes': [
            {
                'party': 'alice',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
            {
                'party': 'bob',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
            {
                'party': 'carol',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
        ],
        'runtime_config': {
            'protocol': spu.spu_pb2.ABY3,
            'field': spu.spu_pb2.FM64,
            'enable_pphlo_profile': False,
            'enable_hal_profile': False,
            'enable_pphlo_trace': False,
            'enable_action_trace': False,
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
def sf_simulation_setup_devices(request):
    devices = DeviceInventory()
    sfd.set_production(False)
    sf.shutdown()
    sf.init(
        ['alice', 'bob', 'carol', 'davy'],
        address='local',
        num_cpus=16,
        log_to_driver=True,
        omp_num_threads=multiprocess.cpu_count(),
    )

    devices.alice = sf.PYU('alice')
    devices.bob = sf.PYU('bob')
    devices.carol = sf.PYU('carol')
    devices.davy = sf.PYU('davy')

    cluster_def = sf.reveal(devices.alice(request.param)())

    devices.spu = sf.SPU(
        cluster_def,
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
        },
    )
    devices.heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])

    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="session", params=SF_PARTIES)
def sf_party_for_4pc(request):
    yield request.param[len(SF_PARTY_PREFIX) :]


@pytest.fixture(scope="module")
def sf_production_setup_devices(request, sf_party_for_4pc):
    devices = DeviceInventory()
    sfd.set_production(True)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address='local',
        num_cpus=8,
        log_to_driver=True,
        cluster_config=cluster(),
        exit_on_failure_cross_silo_sending=True,
        enable_waiting_for_other_parties_ready=False,
    )

    devices.alice = sf.PYU('alice')
    devices.bob = sf.PYU('bob')
    devices.carol = sf.PYU('carol')
    devices.davy = sf.PYU('davy')

    cluster_def = sf.reveal(devices.alice(semi2k_cluster)())

    devices.spu = sf.SPU(
        cluster_def,
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
        },
    )

    cluster_def_2 = sf.reveal(devices.alice(aby3_cluster)())

    devices.spu2 = sf.SPU(
        cluster_def_2,
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
        },
    )

    devices.heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])

    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module")
def sf_production_setup_devices_aby3(request, sf_party_for_4pc):
    devices = DeviceInventory()
    sfd.set_production(True)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address='local',
        num_cpus=8,
        log_to_driver=True,
        cluster_config=cluster(),
        exit_on_failure_cross_silo_sending=True,
        enable_waiting_for_other_parties_ready=False,
    )

    devices.alice = sf.PYU('alice')
    devices.bob = sf.PYU('bob')
    devices.carol = sf.PYU('carol')
    devices.davy = sf.PYU('davy')

    cluster_def = sf.reveal(devices.alice(aby3_cluster)())

    devices.spu = sf.SPU(
        cluster_def,
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
        },
    )

    cluster_def_2 = sf.reveal(devices.alice(aby3_cluster)())

    devices.spu2 = sf.SPU(
        cluster_def_2,
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
        },
    )

    devices.heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])

    yield devices
    del devices
    sf.shutdown()
