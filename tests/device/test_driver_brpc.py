import tempfile
import time

import numpy as np
import pytest

import secretflow as sf
import secretflow.distributed as sfd
from secretflow.device.device.spu import SPUObject
from secretflow.device.driver import reveal, to, wait
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from tests.cluster import cluster, set_self_party
from tests.conftest import DeviceInventory, semi2k_cluster


@pytest.fixture(scope="module")
def env(request, sf_party_for_4pc):
    devices = DeviceInventory()
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.PRODUCTION)
    set_self_party(sf_party_for_4pc)
    sf.init(
        address='local',
        num_cpus=8,
        log_to_driver=True,
        cluster_config=cluster(),
        cross_silo_comm_backend='brpc_link',
        cross_silo_comm_options={
            'exit_on_sending_failure': True,
            'http_max_payload_size': 5 * 1024 * 1024,
            'recv_timeout_ms': 1800 * 1000,
        },
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
            "connect_retry_times": 60,
            "connect_retry_interval_ms": 1000,
        },
        id='spu1',
    )

    yield devices
    del devices
    sf.shutdown()


def _test_wait_should_ok(devices):
    # GIVEN
    def simple_write():
        _, temp_file = tempfile.mkstemp()
        time.sleep(5)
        with open(temp_file, 'w') as f:
            f.write('This is alice.')
        return temp_file

    o = devices.alice(simple_write)()

    # WHEN
    wait([o])

    # THEN
    def check(temp_file):
        with open(temp_file, 'r') as f:
            assert f.read() == 'This is alice.'
        return True

    file_path = reveal(o)
    assert reveal(devices.alice(check)(file_path))


def test_wait_should_ok_prod_brpc(env):
    _test_wait_should_ok(env)


def _test_spu_reveal(devices):
    with pytest.raises(ValueError):
        x = to(devices.spu, 32)

    x = to(devices.alice, 32).to(devices.spu)
    assert isinstance(x, SPUObject)

    x_ = reveal(x)
    assert isinstance(x_, np.ndarray)
    assert x_ == 32


def test_spu_reveal_prod_brpc(env):
    _test_spu_reveal(env)


def _test_spu_reveal_empty_list(devices):
    x = to(devices.alice, []).to(devices.spu)
    assert isinstance(x, SPUObject)

    x_ = reveal(x)
    assert x_ == []


def test_spu_reveal_empty_list_prod_brpc(env):
    _test_spu_reveal_empty_list(env)
