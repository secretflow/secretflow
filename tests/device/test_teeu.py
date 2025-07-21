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

import os
from sys import platform

import numpy as np
import pytest

import secretflow as sf
from secretflow.device import global_state
from secretflow.device.device.teeu import TEEU
from tests.sf_config import build_prod_cluster_config
from tests.sf_fixtures import ClusterConfig, DeviceInventory, mpc_fixture
from tests.sf_services import SERVICE_AUTH


@mpc_fixture(services=[SERVICE_AUTH])
def sf_production_setup_devices_teeu(
    self_party, parties: list[str], cluster: ClusterConfig, auth_port: int
):
    assert len(parties) == 4, f"parties={parties}"

    party_key_pair = None
    files = []
    if self_party in ('alice', 'bob'):
        import tempfile

        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        _, private_key_path = tempfile.mkstemp()
        _, public_key_path = tempfile.mkstemp()
        files = [private_key_path, public_key_path]

        party_key_pair = {
            self_party: {'public_key': public_key_path, 'private_key': private_key_path}
        }

        def gen_rsa(pub_path: str, pri_path: str):
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            key_contents = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            public_key = key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            with open(pri_path, 'wb') as f:
                f.write(key_contents)
            with open(pub_path, 'wb') as f:
                f.write(public_key)

        gen_rsa(public_key_path, private_key_path)

    cluster_config = build_prod_cluster_config(self_party, cluster.fed_addrs)

    global_state.set_auth_manager_host(f'127.0.0.1:{auth_port}')

    sf.init(
        address='local',
        cluster_config=cluster_config,
        num_cpus=8,
        log_to_driver=True,
        party_key_pair=party_key_pair,
        tee_simulation=True,
        enable_waiting_for_other_parties_ready=False,
    )

    devices = DeviceInventory()
    devices.build_pyus(parties)
    yield devices
    del devices

    for file in files:
        try:
            os.remove(file)
        except:
            pass
    sf.shutdown()


@pytest.mark.skipif(platform == 'darwin', reason="TEEU does not support macOS")
@pytest.mark.mpc(parties=4)
def test_teeu_function_should_ok(sf_production_setup_devices_teeu):
    def average(data):
        return np.average(data, axis=0)

    devices = sf_production_setup_devices_teeu

    teeu = TEEU(party='carol', mr_enclave='')
    d1 = devices.alice(lambda: np.random.random([2, 4]))()
    d2 = devices.bob(lambda: np.random.random([2, 4]))()
    d1_tee = d1.to(teeu, allow_funcs=average)
    d2_tee = d2.to(teeu, allow_funcs=average)
    avg_val = teeu(average)([d1_tee, d2_tee])
    avg_val = sf.reveal(avg_val)
    expected_avg = average(sf.reveal([d1, d2]))
    np.testing.assert_equal(avg_val, expected_avg)


@pytest.mark.skipif(platform == 'darwin', reason="TEEU does not support macOS")
@pytest.mark.mpc(parties=4)
def test_teeu_actor_should_ok(sf_production_setup_devices_teeu):
    class Model:
        def __init__(self, x):
            self.x = x.copy()

        def add(self, data):
            self.x += data
            return self.x

    devices = sf_production_setup_devices_teeu

    teeu = TEEU(party='carol', mr_enclave='')
    d1 = devices.alice(lambda: np.random.random([2, 4]))()
    d2 = devices.bob(lambda: np.random.random([2, 4]))()
    d1_tee = d1.to(teeu, allow_funcs=Model)
    d2_tee = d2.to(teeu, allow_funcs=Model)
    model = teeu(Model)(d1_tee)
    sum_val = model.add(d2_tee)
    sum_val = sf.reveal(sum_val)
    plain_model = Model(sf.reveal(d1))
    expected_sum = plain_model.add(sf.reveal(d2))
    np.testing.assert_equal(sum_val, expected_sum)
