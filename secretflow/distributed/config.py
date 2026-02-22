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

import pathlib
from typing import Dict

from secretflow.device import global_state
from secretflow.utils.errors import InvalidArgumentError


def get_cluster_config(cluster_config: Dict):
    if not cluster_config:
        raise InvalidArgumentError(
            "Must provide `cluster_config` when running with production mode."
            " Or if you want to run SecretFlow in simulation mode, you should"
            " provide `parties` and keep `cluster_config` with `None`."
        )
    if "self_party" not in cluster_config:
        raise InvalidArgumentError("Miss self_party in cluster config.")
    if "parties" not in cluster_config:
        raise InvalidArgumentError("Miss parties in cluster config.")
    self_party = cluster_config["self_party"]
    all_parties: Dict = cluster_config["parties"]
    if self_party not in all_parties:
        raise InvalidArgumentError(
            f"Party {self_party} not found in cluster config parties."
        )
    for party in all_parties.values():
        assert (
            "address" in party
        ), f"There is no address for party {party} in cluster config."
    return self_party, all_parties


def parse_tls_config(
    tls_config: Dict[str, str], party: str
) -> Dict[str, global_state.PartyCert]:
    party_certs = {}
    if set(tls_config) != set(("cert", "key", "ca_cert")):
        raise InvalidArgumentError(
            "You should only provide cert, key and ca_cert in tls config."
        )
    key_path = pathlib.Path(tls_config["key"])
    cert_path = pathlib.Path(tls_config["cert"])
    root_cert_path = pathlib.Path(tls_config["ca_cert"])

    if not key_path.exists():
        raise InvalidArgumentError(f"Private key file {key_path} does not exist!")
    if not cert_path.exists():
        raise InvalidArgumentError(f"Cert file {cert_path} does not exist!")
    if not root_cert_path.exists():
        raise InvalidArgumentError(f"CA cert file {root_cert_path} does not exist!")
    party_cert = global_state.PartyCert(
        party_name=party,
        key=key_path.read_text(),
        cert=cert_path.read_text(),
        root_ca_cert=root_cert_path.read_text(),
    )
    party_certs[party] = party_cert
    global_state.set_party_certs(party_certs=party_certs)
