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


from dataclasses import dataclass
from typing import Dict, List


_SELF_PARTY: str = None

_PARTIES: List = None


def set_self_party(self_party: str):
    global _SELF_PARTY
    _SELF_PARTY = self_party


def self_party() -> str:
    global _SELF_PARTY
    return _SELF_PARTY


def parties():
    global _PARTIES
    return _PARTIES


def set_parties(parties: List) -> List:
    global _PARTIES
    _PARTIES = parties


@dataclass
class PartyCert:
    party_name: str = None

    key: str = None
    """Server key in pem.
    """

    cert: str = None
    """Server cert in pem.
    """

    root_ca_cert: str = None
    """Root CA certifcate of clients.
    """


_PARTY_CERTS: Dict[str, PartyCert] = {}
"""A global dict describes party certificate infos.
"""


def set_party_certs(party_certs: Dict[str, PartyCert]):
    global _PARTY_CERTS
    _PARTY_CERTS = party_certs


def party_certs() -> Dict[str, PartyCert]:
    global _PARTY_CERTS
    return _PARTY_CERTS


@dataclass
class PartyKeyPair:
    party_name: str = None

    public_key: str = None
    """The RSA public key in pem.
    """

    private_key: str = None
    """The RSA private key in pem.
    """


_PARTY_KEY_PAIRS: Dict[str, PartyKeyPair] = {}
"""A global dict describes party key pair infos.
"""


def set_party_key_pairs(party_key_pairs: Dict[str, PartyKeyPair]):
    global _PARTY_KEY_PAIRS
    _PARTY_KEY_PAIRS = party_key_pairs


def party_key_pairs() -> Dict[str, PartyKeyPair]:
    global _PARTY_KEY_PAIRS
    return _PARTY_KEY_PAIRS


_TEE_SIMULATION: bool = False
"""Enable tee simulation if True else disable.
"""


def set_tee_simulation(tee_simulation: bool):
    global _TEE_SIMULATION
    _TEE_SIMULATION = tee_simulation


def tee_simulation() -> bool:
    global _TEE_SIMULATION
    return _TEE_SIMULATION


_AUTH_MANAGER_HOST: str = None
_AUTH_MANAGER_MR_ENCLAVE: str = None
_AUHT_MANAGER_CA_CERT: str = None


def set_auth_manager_host(auth_host: str):
    global _AUTH_MANAGER_HOST
    _AUTH_MANAGER_HOST = auth_host


def auth_manager_host() -> str:
    global _AUTH_MANAGER_HOST
    return _AUTH_MANAGER_HOST


def set_auth_manager_mr_enclave(mr_enclave: str):
    global _AUTH_MANAGER_MR_ENCLAVE
    _AUTH_MANAGER_MR_ENCLAVE = mr_enclave


def auth_manager_mr_enclave() -> str:
    global _AUTH_MANAGER_MR_ENCLAVE
    return _AUTH_MANAGER_MR_ENCLAVE


def set_auth_manager_ca_cert(ca_cert: str):
    global _AUHT_MANAGER_CA_CERT
    _AUHT_MANAGER_CA_CERT = ca_cert


def auth_manager_ca_cert() -> str:
    global _AUHT_MANAGER_CA_CERT
    return _AUHT_MANAGER_CA_CERT
