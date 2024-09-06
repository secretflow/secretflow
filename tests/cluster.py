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

SELF_PARTY = None

PYTEST_CLUSTER = int(os.getenv("PYTEST_CLUSTER", 0))


def set_self_party(party: str):
    global SELF_PARTY
    SELF_PARTY = party


def get_self_party() -> str:
    global SELF_PARTY
    return SELF_PARTY


_parties = {
    'alice': {'address': f"127.0.0.1:{61001 + PYTEST_CLUSTER}"},
    'bob': {'address': f"127.0.0.1:{61250 + PYTEST_CLUSTER}"},
    'carol': {'address': f"127.0.0.1:{61500 + PYTEST_CLUSTER}"},
    'davy': {'address': f"127.0.0.1:{61750 + PYTEST_CLUSTER}"},
}


def cluster():
    return {
        'parties': _parties,
        'self_party': get_self_party(),
    }
