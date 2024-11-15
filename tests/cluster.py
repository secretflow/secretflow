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


SELF_PARTY = None


def set_self_party(party: str):
    global SELF_PARTY
    SELF_PARTY = party


def get_self_party() -> str:
    global SELF_PARTY
    return SELF_PARTY


def get_available_port(start_port: int):
    '''
    Get available port for sf cluster in ci pipeline. However, in circleci pipeline
    there is no need to allocate new port and can just return the input port.
    '''
    port = start_port

    return port


def cluster():
    return {
        'parties': {
            'alice': {'address': f"127.0.0.1:{get_available_port(61001)}"},
            'bob': {'address': f"127.0.0.1:{get_available_port(61250)}"},
            'carol': {'address': f"127.0.0.1:{get_available_port(61500)}"},
            'davy': {'address': f"127.0.0.1:{get_available_port(61750)}"},
        },
        'self_party': get_self_party(),
    }
