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

import time
from multiprocessing import Process


def add(a, b):
    return a + b


class A:
    def __init__(self, a):
        self._a = a

    def add(self, b, c):
        return self._a + b + c


addresses = {"alice": "127.0.0.1:9963", "bob": "127.0.0.1:9964"}


def run_add(party: str):
    from api import remote, get
    from secretflow.ic.proxy import LinkProxy

    LinkProxy.init(addresses=addresses, self_party=party)

    a = remote(add).party("alice").remote(3, 6)
    b = remote(add).party("bob").remote(a, 10)
    print(f'{party} output: {b.data}')

    time.sleep(1)

    actor = remote(A).party('alice').remote(a)
    c = actor.add.remote(b, 20)
    print(f'{party} output: {c.data}')

    time.sleep(1)
    print(f'{party} get: {get([a, b, c])}')


if __name__ == '__main__':
    p1 = Process(target=run_add, args=("alice",))
    p2 = Process(target=run_add, args=("bob",))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
