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


class MemCallHolder:
    """
    `MemCallHolder` represents a call node holder when submitting tasks.
    For example,

    f.party("ALICE").remote()
    ~~~~~~~~~~~~~~~~
        ^
        |
    it's a holder.

    """

    def __init__(
        self,
        node_party,
        execute_impl,
        options={},
    ) -> None:
        self._party = None
        self._node_party = node_party
        self._options = options
        self._execute_impl = execute_impl

    def options(self, **options):
        self._options = options
        return self

    def internal_remote(self, *args, **kwargs):
        if not self._node_party:
            raise ValueError("You should specify a party name on the actor.")

        return self._execute_impl(*args, **kwargs)
