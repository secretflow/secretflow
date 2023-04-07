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


class LevelCache:
    def __init__(self, pyu=None):
        self.level_nodes_GH = []
        self.cache = None
        self.pyu = pyu

    def reset_level_nodes_GH(self):
        self.level_nodes_GH = []

    def collect_level_node_GH(self, child_GHL, idx, is_left):
        """collect one level node GH
        Args:
            child_GHL (PYUObject or HEUObject): PYUObject if self.pyu is not None.
            is_left (bool): whether this node is left child.
        """
        if is_left:
            self.level_nodes_GH.append(child_GHL)
            if self.cache is not None:
                if self.pyu is None:
                    cache = self.cache[idx] - child_GHL
                else:
                    cache = self.pyu(lambda x, y: x - y)(self.cache[idx], child_GHL)
                self.level_nodes_GH.append(cache)
        else:
            # right can only happens if not first level. i.e. cache must exist
            if self.pyu is None:
                cache = self.cache[idx] - child_GHL
            else:
                cache = self.pyu(lambda x, y: x - y)(self.cache[idx], child_GHL)
            self.level_nodes_GH.append(cache)
            self.level_nodes_GH.append(child_GHL)

    def update_level_cache(self, is_last_level):
        if not is_last_level:
            self.cache = self.level_nodes_GH
        elif self.cache:
            self.cache = None

    def get_level_nodes_GH(self):
        return self.level_nodes_GH
