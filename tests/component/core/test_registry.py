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

from secretflow.component.core import Registry, get_comp_list_def


def test_registry():
    comp_list_def = get_comp_list_def()
    assert len(comp_list_def.comps) > 0
    definitions = Registry.get_definitions()
    assert len(definitions) == len(comp_list_def.comps)
    first = list(definitions)[0]
    assert Registry.get_definition(first.domain, first.name, first.version)
    assert Registry.get_definition_by_class(first.component_cls)
    assert Registry.get_definition_by_id(first.component_id)
