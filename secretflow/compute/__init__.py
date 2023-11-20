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

from .tracer import Array, Table

__all__ = ["Array", "Table"]


def __init_sf_func():
    import importlib

    g = globals()
    c = importlib.import_module("secretflow.compute.compute")
    sf_funcs = c._gen_sf_funcs()
    for name, func in sf_funcs.items():
        g[name] = func


__init_sf_func()
