# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

from secretflow.component.core import (
    Component,
    Context,
    IServingExporter,
    ServingBuilder,
    register,
)


@register(domain="preprocessing", version="1.0.0", name="my_component", desc="xx")
class MyComponent(Component, IServingExporter):
    def evaluate(self, ctx: Context) -> None:
        logging.info(f"evaluate my_component")

    def export(self, ctx: Context, builder: ServingBuilder, **kwargs) -> None:
        logging.info(f"export my_component")
