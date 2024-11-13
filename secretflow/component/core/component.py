# Copyright 2024 Ant Group Co., Ltd.
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


import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from typing import TypeVar

from secretflow.spec.v1.data_pb2 import DistData

from .common.types import Input, Output, UnionGroup
from .context import Context
from .serving_builder import ServingBuilder

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:
    from typing_extensions import dataclass_transform


@dataclass_transform()
@dataclass
class Component(ABC):
    """
    Component is the base class of all components.

    It uses the metadata within the field function to describe the fields in the component.
    And it leverages the dataclass decorator to automatically generate methods for these described fields,
    Then we can use __dataclass_fields__ to reflect the Component and convert to ComponentDef

    NOTE:
    you do not need to manually add the dataclass decorator to your component,
    as it has already been automatically applied by the __init_subclass__ method.

    More detail for dataclasses:
    https://peps.python.org/pep-0557/

    When you create a new component, you should define its domain and version explicitly.
    The name of the component can be inferred from the class name it is defined within,
    and the description can be taken from the docstring.
    However, for more control over these values, you can set them manually.

    Examples:
        >>> @register(domain="test", version="0.0.1")
        >>> class DemoComponent(Component):
        >>>     value : int = Field.attr(desc="value", default=1)
        >>>     input0: Input = Field.input(desc="input table", types=[DistDataType.INDIVIDUAL_TABLE])
        >>>     # implement abstractmethod
        >>>     def evaluate(self, ctx: Context):
        >>>         print(self.value)
        >>> # create instance directly
        >>> comp = DemoComponent(value=1)
        >>> assert comp.value == 1
        >>> # use Definition to reflect the component and create instance
        >>> definition = Definition(DemoComponent)
        >>> args = {"_minor":0, "value":1, "input0": DistData()}
        >>> comp : DemoComponent = definition.make_component(args)
        >>> asset comp.value == 1
    """

    _minor: int = field(default=-1, kw_only=True)

    @property
    def minor(self) -> int:
        return self._minor

    def is_supported(self, v: int) -> bool:
        return self._minor > v

    def __init_subclass__(cls, **kwargs):
        def _is_custom_class(cls):
            if not isinstance(cls, type) or cls.__module__ == "builtins":
                return False
            if cls in [DistData, Input, Output, UnionGroup]:
                return False
            return True

        def _check_field_dataclass(cls):
            for f in cls.__dataclass_fields__.values():
                if f.name.startswith("_"):
                    continue
                if not _is_custom_class(f.type):
                    continue
                if not is_dataclass(f.type):
                    f.type = dataclass(f.type)
                _check_field_dataclass(f.type)

        dataclass(cls, kw_only=True)
        _check_field_dataclass(cls)

        super().__init_subclass__(**kwargs)

    @abstractmethod
    def evaluate(self, ctx: Context) -> None:
        raise NotImplementedError(f"{type(self)} evaluate is not implemented.")

    def export(self, ctx: Context, builder: ServingBuilder, **kwargs) -> None:
        raise NotImplementedError(f"{type(self)} export is not implemented.")


# TComponent is used to implement the mixin design pattern.
TComponent = TypeVar('TComponent', bound=Component)
