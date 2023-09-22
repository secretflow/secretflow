# Copyright 2023 Ant Group Co., Ltd.
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


import abc
from dataclasses import dataclass, fields
from typing import List

from secretflow.device import HEU, PYU
from ..sgb_actor import SGBActor


@dataclass
class Devices:
    label_holder: PYU
    workers: List[PYU]
    heu: HEU


class Component(abc.ABC):
    @abc.abstractmethod
    def show_params(self):
        pass

    @abc.abstractmethod
    def get_params(self, params: dict):
        pass

    @abc.abstractmethod
    def set_params(self, params: dict):
        pass

    @abc.abstractmethod
    def set_devices(self, devices: Devices):
        pass

    @abc.abstractmethod
    def set_actors(self, actors: List[SGBActor]):
        pass

    @abc.abstractmethod
    def del_actors(self):
        pass


class Composite(Component):
    def __init__(self) -> None:
        self.components = None

    def show_params(self):
        for field in fields(self.components):
            print("showing the params of component", field.name)
            getattr(self.components, field.name).show_params()

    def get_params_dict(self, params: dict = {}):
        for field in fields(self.components):
            getattr(self.components, field.name).get_params(params)

    def set_params(self, params: dict):
        for field in fields(self.components):
            getattr(self.components, field.name).set_params(params)

    def set_devices(self, devices: Devices):
        for field in fields(self.components):
            getattr(self.components, field.name).set_devices(devices)

    def set_actors(self, actors: List[SGBActor]):
        for field in fields(self.components):
            getattr(self.components, field.name).set_actors(actors)

    def del_actors(self):
        for field in fields(self.components):
            getattr(self.components, field.name).del_actors()


def print_params(params):
    for field in fields(params):
        print(field.name, getattr(params, field.name))
