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


import enum
import importlib
import importlib.metadata
import importlib.resources
import io
import os


@enum.unique
class ResourceType(enum.Enum):
    TRANSLATION = "translation.json"


class Resources:
    def __init__(self, base_dir: str, files: dict[ResourceType, str] = None):
        self.base_dir = base_dir
        self.files = files

    @staticmethod
    def from_package(root_package: str) -> "Resources":
        sub_dirs = ["", "resources", "component"]
        for d in sub_dirs:
            p = importlib.resources.files(root_package).joinpath(
                d, ResourceType.TRANSLATION.value
            )
            if p.is_file():
                return Resources(d)
        return Resources("")

    def get_file_path(self, res_tpye: ResourceType) -> str:
        if self.files and res_tpye in self.files:
            file = self.files[res_tpye]
        else:
            file = res_tpye.value

        if self.base_dir:
            file = os.path.join(self.base_dir, file)

        return os.path.normpath(file)

    def get_file(
        self, root_package: str, res_type: ResourceType, as_text: bool = True
    ) -> str | bytes | None:
        fullname = self.get_file_path(res_type)
        sub_dir = os.path.dirname(fullname).replace(os.path.sep, ".")
        filename = os.path.basename(fullname)
        sub_package = f"{root_package}.{sub_dir}" if sub_dir else root_package
        if not importlib.resources.is_resource(sub_package, filename):
            return None

        with importlib.resources.open_binary(sub_package, filename) as r:
            if as_text:
                r = io.TextIOWrapper(r)
            return r.read()
