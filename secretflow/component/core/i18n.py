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

import abc
import json

from secretflow_spec import Definition, Registry
from secretflow_spec.v1.component_pb2 import CompListDef

from .plugin import PluginManager
from .resources import ResourceType
from .utils import _build_comp_list_def


class Translator(abc.ABC):
    @abc.abstractmethod
    def translate(self, text: str) -> str: ...


class PlainTranslator(Translator):
    def translate(self, text):
        return text


# The format of the gettext output:
# {
#     'comp': {
#         'text': 'archieved translation or empty'
#         '...':'...'
#     },
#     '...':{}
# }
def gettext(
    comp_list: CompListDef,
    archives: dict = None,
    translator: Translator = None,
    build_root: bool = False,
):
    def _trim_version(key: str) -> str:
        if ":" not in key:
            # ROOT key
            return key
        prefix, version = key.split(':')
        tokens = version.split('.')
        assert len(tokens) == 3, f"invalid version {version} in {key}"
        return f"{prefix}:{tokens[0]}"

    if archives:
        # ignore major version
        archives = {_trim_version(k): v for k, v in archives.items()}

    def restore_from_archives(text: dict, key: str, archives=None):
        archive_key = _trim_version(key)
        if archives is None or archive_key not in archives:
            return text

        archive = archives[archive_key]
        for k in text.keys():
            if k in archive:
                text[k] = archive[k]

        return text

    ret = {}

    if build_root:
        ROOT = "."
        root_text = {comp_list.name: "", comp_list.desc: ""}
        ret[ROOT] = restore_from_archives(root_text, ROOT, archives)

    for comp in comp_list.comps:
        text = {}
        text[comp.domain] = ""
        text[comp.name] = ""
        text[comp.desc] = ""
        text[comp.version] = ""

        for attr in comp.attrs:
            text[attr.name] = ""
            text[attr.desc] = ""

        for io in list(comp.inputs) + list(comp.outputs):
            text[io.name] = ""
            text[io.desc] = ""

            for t_attr in io.attrs:
                text[t_attr.name] = ""
                text[t_attr.desc] = ""
                for t_attr_a in t_attr.extra_attrs:
                    text[t_attr_a.name] = ""
                    text[t_attr_a.desc] = ""

        key = Definition.build_id(comp.domain, comp.name, comp.version)
        text = restore_from_archives(text, key, archives)
        if translator is not None:
            for k, v in text.items():
                if v == "":
                    text[k] = translator.translate(k)

        ret[key] = text

    return ret


def translate(package: str, archives: dict | None, ts: Translator = None) -> dict:
    '''
    find components by root package, and generate translation
    '''
    if ts is None:
        ts = PlainTranslator()

    definitions = Registry.get_definitions(package)
    if not definitions:
        raise ValueError(f"cannot find components by package<{package}>")
    comp_list_def = _build_comp_list_def(definitions)
    build_root = package in ["*", "secretflow"]
    return gettext(comp_list_def, archives, ts, build_root)


def get_translation() -> dict:
    '''
    load translation.json from all plugins and merge to one
    '''
    plugins = PluginManager.get_plugins()

    archives = {}
    for p in reversed(plugins.values()):
        data = p.resources.get_file(p.package, ResourceType.TRANSLATION)
        if data is None:
            continue
        v = json.loads(data)
        if len(archives) > 0:
            archives.update(v)
        else:
            archives = v

    return archives
