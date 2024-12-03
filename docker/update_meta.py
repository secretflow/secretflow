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

import argparse
import importlib
import json
import os
import sys

import translators

from secretflow.component.core import Translator, translate


class MyTranslator(Translator):
    def __init__(self, lang: str, translator: str):
        self._lang = lang
        self._translator = translator

    def translate(self, text):
        return translators.translate_text(
            text,
            from_language='en',
            to_language=self._lang,
            translator=self._translator,
        )


def do_translate(package: str, root_dir: str, ts: Translator):
    root_package_path = os.path.join(root_dir, package)
    try:
        importlib.import_module(f"{package}.component")
    except Exception as e:
        raise ValueError(f"import fail, {package}, {e}, {sys.path}")
    translation_file = os.path.join(root_package_path, "component", "translation.json")
    with open(translation_file, "r") as f:
        archieve = json.load(f)
    trans = translate(package, archieve, ts)
    with open(translation_file, "w") as f:
        json.dump(trans, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update sf component meta.")
    parser.add_argument(
        '-t', '--translator', type=str, required=False, default="alibaba"
    )
    args = parser.parse_args()

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    root_dir = os.path.dirname(current_dir)

    my_ts = MyTranslator("zh", args.translator)
    do_translate("secretflow", root_dir, my_ts)
    do_translate("secretflow_fl", root_dir, my_ts)
