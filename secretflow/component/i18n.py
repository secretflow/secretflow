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

# 'i18n' aka internationalization
#  Any tools related to internationalization, translation, localization reside here.

import json

import click

from secretflow.component.entry import COMP_LIST, gen_key
from secretflow.spec.v1.component_pb2 import CompListDef

ROOT = "."


# The format of the gettext output:
# {
#     'comp': {
#         'text': 'archieved translation or empty'
#         '...':'...'
#     },
#     '...':{}
# }
def gettext(comp_list: CompListDef, archives=None):
    def restore_from_archives(text, key, archives=None):
        if archives is None:
            return text

        if key not in archives:
            return text
        archive = archives[key]
        for k in text.keys():
            if k in archive:
                text[k] = archive[k]

        return text

    ret = {}
    root_text = {}
    root_text[comp_list.name] = ""
    root_text[comp_list.desc] = ""

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

        key = gen_key(comp.domain, comp.name, comp.version)
        ret[key] = restore_from_archives(text, key, archives)

    return ret


@click.command()
@click.option(
    "--archives",
    "-a",
    required=False,
    type=click.Path(dir_okay=False, readable=True),
    help="path to archives",
)
@click.option(
    "--output",
    "-o",
    required=False,
    type=click.Path(dir_okay=False, writable=True),
    help="path to output",
)
def cli(archives, output):
    archived_text = None
    if archives:
        click.echo("-" * 105)
        click.echo(f"reading archives from {archives}...")
        click.echo("-" * 105)
        with open(archives, "r") as f:
            archived_text = json.load(f)

    text = gettext(COMP_LIST, archived_text)
    click.echo("-" * 105)
    click.echo("gettext result:")
    click.echo("")
    click.echo(json.dumps(text, indent=2))
    click.echo("-" * 105)

    if output:
        click.echo("-" * 105)
        click.echo(f"writing texts to {output}...")
        click.echo("-" * 105)
        with open(output, "w") as f:
            json.dump(text, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    cli()
