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

import click
from secretflow.version import __version__
from secretflow.component.entry import COMP_LIST, COMP_MAP
import json
from google.protobuf.json_format import MessageToJson


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f'SecretFlow version {__version__}.')
    ctx.exit()


@click.group()
@click.option(
    '--version',
    '-v',
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
)
def cli():
    """Welcome to use cli of SecretFlow."""
    pass


@cli.group()
def component():
    """Get information of components in SecretFlow package."""
    pass


@component.command()
def ls():
    """List all components."""
    click.echo('{:<40} {:<40} {:<20}'.format('DOMAIN', 'NAME', 'VERSION'))
    click.echo('-' * 105)
    for comp in COMP_LIST.comps:
        click.echo('{:<40} {:<40} {:<20}'.format(comp.domain, comp.name, comp.version))


@component.command()
@click.option('--file', '-f', required=False, type=click.File(mode='w'))
@click.option(
    '--all',
    '-a',
    is_flag=True,
)
@click.argument('comp_id', required=False)
def inspect(comp_id, all, file):
    """Display definition of components. The format of comp_id is {domain}/{name}:{version}"""

    if all:
        click.echo(f"You are inspecting the compelete comp list.")
        click.echo('-' * 105)
        if file:
            click.echo(
                json.dumps(json.loads(MessageToJson(COMP_LIST)), indent=2), file=file
            )
            click.echo(f'Saved to {file.name}.')
        else:
            click.echo(json.dumps(json.loads(MessageToJson(COMP_LIST)), indent=2))

    elif comp_id:
        if comp_id in COMP_MAP:
            click.echo(
                f"You are inspecting definition of component with id [{comp_id}]."
            )
            click.echo('-' * 105)
            if file:
                click.echo(
                    json.dumps(
                        json.loads(MessageToJson(COMP_MAP[comp_id].definition())),
                        indent=2,
                    ),
                    file=file,
                )
                click.echo(f'Saved to {file.name}.')
            else:
                click.echo(
                    json.dumps(
                        json.loads(MessageToJson(COMP_MAP[comp_id].definition())),
                        indent=2,
                    )
                )
        else:
            click.echo(f"Component with id [{comp_id}] is not found.")

    else:
        click.echo(
            'You must provide comp_id or use --all/-a for the compelete comp list.'
        )
