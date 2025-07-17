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

import base64
import gzip
import importlib
import importlib.util
import json
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

import click
from google.protobuf.json_format import MessageToJson
from secretflow_spec.v1.data_pb2 import StorageConfig
from secretflow_spec.v1.evaluation_pb2 import NodeEvalParam

from secretflow.component.core import Registry, comp_eval, get_comp_list_def
from secretflow.component.core import get_translation as core_get_translation
from secretflow.component.core import load_plugins
from secretflow.component.core import translate as core_translate
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.utils.logging import LOG_FORMAT, get_logging_level, set_logging_level
from secretflow.version import build_message


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(build_message())
    ctx.exit()


@click.group()
@click.option(
    "--version",
    "-v",
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
@click.option(
    "--enable_plugins",
    required=False,
    default=True,
    help="Whether to enable loading plugins",
)
def ls(enable_plugins):
    """List all components."""
    if enable_plugins:
        load_plugins()
    click.echo("{:<40} {:<40} {:<20}".format("DOMAIN", "NAME", "VERSION"))
    click.echo("-" * 105)
    for comp in Registry.get_definitions():
        click.echo("{:<40} {:<40} {:<20}".format(comp.domain, comp.name, comp.version))


@component.command()
@click.option("--file", "-f", required=False, type=click.File(mode="w"))
@click.option("--all", "-a", is_flag=True)
@click.argument("comp_id", required=False)
@click.option(
    "--enable_plugins",
    required=False,
    default=True,
    help="Whether to enable loading plugins",
)
def inspect(comp_id, all, file, enable_plugins):
    """Display definition of components. The format of comp_id is {domain}/{name}:{version}"""
    if enable_plugins:
        load_plugins()

    if all:
        json_data = json.dumps(json.loads(MessageToJson(get_comp_list_def())), indent=2)
        if file:
            click.echo(json_data, file=file)
            click.echo(f"Saved to {file.name}.")
        else:
            click.echo(json_data)

    elif comp_id:
        comp_def = Registry.get_definition_by_id(comp_id)
        if comp_def and comp_def.component_id == comp_id:
            json_data = json.dumps(
                json.loads(MessageToJson(comp_def.component_def)),
                indent=2,
            )
            if file:
                click.echo(json_data, file=file)
                click.echo(f"Saved to {file.name}.")
            else:
                click.echo(json_data)
        else:
            click.echo(f"Component with id [{comp_id}] is not found.")

    else:
        click.echo(
            "You must provide comp_id or use --all/-a for the compelete comp list."
        )


@component.command()
@click.option("--file", "-f", required=False, type=str, default="translation.json")
@click.option(
    "--entry_point",
    "-e",
    required=False,
    type=str,
    default="",
    help="entry_point of plugin, for example 'my_plugin.entry:main'",
)
@click.option(
    "--package",
    "-p",
    required=False,
    default="",
    help="root package name, if empty, it will be infered from entry_point dir",
)
def translate(file: str, entry_point: str, package: str):
    def find_root_package_dir(dir: str) -> str:
        while dir:
            init_filename = os.path.join(dir, '__init__.py')
            if not os.path.isfile(init_filename):
                return dir
            dir = os.path.dirname(dir)

    curr_dir = os.getcwd()
    root_pkg_dir = find_root_package_dir(curr_dir)

    if root_pkg_dir not in sys.path:
        sys.path.insert(0, root_pkg_dir)

    if ':' in entry_point:
        module_name, func_name = entry_point.split(':')
    else:
        module_name, func_name = entry_point, None

    if module_name == "":
        module_name = os.path.relpath(curr_dir, root_pkg_dir).replace(os.path.sep, '.')

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        click.echo(
            f"import fail, root_dir={root_pkg_dir}, module={module_name}, err={e}"
        )
        return

    if func_name:
        try:
            func = getattr(module, func_name)
        except Exception:
            click.echo(f"Module '{module_name}' has no attribute '{func_name}'")
            return
        if callable(func):
            func()
        else:
            click.echo(f"{func_name} is not a func.")

    if not package:
        package = module.__package__.split('.')[0]

    if not package:
        click.echo(f"empty package")
        return

    archieve = None
    if os.path.isfile(file):
        with open(file, "r") as f:
            archieve = json.load(f)

    translation = core_translate(package, archieve)

    click.echo(f"You are translating the compelete comp list.")
    click.echo("-" * 105)
    with open(file, "w") as f:
        click.echo(json.dump(translation, f, indent=2, ensure_ascii=False), file=f)
    click.echo(f"Saved to {file}.")


@component.command(name='get_translation')
@click.option(
    "--enable_plugins",
    required=False,
    default=True,
    help="Whether to enable loading plugins",
)
def get_translation(enable_plugins: bool):
    if enable_plugins:
        load_plugins()

    translation = core_get_translation()
    click.echo(json.dumps(translation, indent=2, ensure_ascii=False))


@component.command()
@click.option(
    "--log_file",
    required=False,
    type=click.Path(dir_okay=False, writable=True),
    help="log file. if not specified, logging to stdout",
)
@click.option(
    "--result_file",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="result file. component saved in file with json format",
)
@click.option("--log_level", required=False, default="INFO")
@click.option("--mem_trace", is_flag=True)
@click.option("--eval_param", required=True, help="base64ed NodeEvalParam binary")
@click.option("--storage", required=True, help="base64ed Storage binary")
@click.option("--cluster", required=True, help="base64ed SFClusterConfig binary")
@click.option(
    "--compressed_params", is_flag=True, help="compress params before base64 encode"
)
@click.option(
    "--enable_plugins",
    required=False,
    default=False,
    help="Whether to enable loading plugins",
)
def run(
    eval_param,
    storage,
    cluster,
    log_file,
    result_file,
    log_level,
    mem_trace,
    compressed_params,
    enable_plugins: bool,
):
    if enable_plugins:
        load_plugins()

    def _get_peak_mem() -> float:
        # only works inside docker
        # use docker's default cgroup
        cgroup_v1_path = "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"
        cgroup_v2_path = "/sys/fs/cgroup/memory.peak"
        try:
            if os.path.exists(cgroup_v1_path):
                max_path = cgroup_v1_path
            else:
                max_path = cgroup_v2_path
            with open(max_path, "r") as f:
                max_usage = int(f.read())
            return float(max_usage) / (1024**3)
        except Exception as e:
            logging.error(f"get_peak_mem error {e}")
            return 0

    set_logging_level(log_level)
    logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)

    ret = {
        "mem_peak": 0,
        "run_time": 0,
        "result": None,
        "error_msg": None,
        "error_code": 0,
    }
    try:
        eval = NodeEvalParam()
        eval_ser = base64.b64decode(eval_param.encode('utf-8'))
        if compressed_params:
            eval_ser = gzip.decompress(eval_ser)
        eval.ParseFromString(eval_ser)

        sto = StorageConfig()
        sto_ser = base64.b64decode(storage.encode('utf-8'))
        if compressed_params:
            sto_ser = gzip.decompress(sto_ser)
        sto.ParseFromString(sto_ser)

        clu = SFClusterConfig()
        clu_ser = base64.b64decode(cluster.encode('utf-8'))
        if compressed_params:
            clu_ser = gzip.decompress(clu_ser)
        clu.ParseFromString(clu_ser)
    except Exception as e:
        ret["error_msg"] = f"parse argv err: {e}"
        ret["error_code"] = -1  # TODO: use real code
        logging.error(ret["error_msg"])
        with open(result_file, "w") as f:
            f.write(json.dumps(ret))
        sys.exit(-1)

    try:
        if log_file:
            with open(log_file, "w") as f:
                with redirect_stdout(f), redirect_stderr(f):
                    result = comp_eval(eval, sto, clu, tracer_report=True)
        else:
            result = comp_eval(eval, sto, clu, tracer_report=True)

        if mem_trace:
            ret["mem_peak"] = _get_peak_mem()
        ret["run_time"] = result["tracer_report"]["run_time"]
        result = result["eval_result"]
        ret["result"] = base64.b64encode(result.SerializeToString()).decode('utf-8')
    except Exception as e:
        ret["error_msg"] = f"run comp err: {e}"
        ret["error_code"] = -1  # TODO: use real code
        logging.error(ret["error_msg"])

    with open(result_file, "w") as f:
        f.write(json.dumps(ret))

    if ret["error_code"] != 0:
        sys.exit(-1)
