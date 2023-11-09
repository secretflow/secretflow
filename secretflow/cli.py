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
import json
import logging
import gzip
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

import click
from google.protobuf.json_format import MessageToJson
from secretflow.spec.v1.data_pb2 import StorageConfig
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from secretflow.component.entry import COMP_LIST, COMP_MAP, comp_eval
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.utils.logging import LOG_FORMAT, get_logging_level, set_logging_level
from secretflow.version import __version__


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"SecretFlow version {__version__}.")
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
def ls():
    """List all components."""
    click.echo("{:<40} {:<40} {:<20}".format("DOMAIN", "NAME", "VERSION"))
    click.echo("-" * 105)
    for comp in COMP_LIST.comps:
        click.echo("{:<40} {:<40} {:<20}".format(comp.domain, comp.name, comp.version))


@component.command()
@click.option("--file", "-f", required=False, type=click.File(mode="w"))
@click.option(
    "--all",
    "-a",
    is_flag=True,
)
@click.argument("comp_id", required=False)
def inspect(comp_id, all, file):
    """Display definition of components. The format of comp_id is {domain}/{name}:{version}"""

    if all:
        click.echo(f"You are inspecting the compelete comp list.")
        click.echo("-" * 105)
        if file:
            click.echo(
                json.dumps(json.loads(MessageToJson(COMP_LIST)), indent=2), file=file
            )
            click.echo(f"Saved to {file.name}.")
        else:
            click.echo(json.dumps(json.loads(MessageToJson(COMP_LIST)), indent=2))

    elif comp_id:
        if comp_id in COMP_MAP:
            click.echo(
                f"You are inspecting definition of component with id [{comp_id}]."
            )
            click.echo("-" * 105)
            if file:
                click.echo(
                    json.dumps(
                        json.loads(MessageToJson(COMP_MAP[comp_id].definition())),
                        indent=2,
                    ),
                    file=file,
                )
                click.echo(f"Saved to {file.name}.")
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
            "You must provide comp_id or use --all/-a for the compelete comp list."
        )


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
def run(
    eval_param,
    storage,
    cluster,
    log_file,
    result_file,
    log_level,
    mem_trace,
    compressed_params,
):
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
