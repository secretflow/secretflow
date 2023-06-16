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

import logging
import subprocess
import sys

import click

from secretflow.component.entry import comp_eval
from secretflow.kuscia.ray_config import RayConfig
from secretflow.kuscia.sf_config import get_sf_cluster_config
from secretflow.kuscia.task_config import KusicaTaskConfig

_LOG_FORMAT = "%(asctime)s|{}|%(levelname)s|fascia|%(filename)s:%(funcName)s:%(lineno)d| %(message)s"


def start_ray(ray_conf: RayConfig):
    logging.info(f"ray_conf: {ray_conf}")

    ray_cmd = ray_conf.generate_ray_cmd()

    logging.info(
        f"Trying to start ray head node at {ray_conf.ray_node_ip_address}, start command: {ray_cmd}"
    )

    process = subprocess.run(ray_cmd, capture_output=True, shell=True)

    if process.returncode != 0:
        err_msg = f"Failed to start ray head node, start command: {ray_cmd}, stderr: {process.stderr}"
        logging.critical(err_msg)
        logging.critical("This process will exit now!")
        sys.exit(-1)
    else:
        logging.info(process.stdout)
        logging.info(
            f"Succeeded to start ray head node at {ray_conf.ray_node_ip_address}."
        )


@click.command()
@click.argument("task_config_path", type=click.Path(exists=True))
def main(task_config_path):
    task_conf = KusicaTaskConfig.from_file(task_config_path)

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=_LOG_FORMAT.format(task_conf.party_name),
        force=True,
    )

    ray_config = RayConfig.from_kuscia_task_config(task_conf)
    start_ray(ray_config)

    sf_cluster_config = get_sf_cluster_config(task_conf)
    res = comp_eval(task_conf.sf_node_eval_param, sf_cluster_config)
    logging.info(f"Succeeded to run component. The result is {res}")

    sys.exit(0)


if __name__ == "__main__":
    main()
