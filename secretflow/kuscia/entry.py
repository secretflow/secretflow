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
import multiprocessing
import subprocess
import sys

from absl import app, flags

import secretflow as sf
from secretflow.component.entry import run
from secretflow.kuscia.task_config import TaskConfig

_LOG_FORMAT = '%(asctime)s|{}|%(levelname)s|fascia|%(filename)s:%(funcName)s:%(lineno)d| %(message)s'

flags.DEFINE_string("task_config_path", None, "Task config file path.")
FLAGS = flags.FLAGS

_RAY_GRPC_ENV = 'RAY_BACKEND_LOG_LEVEL=debug ' 'RAY_grpc_enable_http_proxy=true '


def start_ray(task_conf: TaskConfig):
    logging.info(f'task_conf: {task_conf}')

    ray_cmd = (
        f'{_RAY_GRPC_ENV}'
        f'OMP_NUM_THREADS={multiprocessing.cpu_count()} '
        'ray start --head --include-dashboard=false --disable-usage-stats'
        f' --num-cpus=8'
        f' --node-ip-address={task_conf.ray_node_ip_address}'
        f' --port={task_conf.ray_gcs_port}'
    )

    if task_conf.ray_node_manager_port:
        ray_cmd += f' --node-manager-port={task_conf.ray_node_manager_port}'
    if task_conf.ray_object_manager_port:
        ray_cmd += f' --object-manager-port={task_conf.ray_object_manager_port}'
    if task_conf.ray_client_server_port:
        ray_cmd += f' --ray-client-server-port={task_conf.ray_client_server_port}'
    if task_conf.ray_worker_ports:
        ray_cmd += (
            f' --worker-port-list={",".join(map(str, task_conf.ray_worker_ports))}'
        )

    logging.info(
        f'Trying to start ray head node at {task_conf.ray_node_ip_address}, start command: {ray_cmd}'
    )

    process = subprocess.run(ray_cmd, capture_output=True, shell=True)
    if process.returncode != 0:
        err_msg = f'Failed to start ray head node, start command: {ray_cmd}, stderr: {process.stderr}'
        logging.critical(err_msg)
        logging.critical(f'This processor will exit now!')
        sys.exit(-1)
    else:
        logging.info(process.stdout)
        logging.info(
            f'Succeeded to start ray head node at {task_conf.ray_node_ip_address}.'
        )

        logging.info(
            f'Starting secretflow with cluster config: {task_conf.cluster_config}'
        )
        retry_policy = {
            "maxAttempts": 5,
            "initialBackoff": "20s",
            "maxBackoff": "20s",
            "backoffMultiplier": 1,
            "retryableStatusCodes": ["UNAVAILABLE"],
        }
        sf.init(
            address=f'{task_conf.ray_node_ip_address}:{task_conf.ray_gcs_port}',
            cluster_config=task_conf.cluster_config,
            logging_level='debug',
            cross_silo_grpc_retry_policy=retry_policy,
            cross_silo_send_max_retries=3,
        )


def main(argv):
    del argv

    task_conf = TaskConfig().parse_from_file(FLAGS.task_config_path)

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=_LOG_FORMAT.format(task_conf.party_name),
        force=True,
    )

    start_ray(task_conf)

    run(task_conf.comp_node, task_conf.spu_cluster_config)
    logging.info('Succeeded to run component.')

    sf.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    app.run(main)
