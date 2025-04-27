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

import multiprocessing
import os
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

from secretflow.kuscia.task_config import KusciaTaskConfig


@dataclass
class RayConfig:
    ray_node_ip_address: str
    ray_node_manager_port: int = None
    ray_object_manager_port: int = None
    ray_client_server_port: int = None
    ray_worker_ports: List[int] = None
    ray_min_worker_port: int = None
    ray_max_worker_port: int = None
    ray_gcs_port: List[int] = None

    def generate_ray_cmd(self) -> Tuple[List, Dict]:
        if self.ray_node_ip_address == "local":
            return None, None

        envs = os.environ.copy()
        envs["RAY_BACKEND_LOG_LEVEL"] = "debug"
        envs["OMP_NUM_THREADS"] = f"{multiprocessing.cpu_count()}"

        ray_cmd = [
            "ray",
            "start",
            "--head",
            "--include-dashboard=false",
            "--disable-usage-stats",
            "--num-cpus=32",
            f"--node-ip-address={self.ray_node_ip_address}",
            f"--port={self.ray_gcs_port}",
        ]

        if self.ray_node_manager_port:
            ray_cmd.append(f"--node-manager-port={self.ray_node_manager_port}")
        if self.ray_object_manager_port:
            ray_cmd.append(f"--object-manager-port={self.ray_object_manager_port}")
        if self.ray_client_server_port:
            ray_cmd.append(f"--ray-client-server-port={self.ray_client_server_port}")
        if self.ray_worker_ports:
            ray_cmd.append(
                f'--worker-port-list={",".join(map(str, self.ray_worker_ports))}'
            )
        elif self.ray_min_worker_port & self.ray_max_worker_port:
            ray_cmd.append(f"--min-worker-port={self.ray_min_worker_port}")
            ray_cmd.append(f"--max-worker-port={self.ray_max_worker_port}")
        return ray_cmd, envs

    @classmethod
    def from_kuscia_task_config(cls, config: KusciaTaskConfig):
        allocated_port = config.task_allocated_ports
        cluster_define = config.task_cluster_def

        ray_worker_ports = []
        ray_min_worker_port = 0
        ray_max_worker_port = 0
        ray_node_manager_port = 0
        ray_object_manager_port = 0
        ray_client_server_port = 0
        ray_node_ip_address = ""
        ray_gcs_port = 0

        party_name = config.party_name
        for port in allocated_port.ports:
            if port.name.startswith("ray-worker"):
                ray_worker_ports.append(port.port)
            elif port.name == "node-manager":
                ray_node_manager_port = port.port
            elif port.name == "object-manager":
                ray_object_manager_port = port.port
            elif port.name == "client-server":
                ray_client_server_port = port.port

        # ray work nums = cpu nums + 1
        # ray max cpu nums is 32
        # port range allocated by ray workers [10000, 20000]
        ray_min_worker_port = random.randint(10000, 19900)
        ray_max_worker_port = ray_min_worker_port + 100

        for party in cluster_define.parties:
            if party.name == party_name:
                for service in party.services:
                    if service.port_name == "global":
                        segs = service.endpoints[0].split(":")
                        ray_node_ip_address = segs[0]
                        if len(segs) == 2:
                            ray_gcs_port = int(segs[1])
                        else:
                            ray_gcs_port = 80

        return RayConfig(
            ray_node_ip_address,
            ray_node_manager_port,
            ray_object_manager_port,
            ray_client_server_port,
            ray_worker_ports,
            ray_min_worker_port,
            ray_max_worker_port,
            ray_gcs_port,
        )
