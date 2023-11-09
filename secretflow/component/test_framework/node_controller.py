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

import base64
import json
import logging
import math
import os
import gzip
import shutil
import socket
import subprocess
import time
import uuid
from typing import Dict, Tuple

from secretflow.spec.extend.cluster_pb2 import SFClusterConfig, SFClusterDesc
from secretflow.spec.v1.data_pb2 import StorageConfig
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam, NodeEvalResult

from .test_case import NetCase, TestNode, TestVersion

TEST_DOCKER_NAME_TEMPLATE = "sf-auto-test-docker-{party}"

DOCKER_IMAGE = "secretflow/secretflow-anolis8:{version}"

DOCKER_CMD_TEMPLATE = 'docker exec {name} /bin/bash -c "{cmd}"'

DOCKER_LOCAL_PATH = "/root/sf-auto-test"

INIT_DOCKER_TEMPLATE = (
    "docker rm -f {name} && "
    "docker run -d -it --name {name} --rm "
    "--cap-add=NET_ADMIN "
    "--cap-add=SYS_PTRACE --security-opt seccomp=unconfined "
    # mount local_fs_path to /root/sf-auto-test
    "--mount type=bind,source={local_path},target={docker_path} "
    "-p {rayfed}:{rayfed} -p {spu}:{spu} "
    "--shm-size={shm_size}gb "
)

DOCKER_INSTALL_WHL = (
    "pip uninstall {whl} -y && "
    "pip install {whl} -i https://mirrors.bfsu.edu.cn/pypi/web/simple"
)

RESTART_DOCKER_TEMPLATE = "docker restart {docker}"

DOCKER_INSTALL_TC = "yum install iproute-tc -y"

TEST_DOCKER_CLEAR_UP = "docker rm -f {name}"

NET_BANDWIDTH_TEMPLATE = "tc qdisc add dev {nic} root handle 1: tbf rate {limit}mbit burst {burst}kb latency 800ms"

NET_LATENCY_TEMPLATE = (
    "tc qdisc add dev {nic} parent 1:1 handle 10: netem delay {delay}msec limit {limit}"
)

NET_CLEAR_TEMPLATE = "tc qdisc del dev {nic} root || true"


class NodeController(object):
    def __init__(self, node: TestNode, aci_mode=False) -> None:
        self.aci_mode = aci_mode
        self.node = node
        self.docker_name = TEST_DOCKER_NAME_TEMPLATE.format(party=self.node.party)
        if not aci_mode and socket.getfqdn(node.hostname) != socket.gethostname():
            from paramiko.client import SSHClient

            self.ssh = SSHClient()
            self.ssh.load_system_host_keys()
            self.ssh.connect(
                node.hostname,
                port=node.ssh_port,
                username=node.ssh_user,
                password=node.ssh_passwd,
            )
        else:
            # running locally
            self.ssh = None

    def clear_up(self) -> None:
        if not self.aci_mode:
            self._exec_cmd(TEST_DOCKER_CLEAR_UP.format(name=self.docker_name))

    def _ssh_cmd(self, cmd: str) -> bytes:
        from paramiko.ssh_exception import SSHException

        try:
            logging.info(f"Exe ssh to {self.node.hostname}: {cmd}")
            _, stdout, stderr = self.ssh.exec_command(cmd, get_pty=True)
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                raise RuntimeError(
                    f"exec cmd {cmd} got exit code {exit_code}, error {stderr.read().decode('utf-8')}"
                )
            return stdout.read()
        except SSHException as e:
            raise RuntimeError(f"ssh error {e}") from None

    def _local_cmd(self, cmd: str) -> bytes:
        try:
            logging.info(f"Exe local on {self.node.hostname}: {cmd}")
            c = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True)
            if c.returncode != 0:
                raise RuntimeError(
                    f"exec cmd {cmd} got exit code {c.returncode}, error {c.stderr.decode('utf-8')}"
                )
            return c.stdout
        except Exception as e:
            raise RuntimeError(f"subprocess error {e}") from None

    def _exec_cmd(self, cmd: str, in_docker: bool = False) -> bytes:
        if in_docker and not self.aci_mode:
            cmd = DOCKER_CMD_TEMPLATE.format(name=self.docker_name, cmd=cmd)

        if self.ssh:
            return self._ssh_cmd(cmd)
        else:
            return self._local_cmd(cmd)

    def _cp_file_to_fs_path(self, file: str) -> None:
        dst_path = os.path.join(self.node.local_fs_path, os.path.basename(file))
        if self.ssh:
            with self.ssh.open_sftp() as ftp_client:
                try:
                    ftp_client.remove(dst_path)
                except Exception:
                    # ignore remove err, may not exist.
                    pass
                logging.info(
                    f"ftp: {file} to {self.node.hostname}:{self.node.local_fs_path}"
                )
                ftp_client.put(file, dst_path)
        else:
            if os.path.exists(dst_path):
                if os.path.samefile(file, dst_path):
                    # if pwd is the local_fs_path, skip cp
                    return
                else:
                    os.remove(dst_path)
            logging.info(
                f"cp: {file} to {self.node.hostname}:{self.node.local_fs_path}"
            )
            shutil.copyfile(file, dst_path)

    def _init_docker(self, sf_version: TestVersion):
        # 创建logs目录
        logs_dir = os.path.join(self.node.local_fs_path, "logs")
        cmd = f"if ! [ -e {logs_dir} ]; then mkdir {logs_dir}; fi"
        self._exec_cmd(cmd)

        if self.node.docker_mem_limit > 0:
            shm_size_gb = self.node.docker_mem_limit / 2.0 / 1024
        else:
            cmd = "grep MemTotal /proc/meminfo | awk '{print $2}'"
            total_mem_kd = float(self._exec_cmd(cmd))
            shm_size_gb = total_mem_kd / (1024**2) / 2

        cmd = INIT_DOCKER_TEMPLATE.format(
            name=self.docker_name,
            local_path=self.node.local_fs_path,
            docker_path=DOCKER_LOCAL_PATH,
            rayfed=self.node.rayfed_port,
            spu=self.node.spu_port,
            shm_size=shm_size_gb,
        )
        if self.node.docker_mem_limit > 0:
            cmd += "--memory={mem}m ".format(mem=self.node.docker_mem_limit)
        if self.node.docker_cpu_limit > 0:
            cmd += "--cpus={core} ".format(core=self.node.docker_cpu_limit)

        if sf_version.source == "public":
            docker_version = sf_version.version
        else:
            # install whl inside secretflow-anolis8:latest
            docker_version = "latest"

        cmd += DOCKER_IMAGE.format(version=docker_version)
        self._exec_cmd(cmd)

        # wait for container to be ready
        logging.info("sleep 5s")
        time.sleep(5)

        if sf_version.source == "whl":
            whl = " ".join(
                [
                    os.path.join(DOCKER_LOCAL_PATH, os.path.basename(w))
                    for w in sf_version.whl_paths
                ]
            )
            cmd = DOCKER_INSTALL_WHL.format(whl=whl)
            self._exec_cmd(cmd, in_docker=True)
        self._exec_cmd(DOCKER_INSTALL_TC, in_docker=True)

    def init_env(self, sf_version: TestVersion) -> None:
        if self.aci_mode:
            return
        self.current_sf_version = sf_version
        if sf_version.source == "whl":
            for w in sf_version.whl_paths:
                self._cp_file_to_fs_path(w)
        self._init_docker(sf_version)

    def _limit_network_cmd(self, limit: NetCase) -> str:
        cmds = [NET_CLEAR_TEMPLATE.format(nic=self.node.nic_name)]
        if limit.limit_mb > 0:
            burst = 2 ** math.ceil(math.log2(limit.limit_mb))
            cmds.append(
                NET_BANDWIDTH_TEMPLATE.format(
                    nic=self.node.nic_name, limit=limit.limit_mb, burst=burst
                )
            )

        if limit.limit_ms > 0:
            assert limit.limit_mb > 0
            netem_limit = int(
                limit.limit_mb * 1000 * 1000 / 1280 / 8 * limit.limit_ms / 1000 * 2
            )
            cmds.append(
                NET_LATENCY_TEMPLATE.format(
                    nic=self.node.nic_name, delay=limit.limit_ms, limit=netem_limit
                )
            )

        return " && ".join(cmds)

    def limit_network(self, limit: NetCase) -> None:
        if self.aci_mode:
            return
        self.current_net_limit = limit
        cmd = self._limit_network_cmd(limit)
        self._exec_cmd(cmd, in_docker=True)

    def run_comp(
        self,
        uid: str,
        eval: NodeEvalParam,
        clu_desc: SFClusterDesc,
        nodes: Dict[str, TestNode],
    ) -> Tuple[float, float, NodeEvalResult]:
        uuid_path = str(uuid.uuid4())
        comp_name = f"{uid}:{eval.domain}:{eval.name}:{eval.version}"
        logging.info(
            f"run comp {comp_name} on node {self.node.party} with uuid_path {uuid_path}"
        )
        if not self.aci_mode:
            # restart docker to flush memory.max_usage_in_bytes in docker's cgroup
            self._exec_cmd(RESTART_DOCKER_TEMPLATE.format(docker=self.docker_name))
            # need rebuild net limit too.
            cmd = self._limit_network_cmd(self.current_net_limit)
            self._exec_cmd(cmd, in_docker=True)

        local_path = DOCKER_LOCAL_PATH if not self.aci_mode else self.node.local_fs_path

        clu_config = SFClusterConfig()
        clu_config.desc.CopyFrom(clu_desc)

        public_config = SFClusterConfig.PublicConfig()
        pyus = list(clu_desc.parties)
        public_config.ray_fed_config.CopyFrom(
            SFClusterConfig.RayFedConfig(
                parties=pyus,
                addresses=[
                    "{host}:{port}".format(
                        host=nodes[p].hostname, port=nodes[p].rayfed_port
                    )
                    for p in pyus
                ],
                listen_addresses=[
                    "0.0.0.0:{port}".format(port=nodes[p].rayfed_port) for p in pyus
                ],
            )
        )
        spu_parties = list(clu_desc.devices[0].parties)
        public_config.spu_configs.append(
            SFClusterConfig.SPUConfig(
                name="spu",
                parties=spu_parties,
                addresses=[
                    "{host}:{port}".format(
                        host=nodes[p].hostname, port=nodes[p].spu_port
                    )
                    for p in spu_parties
                ],
                listen_addresses=[
                    "0.0.0.0:{port}".format(port=nodes[p].spu_port) for p in spu_parties
                ],
            )
        )
        clu_config.public_config.CopyFrom(public_config)

        private_config = SFClusterConfig.PrivateConfig()
        private_config.self_party = self.node.party
        private_config.ray_head_addr = "local"
        storage_config = StorageConfig()
        storage_config.type = "local_fs"
        storage_config.local_fs.wd = local_path
        clu_config.private_config.CopyFrom(private_config)

        eval_encode = base64.b64encode(gzip.compress(eval.SerializeToString())).decode(
            "utf-8"
        )

        clu_encode = base64.b64encode(
            gzip.compress(clu_config.SerializeToString())
        ).decode("utf-8")

        sto_encode = base64.b64encode(
            gzip.compress(storage_config.SerializeToString())
        ).decode("utf-8")

        log_file = os.path.join(local_path, "logs", f"{uuid_path}.log")
        result_file = os.path.join(local_path, "logs", f"{uuid_path}.result")

        cmd = (
            f"secretflow component run --log_file={log_file} --result_file={result_file} "
            f"--eval_param={eval_encode} --storage={sto_encode} --cluster={clu_encode} "
            f"--mem_trace --compressed_params"
        )
        self._exec_cmd(cmd, in_docker=True)
        result = self._exec_cmd(f"cat {result_file}", in_docker=True)
        result = json.loads(result)

        if result["error_msg"] is not None:
            logging.error(
                f"run comp {comp_name} on node {self.node.party} error {result['error_msg']}"
            )
            raise RuntimeError(f"run comp {comp_name} error {result['error_msg']}")
        else:
            evla_result = NodeEvalResult()
            evla_result.ParseFromString(
                base64.b64decode(result["result"].encode("utf-8"))
            )

        return (result["mem_peak"], result["run_time"], evla_result)
