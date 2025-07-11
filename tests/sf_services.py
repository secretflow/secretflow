# Copyright 2025 Ant Group Co., Ltd.
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


import atexit
import logging
from typing import Any, Callable

from secretflow.utils.testing import unused_tcp_port
from tests.sf_config import build_s3_config, get_storage_root

_service_params: dict[str, Any] = {}


def set_service_param(key: str, value: Any):
    _service_params[key] = value


def get_service_param(key: str) -> Any:
    return _service_params[key]


def get_service_params() -> dict[str, Any]:
    return _service_params


SERVICE_KEY_MINIO_PORT = "minio_port"
SERVICE_KEY_AUTH_PORT = "auth_port"


def minio_server_is_ready(config: dict):
    import logging

    import s3fs

    endpoint_url = config["endpoint_url"]
    s3_client = s3fs.S3FileSystem(
        anon=False,
        key=config["access_key"],
        secret=config["access_secret"],
        client_kwargs={'endpoint_url': endpoint_url},
        config_kwargs={
            'signature_version': config["version"],
            's3': {
                "addressing_style": (
                    "virtual" if config.get("virtual_host", False) else "path"
                )
            },
        },
    )
    try:
        s3_client.ls(config["bucket"], detail=False)
        logging.info(f"minio server is ready: {endpoint_url}")
        return True
    except Exception as e:
        logging.warning(f"minio server is not ready: {e}")
        return False


def start_minio_server():
    import os
    import stat
    import subprocess
    import time
    import urllib
    import uuid

    storage_root = get_storage_root()

    storage_path = os.path.join(storage_root, str(uuid.uuid4()))
    # os.makedirs(storage_path, exist_ok=True)
    # minio_path = os.path.join(storage_path, f"minio_{self_party}")
    minio_path = storage_path
    os.makedirs(minio_path, exist_ok=True)

    minio_server = os.path.join(minio_path, "minio")
    if not os.path.exists(minio_server) or not os.path.isfile(minio_server):
        # system = "linux"
        # arch = "amd64"
        # if platform.system() == "Darwin":
        #     system = "darwin"
        # if platform.machine() == "arm64" or platform.machine() == "aarch64":
        #     arch = "arm64"
        urllib.request.urlretrieve(
            f"https://dl.min.io/server/minio/release/{system}-{arch}/minio",
            minio_server,
        )
        st = os.stat(minio_server)
        os.chmod(minio_server, st.st_mode | stat.S_IEXEC)

    minio_data_path = os.path.join(minio_path, "data")
    os.makedirs(
        os.path.join(minio_data_path, "sf-test"),
        exist_ok=True,
    )

    ms_env = os.environ.copy()
    ms_env["MINIO_BROWSER"] = "off"
    ms_env["MINIO_UPDATE"] = "off"
    ms_env["MINIO_ROOT_USER"] = "sf_test_aaa"
    ms_env["MINIO_ROOT_PASSWORD"] = "sf_test_sss"

    port = unused_tcp_port()
    if port == 0:
        raise ValueError(f"no available port for minio.")

    set_service_param(SERVICE_KEY_MINIO_PORT, port)
    endpoint = f"127.0.0.1:{port}"
    ms = subprocess.Popen(
        [minio_server, "server", minio_data_path, "--address", endpoint],
        env=ms_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    logging.info(f"run minio server: pid={ms.pid}, addr={endpoint}, path={minio_path}")
    s3_conf = build_s3_config(port)

    wait_count = 0
    while True:
        time.sleep(0.4)
        if minio_server_is_ready(s3_conf):
            break
        wait_count += 1
        if wait_count > 2:
            logging.error(
                f"minio server is not ready for {wait_count * 0.4}s, terminate."
            )
            raise RuntimeError(
                f"minio server is not ready for {wait_count * 0.4}s, terminate."
            )

    def clean():
        if ms.poll() is None:
            ms.terminate()
            ms.wait()

    atexit.register(clean)


def start_auth_server():
    from tests.utils.auth_manager import start_auth_server

    auth_port = unused_tcp_port()
    set_service_param(SERVICE_KEY_AUTH_PORT, auth_port)
    server = start_auth_server(auth_port)

    def clean():
        server.stop(grace=True)

    atexit.register(clean)


SERVICE_MINIO = "minio"
SERVICE_AUTH = "auth"

_services = {
    SERVICE_MINIO: start_minio_server,
    SERVICE_AUTH: start_auth_server,
}


def get_services() -> dict[str, Callable]:
    return _services
