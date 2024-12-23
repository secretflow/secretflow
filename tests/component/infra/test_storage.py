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

import logging
import os
import platform
import stat
import subprocess
import tempfile
import urllib
import uuid

from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.data_pb2 import StorageConfig
from secretflow.utils import secure_pickle as pickle


def build_s3_config():
    minio_path = tempfile.gettempdir()
    os.makedirs(minio_path, exist_ok=True)

    minio_server = os.path.join(minio_path, "minio")

    system = "linux"
    arch = "amd64"
    if platform.system() == "Darwin":
        system = "darwin"
    if platform.machine() == "arm64" or platform.machine() == "aarch64":
        arch = "arm64"
    urllib.request.urlretrieve(
        f"https://dl.min.io/server/minio/release/{system}-{arch}/minio",
        minio_server,
    )

    st = os.stat(minio_server)
    os.chmod(minio_server, st.st_mode | stat.S_IEXEC)

    minio_data_path = os.path.join(minio_path, "data")

    ms_env = os.environ.copy()
    ms_env["MINIO_BROWSER"] = "off"
    ms_env["MINIO_ACCESS_KEY"] = "sf_test_aaa"
    ms_env["MINIO_SECRET_KEY"] = "sf_test_sss"

    endpoint = "127.0.0.1:63111"
    config = StorageConfig(
        type="s3",
        s3=StorageConfig.S3Config(
            endpoint=f"http://{endpoint}",
            bucket=str(uuid.uuid4()),
            prefix=str(uuid.uuid4()),
            access_key_id="sf_test_aaa",
            access_key_secret="sf_test_sss",
            virtual_host=False,
            version="s3v4",
        ),
    )

    os.makedirs(os.path.join(minio_data_path, config.s3.bucket), exist_ok=True)

    ms = subprocess.Popen(
        [minio_server, "server", minio_data_path, "--address", endpoint],
        env=ms_env,
    )

    yield config

    ms.kill()


def test_all():
    def test_fn(config):
        remote_fn = str(uuid.uuid4())
        comp_io = ComponentStorage(config)
        # make sure ComponentStorage is always pickle-able
        comp_io = pickle.loads(pickle.dumps(comp_io))
        with tempfile.TemporaryDirectory() as wd:
            local_fn = os.path.join(wd, remote_fn)
            with open(local_fn, "w") as f:
                f.write("hello world")

            comp_io.upload_file(remote_fn, local_fn)
            os.remove(local_fn)
            comp_io.download_file(remote_fn, local_fn)
            with open(local_fn, "r") as f:
                assert f.read() == "hello world"

            logging.warning(comp_io.get_file_meta(remote_fn))

            with comp_io.get_writer(remote_fn) as w:
                w.write(b"test ")
                w.write(b"test!")
                w.write(b"hello world")
                assert w.writable()

            read_bytes = b""
            with comp_io.get_reader(remote_fn) as r:
                assert not r.writable()
                while True:
                    b = r.read(2)
                    if len(b) == 0:
                        break
                    read_bytes += b

            assert read_bytes == b"test test!hello world"

    # test s3, should_ok
    for c in build_s3_config():
        test_fn(c)
    # test local fs, should_ok
    with tempfile.TemporaryDirectory() as remote_wd:
        local_config = StorageConfig(
            type="local_fs", local_fs=StorageConfig.LocalFSConfig(wd=f"{remote_wd}")
        )
        test_fn(local_config)
