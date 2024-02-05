# Copyright 2024 Ant Group Co., Ltd.
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

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import secretflow as sf
from secretflow.component.ml.nn.sl.compile.compile import ModelConfig, build_model_paths
from secretflow.component.ml.nn.sl.distribute import package
from secretflow.device import PYU, PYUObject
from secretflow.ml.nn import SLModel


def save(
    slmodel: SLModel,
    label_pyu: PYU,
    model_configs: Dict[PYU, ModelConfig],
    tmpdirs: Dict[PYU, str],
) -> Tuple[List[PYUObject], List[Dict]]:
    tmp_configs = {pyu: build_model_paths(Path(path)) for pyu, path in tmpdirs.items()}
    base_model_path = {}
    fuse_model_path = ""
    meta_dict = {}
    for pyu, config in tmp_configs.items():
        meta = {
            "base": False,
            "fuse": False,
        }
        if pyu == label_pyu:
            base_model_path[pyu] = str(config.server_base_path)
            fuse_model_path = str(config.server_fuse_path)
            meta["fuse"] = True
            if slmodel.base_model_dict.get(pyu, None) is not None:
                meta["base"] = True
        else:
            base_model_path[pyu] = str(config.client_base_path)
            meta["base"] = True
        meta_dict[pyu] = meta

    # save models
    slmodel.save_model(
        base_model_path=base_model_path,
        fuse_model_path=fuse_model_path,
    )

    # save loss
    def _copy_to(src, dst):
        if not src:
            return
        if Path(src).is_dir():
            Path(dst).mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    wait_copy = []
    for pyu, tmp_config in tmp_configs.items():
        config = model_configs[pyu]
        wait_copy.append(pyu(_copy_to)(config.loss_path, tmp_config.loss_path))

    sf.wait(wait_copy)

    package_paths = []
    for pyu, config in tmp_configs.items():
        package_path = pyu(package.pack)(
            config.model_path, Path(tmpdirs[pyu]).joinpath("package")
        )
        package_paths.append(package_path)

    sf.wait(package_paths)

    def _read_bytes(path: Path):
        with open(path, "rb") as f:
            return f.read()

    content_list = []
    meta_list = []
    for package_path in package_paths:
        pyu = package_path.device
        package_content = pyu(_read_bytes)(package_path)
        content_list.append(package_content)

        config = model_configs[pyu]
        meta_dict[pyu]["loss_config"] = config.loss_config
        meta_dict[pyu]["optimizer_config"] = config.optimizer_config
        meta_dict[pyu]["metrics_config"] = config.metrics_config

        meta_list.append(meta_dict[pyu])

    return content_list, meta_list


def load(
    models: List[PYUObject], meta_list: List[Dict[str, bool]], tmpdirs: Dict[PYU, str]
) -> Dict[PYU, ModelConfig]:
    def _save_bytes(content: bytes, path: Path):
        path = path.joinpath("model_package.tar.gz")
        with open(path, "wb") as f:
            f.write(content)

        return path

    result: Dict[PYU, ModelConfig] = {}
    for idx, content in enumerate(models):
        meta = meta_list[idx]
        pyu = content.device
        model_path = pyu(_save_bytes)(content, Path(tmpdirs[pyu]))
        model_config: ModelConfig = sf.reveal(
            pyu(package.unpack)(model_path, tmpdirs[pyu])
        )
        if meta.get("fuse", False):
            result[pyu] = ModelConfig(
                server_fuse_path=model_config.server_fuse_path,
                server_base_path=model_config.server_base_path
                if meta.get("base", False)
                else None,
            )
        else:
            result[pyu] = ModelConfig(client_base_path=model_config.client_base_path)

        result[pyu].loss_path = model_config.loss_path
        result[pyu].loss_config = meta.get("loss_config", None)
        result[pyu].optimizer_config = meta.get("optimizer_config", None)
        result[pyu].metrics_config = meta.get("metrics_config", None)

    return result
