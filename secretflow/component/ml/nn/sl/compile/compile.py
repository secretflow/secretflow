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

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import secretflow as sf
from secretflow import PYU
from secretflow.component.component import CompEvalError


@dataclass
class ModelConfig:
    model_path: Path = None
    server_fuse_path: Path = None
    server_base_path: Path = None
    client_base_path: Path = None
    loss_path: Path = None
    loss_config: str = None
    optimizer_config: Dict = None
    metrics_config: List = None


def build_model_paths(tmpdir: Path):
    model_path = tmpdir.joinpath("model")
    server_fuse_path = model_path.joinpath("server", "fuse")
    server_base_path = model_path.joinpath("server", "base")
    client_base_path = model_path.joinpath("client", "base")
    loss_path = model_path.joinpath("loss")

    return ModelConfig(
        model_path=model_path,
        server_fuse_path=server_fuse_path,
        server_base_path=server_base_path,
        client_base_path=client_base_path,
        loss_path=loss_path,
    )


def do_compile_all(
    models,
    learning_rate,
    loss,
    custom_loss,
    optimizer,
    optimizer_params,
    metrics,
    tmpdir: str,
    backend: str = "tensorflow",
    safe_only=True,
):
    tmpdir = Path(tmpdir)
    model_config = build_model_paths(tmpdir)

    if backend == "tensorflow":
        from .tensorflow.loss import compile_loss
        from .tensorflow.metric import compile_metircs
        from .tensorflow.model import compile_models
        from .tensorflow.optimizer import compile_optimizer
    else:
        raise CompEvalError(f"Unsupported backend: {backend}")

    if safe_only:
        model_config = ModelConfig()
        if loss and not custom_loss:
            model_config.loss_config = compile_loss(loss, "", None)
    else:
        compile_models(
            models,
            model_config.server_fuse_path,
            model_config.server_base_path,
            model_config.client_base_path,
        )
        model_config.loss_config = compile_loss(
            loss, custom_loss, model_config.loss_path
        )

    model_config.optimizer_config = compile_optimizer(
        optimizer, optimizer_params, learning_rate
    )
    model_config.metrics_config = compile_metircs(metrics)

    return model_config


def compile_by_self(
    parties,
    models: str,
    learning_rate: float,
    loss: str,
    custom_loss: str,
    optimizer: str,
    optimizer_params: str,
    metrics: List[str],
    backend: str = "tensorflow",
) -> Dict[PYU, ModelConfig]:
    """All parties compile the model by themselves."""

    model_configs = {}
    for party in parties:
        tmpdir = party(tempfile.mkdtemp)()
        model_configs[party] = party(do_compile_all)(
            models,
            learning_rate,
            loss,
            custom_loss,
            optimizer,
            optimizer_params,
            metrics,
            tmpdir,
            backend=backend,
            safe_only=False,
        )

    return sf.reveal(model_configs)


def compile_by_initiator(
    parties,
    initiator: PYU,
    models: str,
    learning_rate: float,
    loss: str,
    custom_loss: str,
    optimizer: str,
    optimizer_params: str,
    metrics: List[str],
    backend: str = "tensorflow",
) -> Dict[PYU, ModelConfig]:
    """The initiator party compile the model and send it to all other parties."""

    model_configs = {}
    tmpdir = initiator(tempfile.mkdtemp)()
    model_configs[initiator] = sf.reveal(
        initiator(do_compile_all)(
            models,
            learning_rate,
            loss,
            custom_loss,
            optimizer,
            optimizer_params,
            metrics,
            tmpdir,
            backend=backend,
            safe_only=False,
        )
    )

    from ..distribute import distribute, package

    def _pack(model_config: ModelConfig, tmpdir: str):
        return package.pack(model_config.model_path, tmpdir)

    src_model_package_path = initiator(_pack)(model_configs[initiator], tmpdir)

    m1_configs = {}
    m2_configs = {}

    for party in parties:
        if party == initiator:
            continue

        party_tmpdir = party(tempfile.mkdtemp)()
        m1_configs[party] = party(do_compile_all)(
            models,
            learning_rate,
            loss,
            custom_loss,
            optimizer,
            optimizer_params,
            metrics,
            party_tmpdir,
            backend=backend,
            safe_only=True,
        )

        dst_model_package_path = distribute.send(
            src_model_package_path, initiator, party
        )
        m2_configs[party] = party(package.unpack)(dst_model_package_path, party_tmpdir)

    m1_configs = sf.reveal(m1_configs)
    m2_configs = sf.reveal(m2_configs)

    for party, m1 in m1_configs.items():
        m2 = m2_configs[party]

        model_configs[party] = ModelConfig(
            # m2 part
            model_path=m2.model_path,
            server_fuse_path=m2.server_fuse_path,
            server_base_path=m2.server_base_path,
            client_base_path=m2.client_base_path,
            loss_path=m2.loss_path,
            # m1 part
            loss_config=m1.loss_config,
            optimizer_config=m1.optimizer_config,
            metrics_config=m1.metrics_config,
        )
    return model_configs
