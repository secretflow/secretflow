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

import json
from typing import List

from secretflow.component.component import CompEvalContext, CompEvalError
from secretflow.component.ml.nn.sl.compile.compile import (
    compile_by_initiator,
    compile_by_self,
)
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU
from secretflow.ml.nn.sl.sl_model import SLModel
from secretflow.utils import compressor as sfcompressor


def prepare_pyus(x: VDataFrame, y: VDataFrame):
    parties = set(x.partitions.keys())
    parties.update(y.partitions.keys())
    # ensure all parties have save order
    parties = sorted(list(parties))
    label_pyu = next(iter(y.partitions))
    return parties, label_pyu


def fit(
    ctx: CompEvalContext,
    x: VDataFrame,
    y: VDataFrame,
    val_x: VDataFrame,
    val_y: VDataFrame,
    models: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    loss: str,
    custom_loss: str,
    optimizer: str,
    optimizer_params: str,
    metrics: List[str],
    model_input_scheme: str,
    strategy: str,
    strategy_params: str,
    compressor: str,
    compressor_params: str,
    backend: str = "tensorflow",
):
    if backend == "tensorflow":
        from .tensorflow.data import create_dataset_builder
        from .tensorflow.model import create_model_builder
    else:
        raise CompEvalError(f"Unsupported backend: {backend}")

    parties, label_pyu = prepare_pyus(x, y)
    initiator = None
    if ctx.initiator_party:
        initiator = PYU(ctx.initiator_party)

    if initiator:
        model_configs = compile_by_initiator(
            parties,
            initiator,
            models,
            learning_rate,
            loss,
            custom_loss,
            optimizer,
            optimizer_params,
            metrics,
            backend=backend,
        )
    else:
        model_configs = compile_by_self(
            parties,
            models,
            learning_rate,
            loss,
            custom_loss,
            optimizer,
            optimizer_params,
            metrics,
            backend=backend,
        )

    base_model_dict = {}
    server_fuse_builder = None
    for pyu, config in model_configs.items():
        if pyu == label_pyu:
            base_model_dict[pyu] = create_model_builder(config.server_base_path, config)

            assert config.server_fuse_path is not None
            server_fuse_builder = create_model_builder(config.server_fuse_path, config)
        else:
            assert config.client_base_path is not None
            base_model_dict[pyu] = create_model_builder(config.client_base_path, config)

    label_name = y.columns[0]
    dataset_builder = create_dataset_builder(
        pyus=parties,
        label_pyu=label_pyu,
        label_name=label_name,
        model_input_scheme=model_input_scheme,
        batch_size=batch_size,
        epochs=epochs,
    )

    strategy_dict = {}
    if strategy_params:
        strategy_dict = dict(json.loads(strategy_params))

    compressor_obj = None
    if compressor:
        compressor_dict = {}
        if compressor_params:
            compressor_dict = json.loads(compressor_params)
        compressor_obj = sfcompressor.get(str(compressor).strip(), compressor_dict)

    slmodel = SLModel(
        base_model_dict=base_model_dict,
        device_y=label_pyu,
        model_fuse=server_fuse_builder,
        random_seed=100,
        compressor=compressor_obj,
        strategy=strategy,
        **strategy_dict,
    )

    history = slmodel.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_x, val_y) if val_x and val_y else None,
        random_seed=100,
        dataset_builder=dataset_builder,
    )

    return slmodel, history, model_configs
