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

from typing import Dict

from secretflow.component.component import CompEvalContext, CompEvalError
from secretflow.component.ml.nn.sl.compile.compile import ModelConfig
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU
from secretflow.ml.nn.sl.sl_model import SLModel


def predict(
    ctx: CompEvalContext,
    batch_size: int,
    feature_dataset: VDataFrame,
    model: Dict[PYU, ModelConfig],
    model_input_scheme: str,
    backend: str = "tensorflow",
) -> FedNdarray:
    if backend == "tensorflow":
        from .tensorflow.data import create_dataset_builder
        from .tensorflow.model import create_model_builder
    else:
        raise CompEvalError(f"Unsupported backend: {backend}")

    base_model_dict = {}
    server_fuse_builder = None
    device_y = None
    for pyu, config in model.items():
        if config.server_fuse_path is not None:
            device_y = pyu
        if pyu == device_y:
            base_model_dict[pyu] = create_model_builder(config.server_base_path, config)

            assert config.server_fuse_path is not None
            server_fuse_builder = create_model_builder(config.server_fuse_path, config)
        else:
            assert config.client_base_path is not None
            base_model_dict[pyu] = create_model_builder(config.client_base_path, config)

    slmodel = SLModel(
        base_model_dict=base_model_dict,
        model_fuse=server_fuse_builder,
        device_y=device_y,
    )

    dataset_builder = create_dataset_builder(
        model.keys(),
        model_input_scheme=model_input_scheme,
        batch_size=batch_size,
    )

    result = slmodel.predict(
        feature_dataset,
        batch_size=batch_size,
        dataset_builder=dataset_builder,
    )

    def _combine(pred):
        import numpy as np

        return np.concatenate(pred, axis=0)

    pred_y = device_y(_combine)(result)

    return FedNdarray(
        partitions={device_y: pred_y},
        partition_way=PartitionWay.VERTICAL,
    )
