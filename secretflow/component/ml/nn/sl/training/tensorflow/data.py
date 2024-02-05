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

from typing import List

from secretflow import PYU
from secretflow.component.ml.nn.sl.base import ModelInputScheme


def _create_dataset_builder_without_label(batch_size: int = 256, epochs: int = 1):
    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        x = [dict(t) if isinstance(t, pd.DataFrame) else t for t in x]
        x = x[0] if len(x) == 1 else tuple(x)
        data_set = (
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(epochs)
        )

        return data_set

    return dataset_builder


def _create_dataset_builder_with_label(
    label_name: str, batch_size: int = 256, epochs: int = 1
):
    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        def _parse_label(row_sample, label):
            y_t = label[label_name]
            y = tf.expand_dims(y_t, axis=1)
            return row_sample, y

        x = [dict(t) if isinstance(t, pd.DataFrame) else t for t in x]
        x = x[0] if len(x) == 1 else tuple(x)
        data_set = (
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(epochs)
        )

        data_set = data_set.map(_parse_label)

        return data_set

    return dataset_builder


def create_dataset_builder(
    pyus: List[PYU],
    label_pyu: PYU = None,
    label_name: str = None,
    model_input_scheme: str = ModelInputScheme.TENSOR,
    batch_size: int = 256,
    epochs: int = 1,
):
    if model_input_scheme == ModelInputScheme.TENSOR:
        return None
    elif model_input_scheme == ModelInputScheme.TENSOR_DICT:
        builders = {}
        for pyu in pyus:
            if pyu == label_pyu:
                builders[pyu] = _create_dataset_builder_with_label(
                    label_name,
                    batch_size=batch_size,
                    epochs=epochs,
                )
            else:
                builders[pyu] = _create_dataset_builder_without_label(
                    batch_size=batch_size,
                    epochs=epochs,
                )
        return builders
    else:
        raise ValueError(f"Model input scheme not supported: {model_input_scheme}")
