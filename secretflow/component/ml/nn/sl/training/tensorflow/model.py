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

from pathlib import Path

from secretflow.component.ml.nn.sl.compile.compile import ModelConfig
from secretflow.component.ml.nn.sl.compile.tensorflow.metric import get_metrics
from secretflow.component.ml.nn.sl.compile.tensorflow.optimizer import get_optimizer


def create_model_builder(model_path: Path, model_config: ModelConfig):
    if model_path is None:
        return None

    def model_builder():
        import tensorflow as tf

        model = tf.saved_model.load(model_path)
        if model_config.loss_config:
            loss_func = model_config.loss_config
        else:
            loss_func = tf.saved_model.load(model_config.loss_path)
        metrics = get_metrics(model_config.metrics_config)
        optimizer = get_optimizer(model_config.optimizer_config)

        class WrapperModel(tf.keras.Model):
            def __init__(self, loaded):
                super().__init__()
                self.loaded = loaded
                self._trainable_weights = []
                for t in loaded.trainable_variables:
                    self._trainable_weights.append(t)

            def call(self, x):
                return self.loaded(x, training=True)

            def output_num(self):
                return len(self.loaded.signatures["serving_default"].outputs)

            def save(self, filepath, signatures=None, options=None, **kwargs):
                tf.saved_model.save(
                    self.loaded, filepath, signatures=signatures, options=options
                )

        wrapper = WrapperModel(model)
        wrapper.compile(
            loss=loss_func,
            optimizer=optimizer,
            metrics=metrics,
        )

        assert len(model.__call__.concrete_functions) > 0
        concrete_function = model.__call__.concrete_functions[0]

        assert (
            len(concrete_function.structured_input_signature) > 0
            and len(concrete_function.structured_input_signature[0]) > 0
        )

        import jax

        input_spec = concrete_function.structured_input_signature[0][0]
        input_shape = jax.tree_map(
            lambda spec: spec.shape,
            input_spec,
            is_leaf=lambda spec: isinstance(spec, tf.TensorSpec),
        )

        wrapper.build(input_shape)

        return wrapper

    return model_builder
