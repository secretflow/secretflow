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

import tensorflow as tf
from keras.engine import functional


class ModelWrapper(tf.keras.Model):
    def __new__(cls, *args, **kwargs):
        from keras.engine.training import is_functional_model_init_params

        # signature detection
        if is_functional_model_init_params(args, kwargs) and cls == ModelWrapper:
            # a subclass of ModelWrapper is required, not the original Functional
            return FunctionalWrapper(skip_init=True, *args, **kwargs)
        else:
            return super(ModelWrapper, cls).__new__(cls, *args, **kwargs)

    def train_step(self, data):
        raise RuntimeError("Can not call this method.")

    def make_train_function(self, force=False):
        raise RuntimeError("Can not call this method.")

    def fit(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def test_step(self, data):
        raise RuntimeError("Can not call this method.")

    def make_test_function(self, force=False):
        raise RuntimeError("Can not call this method.")

    def evaluate(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def predict_step(self, data):
        raise RuntimeError("Can not call this method.")

    def make_predict_function(self, force=False):
        raise RuntimeError("Can not call this method.")

    def predict(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def train_on_batch(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def test_on_batch(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def predict_on_batch(self, x):
        raise RuntimeError("Can not call this method.")

    def evaluate_generator(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def predict_generator(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def save(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def save_weights(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")

    def load_weights(self, *args, **kwargs):
        raise RuntimeError("Can not call this method.")


class FunctionalWrapper(functional.Functional, ModelWrapper):
    pass


class SequentialWrapper(tf.keras.Sequential, ModelWrapper):
    pass


# replace the class with a wrapper class at runtime
tensorflow_wrapper = {
    tf.keras.Sequential: SequentialWrapper,
    tf.keras.Model: ModelWrapper,
}
