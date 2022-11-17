#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# # Copyright 2022 Ant Group Co., Ltd.
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


from typing import Callable

import tensorflow as tf


class custom_loss:
    """Decorator to define a function with a custom loss.

    This decorator allows to define loss functions with additional keyword arguments.
    These keyword arguments must match the results of model's forward pass.

    Examples:
        >>> import tensorflow as tf
        >>> # define model
        >>> class MyModel(tf.keras.Model):
        >>>     def call(self, inputs, **kwargs):
        >>>         # do forward pass
        >>>         return None, y_pred, {'kwarg1': kwarg1, 'kwarg2': kwarg2}
        >>> # define loss function
        >>> @custom_loss
        >>> def my_loss(y_true, y_pred, kwarg1 = None, kwarg2 = None):
        >>>     # cumpute loss
        >>>     pass
        >>> # compile model with custom loss function
        >>> model = MyModel(...)
        >>> model.compile(
        >>>     loss=my_loss,
        >>>     optimizer=tf.keras.optimizers.Adam(0.01),
        >>>     metrics=['acc'],
        >>> )


    Note: `custom_loss`, `my_loss` and `MyModel` need to be added to custom_objects when loading the model.
    """

    def __init__(self, func: Callable):
        self.name = func.__name__
        self.func = func
        self.kwargs = {}

    def with_kwargs(self, kwargs):
        self.kwargs = kwargs if kwargs else {}

    def __call__(self, y_true, y_pred):
        return self.func(y_true, y_pred, **self.kwargs)

    def get_config(self):
        return {
            'name': self.name,
        }

    @classmethod
    def from_config(cls, config):
        custom_objects = tf.keras.utils.get_custom_objects()
        # The object with func name has already been wrapped, so return it directly.
        return custom_objects[config['name']]
