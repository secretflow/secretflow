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


from dataclasses import dataclass
from typing import Any, Callable, List, Union

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


@dataclass
class TensorInfo:
    name: str = None
    tensor_name: str = None
    dtype: str = None
    shape: List[int] = None


def wrap_onnx_input_output(io_pb):
    from onnx.onnx_pb import TensorProto as tp

    supported_dtypes = {
        tp.DataType.FLOAT: 'float32',
        tp.DataType.INT64: 'int64',
        tp.DataType.STRING: 'string',
    }
    results = []
    for info_pb in io_pb:
        tensor_info = TensorInfo()
        tensor_info.dtype = (
            supported_dtypes[info_pb.type.tensor_type.elem_type]
            if info_pb.type.tensor_type.elem_type in supported_dtypes
            else ""
        )
        tensor_info.shape = []
        tensor_info.shape.extend(
            [
                -1 if dim.dim_param else dim.dim_value
                for dim in info_pb.type.tensor_type.shape.dim
            ]
        )
        tensor_info.name = info_pb.name
        tensor_info.tensor_name = info_pb.name
        results.append(tensor_info)
    return results


def wrap_tf_input_output(io_pb):
    from tensorflow.core.framework import types_pb2

    supported_dtypes = {
        types_pb2.DT_FLOAT: 'float32',
        types_pb2.DT_INT64: 'int64',
        types_pb2.DT_STRING: 'string',
    }
    results = []
    for name, info_pb in sorted(io_pb.items()):
        tensor_info = TensorInfo()
        tensor_info.shape = []
        tensor_info.name = name
        tensor_info.tensor_name = info_pb.name
        tensor_info.dtype = (
            supported_dtypes[info_pb.dtype] if info_pb.dtype in supported_dtypes else ""
        )
        tensor_info.shape.extend([dim.size for dim in info_pb.tensor_shape.dim])
        results.append(tensor_info)
    return results


@dataclass
class ForwardData:
    """
    ForwardData is a dataclass for data uploaded by each party to label party for computation.

    hidden: base model hidden layers outputs
    losses: the sum of base model losses should added up to fuse model loss
    """

    hidden: Union[Any, List[Any]] = None
    losses: Any = None
