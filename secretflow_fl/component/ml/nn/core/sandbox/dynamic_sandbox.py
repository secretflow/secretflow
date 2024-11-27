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

import inspect
from numbers import Number
from typing import Dict


class RuntimeModuleContainer(object):
    def __init__(self, obj, whitelist: Dict, object_wrapper: Dict):
        self.obj = obj
        self.whitelist = whitelist
        self.object_wrapper = object_wrapper

    def __getattribute__(self, name: str):
        obj = object.__getattribute__(self, "obj")
        whitelist = object.__getattribute__(self, "whitelist")
        object_wrapper = object.__getattribute__(self, "object_wrapper")
        if name not in whitelist:
            raise AttributeError(f"{name} not exists")

        attr = getattr(obj, name)
        if inspect.ismodule(attr):
            return RuntimeModuleContainer(attr, whitelist[name], object_wrapper)
        elif not whitelist[name]:
            if attr in object_wrapper:
                return object_wrapper[attr]
            else:
                return attr
        else:
            raise AttributeError(name)


def runtime_module_container(obj, whitelist: Dict, object_wrapper: Dict):
    return RuntimeModuleContainer(
        obj, whitelist=whitelist, object_wrapper=object_wrapper
    )


class _SubscriptWrapper(object):
    def __init__(self, getter_setter):
        super().__init__()
        self.getter_setter = getter_setter

    def _check_slice_key(self, key):
        if key is None or isinstance(key, Number):
            return

        elif isinstance(key, tuple):
            for v in key:
                self._check_slice_key(v)

        elif isinstance(key, slice):
            for v in [key.start, key.stop, key.step]:
                self._check_slice_key(v)
        elif isinstance(key, str):
            raise ValueError("string slice of subscript is not allowed")

        else:
            try:
                import tensorflow as tf

                if isinstance(key, tf.DType) and not isinstance(key, tf.string):
                    return
            except Exception:
                pass

            try:
                import torch

                if isinstance(key, torch.dtype):
                    return
            except Exception:
                pass

            raise ValueError("object slice of subscript is not allowed")


def _wrap_getitem(obj):
    cls = type(obj)
    orig_getitem = cls.__getitem__
    if getattr(orig_getitem, "__name__", "") == "__subscript_wrapper__":
        return

    class _Wrapper(_SubscriptWrapper):
        def __call__(self, _self, key):
            item = self.getter_setter(_self, key)
            if inspect.ismodule(item) or callable(item):
                self._check_slice_key(key)

            return item

    cls.__getitem__ = lambda self, key: _Wrapper(orig_getitem)(self, key)
    cls.__getitem__.__name__ = "__subscript_wrapper__"


def _wrap_setitem(obj):
    cls = type(obj)
    orig_setitem = cls.__setitem__
    if getattr(orig_setitem, "__name__", "") == "__subscript_wrapper__":
        return

    class _Wrapper(_SubscriptWrapper):
        def __call__(self, _self, key, value):
            if inspect.ismodule(value) or callable(value):
                self._check_slice_key(key)
            self.getter_setter(_self, key, value)

    cls.__setitem__ = lambda self, key, value: _Wrapper(orig_setitem)(self, key, value)
    cls.__setitem__.__name__ = "__subscript_wrapper__"


def subscript_wrapper(obj):
    if obj is None:
        return obj

    if hasattr(obj, "__getitem__"):
        _wrap_getitem(obj)

    if hasattr(obj, "__setitem__"):
        _wrap_setitem(obj)

    return obj
