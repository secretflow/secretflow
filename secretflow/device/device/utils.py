# Copyright 2022 Ant Group Co., Ltd.
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

import jax
import ray


def check_num_returns(fn):
    # inspect.signature fails on some builtin method (e.g. numpy.random.rand).
    # You can wrap a self define function which calls builtin function inside
    # with return annotation to get multi returns for now.
    if inspect.isbuiltin(fn):
        sig = inspect.signature(lambda *arg, **kwargs: fn(*arg, **kwargs))
    else:
        sig = inspect.signature(fn)

    if sig.return_annotation is None or sig.return_annotation == sig.empty:
        num_returns = 1
    else:
        if (
            hasattr(sig.return_annotation, '_name')
            and sig.return_annotation._name == 'Tuple'
        ):
            num_returns = len(sig.return_annotation.__args__)
        elif isinstance(sig.return_annotation, tuple):
            num_returns = len(sig.return_annotation)
        else:
            num_returns = 1

    return num_returns


def flatten(args, kwargs):
    arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
    refs = {
        pos: arg for pos, arg in enumerate(arg_flat) if isinstance(arg, ray.ObjectRef)
    }

    actual_vals = ray.get(list(refs.values()))
    for pos, actual_val in zip(refs.keys(), actual_vals):
        arg_flat[pos] = actual_val

    return jax.tree_util.tree_unflatten(arg_tree, arg_flat)
