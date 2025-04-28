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


import jax

from . import fed as sf_fed

# Put all ray related code here and use lazy import to
# avoid import ray in lite version.


def resolve_args(*args, **kwargs):
    try:
        # lazy import ray
        import ray
    except ImportError:
        return args, kwargs
    else:
        arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
        refs = {
            pos: arg
            for pos, arg in enumerate(arg_flat)
            if isinstance(arg, ray.ObjectRef)
        }

        if not refs:
            return args, kwargs

        actual_vals = ray.get(list(refs.values()))
        for pos, actual_val in zip(refs.keys(), actual_vals):
            arg_flat[pos] = actual_val
        args, kwargs = jax.tree_util.tree_unflatten(arg_tree, arg_flat)

        return args, kwargs


def resolve_arg_flat_tree(args, kwargs):
    arg_flat, arg_tree = jax.tree_util.tree_flatten((args, kwargs))

    try:
        # lazy import ray
        import ray
    except ImportError:
        return arg_flat, arg_tree
    else:
        refs = {
            pos: arg
            for pos, arg in enumerate(arg_flat)
            if isinstance(arg, ray.ObjectRef)
        }
        if not refs:
            return arg_flat, arg_tree

        actual_vals = ray.get(list(refs.values()))
        for pos, actual_val in zip(refs.keys(), actual_vals):
            arg_flat[pos] = actual_val

        return arg_flat, arg_tree


def get_obj_ref(x):
    try:
        # lazy import ray
        import ray
    except ImportError:
        return x
    else:
        return ray.get(x) if isinstance(x, ray.ObjectRef) else x


def assert_is_fed_obj(x):
    try:
        # lazy import ray
        import ray
        import fed as rayfed
    except ImportError:
        assert isinstance(
            x, sf_fed.FedObject
        ), f"shares_name in spu obj should be FedObject, but got {type(x)} "
    else:
        assert isinstance(
            x, (ray.ObjectRef, sf_fed.FedObject, rayfed.FedObject)
        ), f"shares_name in spu obj should be ObjectRef or FedObject, but got {type(x)} "
