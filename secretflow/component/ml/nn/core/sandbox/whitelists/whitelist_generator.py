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
import json

__tensorflow_blacklist = {
    "_leaf": [
        "AggregationMethod",
        "Assert",
        "CriticalSection",
        "DeviceSpec",
        "GradientTape",
        "Graph",
        "RegisterGradient",
        "UnconnectedGradients",
        "autodiff",
        "autograph",
        "compat",
        "config",
        "data",
        "debugging",
        "device",
        "distribute",
        "dtensor",
        "errors",
        "estimator",
        "experimental",
        "io",
        "lite",
        "mlir",
        "profiler",
        "queue",
        "summary",
        "sysconfig",
        "test",
        "tools",
        "tpu",
        "train",
        "tsl",
        "types",
        "version",
        "xla",
        "optimizers",
        "executing_eagerly",
        "get_current_name_scope",
        "get_logger",
        "gradients",
        "graph_util",
        "import_graph_def",
        "init_scope",
        "inside_function",
        "load_library",
        "load_op_library",
        "make_ndarray",
        "make_tensor_proto",
        "lookup",
        "name_scope",
        "print",
        "py_function",
        "raw_ops",
        "recompute_grad",
        "register_tensor_conversion_function",
        "saved_model",
        "timestamp",
        "variable_creator_scope",
        "guarantee_const",
    ],
    "keras": {
        "_leaf": [
            "backend",
            "datasets",
            "dtensor",
            "estimator",
            "wrappers",
            "callbacks",
            "losses",
            "metrics",
            "optimizers",
            "preprocessing",
            "utils",
            "applications",
            "experimental",
            "mixed_precision",
        ]
    },
    "*": ["get", "serialize", "deserialize"],
}


def gen_attr_tree(obj, blacklist={}, depth=-1):
    if depth == 0 or not inspect.ismodule(obj):
        return {}

    res = {}
    if hasattr(obj, "__all__"):
        all = obj.__all__
    else:
        all = dir(obj)

    depth = depth - 1 if depth > 0 else depth
    for name in all:
        if (
            name.startswith("_")
            or name.endswith("_")
            or name in blacklist.get("_leaf", [])
            or name in blacklist.get("*", [])
        ):
            continue
        sub_blacklist = blacklist.get(name, {})
        sub_blacklist["*"] = blacklist.get("*", [])
        res[name] = gen_attr_tree(
            getattr(obj, name), blacklist=sub_blacklist, depth=depth
        )

    return res


def gen_tensorflow_whitelist():
    import tensorflow as tf

    res = gen_attr_tree(tf, __tensorflow_blacklist, 4)

    return json.dumps(res, indent=2)


def gen_secretflow_whitelist():
    from secretflow.ml.nn import applications

    res = gen_attr_tree(applications, {}, 2)

    return json.dumps(res, indent=2)
