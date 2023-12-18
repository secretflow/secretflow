# Copyright 2023 Ant Group Co., Ltd.
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

import numpy as np
import pyarrow.compute as pc

from .tracer import _Tracer, _TracerType, Array


def __sf_wrapper(py_func, c_func):
    arity = c_func.arity
    py_options_class = pc._get_options_class(c_func)
    py_func_name = py_func.__name__

    def get_py_options(*args, options=None, **kwargs):
        if py_options_class is None:
            py_options = None
        else:
            if arity is not Ellipsis:
                py_option_args = args[arity:]
            else:
                py_option_args = ()
            py_options = pc._handle_options(
                py_func_name, py_options_class, options, py_option_args, kwargs
            )

        if py_options:
            py_options = py_options.serialize().to_pybytes()

        return py_options

    def wrapper(*args, **kwargs):
        if arity is not Ellipsis:
            inputs = args[:arity]
            remain = args[arity:]
        else:
            inputs = args
            remain = []

        assert all(
            [
                isinstance(i, (Array, int, float, str, np.floating, np.integer))
                for i in inputs
            ]
        )
        assert any([isinstance(i, Array) for i in inputs])

        py_inputs = []
        tracer_inputs = []
        for i in inputs:
            if isinstance(i, Array):
                py_inputs.append(i._arrow)
                tracer_inputs.append(i._trace)
            else:
                py_inputs.append(i)
                tracer_inputs.append(i)

        py_inputs.extend(remain)
        arrow = py_func(*py_inputs, **kwargs)

        py_options = get_py_options(*args, **kwargs)
        tracer = _Tracer(
            c_func.name,
            output_type=_TracerType.ARROW,
            inputs=tracer_inputs,
            py_kwargs=kwargs,
            py_args=remain,
            options=py_options,
        )

        return Array(arrow, tracer)

    # copy attrs
    wrapper.__name__ = py_func.__name__
    wrapper.__qualname__ = py_func.__name__
    wrapper.__doc__ = py_func.__doc__
    if hasattr(py_func, "__signature__"):
        wrapper.__signature__ = py_func.__signature__
    return wrapper


def _gen_sf_funcs():
    reg = pc.function_registry()
    sf_funcs = dict()
    for name in reg.list_functions():
        c_func = reg.get_function(name)
        if c_func.kind != "scalar":
            # export scalar function only
            continue
        py_func = getattr(pc, name)
        sf_func = __sf_wrapper(py_func, c_func)
        sf_funcs[name] = sf_func
        if py_func.__name__ != name:
            sf_funcs[py_func.__name__] = sf_func

    return sf_funcs
