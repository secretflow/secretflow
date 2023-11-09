import pyarrow.compute as pc

import numpy as np
from .tracer import Array, Table, _Tracer, _TracerType

__all__ = ["Array", "Table"]


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
            py_options = py_options.serialize().to_pybytes()

        return py_options

    def wrapper(*args, **kwargs):
        if arity is not Ellipsis:
            inputs = args[:arity]
            remain = args[arity:]
        else:
            inputs = arity
            remain = []

        assert all(
            [
                isinstance(i, (Array, int, float, str, np.floating, np.integer))
                for i in inputs
            ]
        ), f"..{inputs}.. {[type(i) for i in inputs]}"
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


def __init_pyarrow_func():
    reg = pc.function_registry()
    g = globals()
    for name in reg.list_functions():
        c_func = reg.get_function(name)
        if c_func.kind != "scalar":
            # export scalar function only
            continue
        py_func = getattr(pc, name)
        sf_func = __sf_wrapper(py_func, c_func)
        g[name] = sf_func
        if py_func.__name__ != name:
            g[py_func.__name__] = sf_func


__init_pyarrow_func()
