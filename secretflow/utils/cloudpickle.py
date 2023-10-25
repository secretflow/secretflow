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

"""
This file mainly borrows from ray.cloudpickle.cloudpickle_fast.py.

We introduce a custom serialization omitting code positions,
e.g., function filename, first line no, .etc.

It is useful for some scenarios that need to generate the signature of
a function deterministicly and is used to sign python functions to execute
in TEE.
"""


import copyreg
import io
import logging
import pickle
import sys
import types
import typing
import weakref
from collections import ChainMap, OrderedDict
from enum import Enum

import _collections_abc
from ray.cloudpickle import Pickler, cloudpickle, cloudpickle_fast


def _code_reduce(obj):
    """A custom codeobject reducer.

    This reducer omits the code filename and firstlineno.
    """
    if hasattr(obj, "co_linetable"):  # pragma: no branch
        # Python 3.10 and later: obj.co_lnotab is deprecated and constructor
        # expects obj.co_linetable instead.
        args = (
            obj.co_argcount,
            obj.co_posonlyargcount,
            obj.co_kwonlyargcount,
            obj.co_nlocals,
            obj.co_stacksize,
            obj.co_flags,
            obj.co_code,
            obj.co_consts,
            obj.co_names,
            obj.co_varnames,
            obj.co_name,
            obj.co_linetable,
            obj.co_freevars,
            obj.co_cellvars,
        )
    elif hasattr(obj, "co_posonlyargcount"):
        # Backward compat for 3.9 and older
        args = (
            obj.co_argcount,
            obj.co_posonlyargcount,
            obj.co_kwonlyargcount,
            obj.co_nlocals,
            obj.co_stacksize,
            obj.co_flags,
            obj.co_code,
            obj.co_consts,
            obj.co_names,
            obj.co_varnames,
            obj.co_name,
            obj.co_lnotab,
            obj.co_freevars,
            obj.co_cellvars,
        )
    else:
        # Backward compat for even older versions of Python
        args = (
            obj.co_argcount,
            obj.co_kwonlyargcount,
            obj.co_nlocals,
            obj.co_stacksize,
            obj.co_flags,
            obj.co_code,
            obj.co_consts,
            obj.co_names,
            obj.co_varnames,
            obj.co_name,
            obj.co_lnotab,
            obj.co_freevars,
            obj.co_cellvars,
        )
    return types.CodeType, args


'''borrow from cloudpickle.py
The cloudpickle library utilizes random UUID as the class_tracker_id,
leading to a failure in decrypting the data with the serialization of a class in TEEU.
To solve this problem, we replace it with '00000000000000000000000000000000'.
'''


def _get_or_create_tracker_id(class_def):
    return '00000000000000000000000000000000'


def _decompose_typevar(obj):
    return (
        obj.__name__,
        obj.__bound__,
        obj.__constraints__,
        obj.__covariant__,
        obj.__contravariant__,
        _get_or_create_tracker_id(obj),
    )


def _typevar_reduce(obj):
    # TypeVar instances require the module information hence why we
    # are not using the _should_pickle_by_reference directly
    module_and_name = cloudpickle._lookup_module_and_qualname(obj, name=obj.__name__)

    if module_and_name is None:
        return (cloudpickle._make_typevar, _decompose_typevar(obj))
    elif cloudpickle._is_registered_pickle_by_value(module_and_name[0]):
        return (cloudpickle._make_typevar, _decompose_typevar(obj))

    return (getattr, module_and_name)


'''borrow from cloudpickle_fast.py'''


def _class_getnewargs(obj):
    type_kwargs = {}
    if "__module__" in obj.__dict__:
        type_kwargs["__module__"] = obj.__module__

    __dict__ = obj.__dict__.get('__dict__', None)
    if isinstance(__dict__, property):
        type_kwargs['__dict__'] = __dict__

    return (
        type(obj),
        obj.__name__,
        cloudpickle_fast._get_bases(obj),
        type_kwargs,
        _get_or_create_tracker_id(obj),
        None,
    )


def _enum_getnewargs(obj):
    members = {e.name: e.value for e in obj}
    return (
        obj.__bases__,
        obj.__name__,
        obj.__qualname__,
        members,
        obj.__module__,
        _get_or_create_tracker_id(obj),
        None,
    )


def _dynamic_class_reduce(obj):
    """
    Save a class that can't be stored as module global.

    This method is used to serialize classes that are defined inside
    functions, or that otherwise can't be serialized as attribute lookups
    from global modules.
    """
    if Enum is not None and issubclass(obj, Enum):
        return (
            cloudpickle_fast._make_skeleton_enum,
            _enum_getnewargs(obj),
            cloudpickle_fast._enum_getstate(obj),
            None,
            None,
            cloudpickle_fast._class_setstate,
        )
    else:
        return (
            cloudpickle_fast._make_skeleton_class,
            _class_getnewargs(obj),
            cloudpickle_fast._class_getstate(obj),
            None,
            None,
            cloudpickle_fast._class_setstate,
        )


def _class_reduce(obj):
    """Select the reducer depending on the dynamic nature of the class obj"""
    if obj is type(None):  # noqa
        return type, (None,)
    elif obj is type(Ellipsis):
        return type, (Ellipsis,)
    elif obj is type(NotImplemented):
        return type, (NotImplemented,)
    elif obj in cloudpickle_fast._BUILTIN_TYPE_NAMES:
        return cloudpickle_fast._builtin_type, (
            cloudpickle_fast._BUILTIN_TYPE_NAMES[obj],
        )
    elif not cloudpickle_fast._should_pickle_by_reference(obj):
        return _dynamic_class_reduce(obj)
    return NotImplemented


class CodePositionIndependentCloudPickler(Pickler):
    _dispatch_table = {}
    _dispatch_table[classmethod] = cloudpickle_fast._classmethod_reduce
    _dispatch_table[io.TextIOWrapper] = cloudpickle_fast._file_reduce
    _dispatch_table[logging.Logger] = cloudpickle_fast._logger_reduce
    _dispatch_table[logging.RootLogger] = cloudpickle_fast._root_logger_reduce
    _dispatch_table[memoryview] = cloudpickle_fast._memoryview_reduce
    _dispatch_table[property] = cloudpickle_fast._property_reduce
    _dispatch_table[staticmethod] = cloudpickle_fast._classmethod_reduce
    _dispatch_table[types.CellType] = cloudpickle_fast._cell_reduce
    # Use custom code reducer.
    _dispatch_table[types.CodeType] = _code_reduce
    _dispatch_table[
        types.GetSetDescriptorType
    ] = cloudpickle_fast._getset_descriptor_reduce
    _dispatch_table[types.ModuleType] = cloudpickle_fast._module_reduce
    _dispatch_table[types.MethodType] = cloudpickle_fast._method_reduce
    _dispatch_table[types.MappingProxyType] = cloudpickle_fast._mappingproxy_reduce
    _dispatch_table[weakref.WeakSet] = cloudpickle_fast._weakset_reduce
    _dispatch_table[typing.TypeVar] = _typevar_reduce
    _dispatch_table[_collections_abc.dict_keys] = cloudpickle_fast._dict_keys_reduce
    _dispatch_table[_collections_abc.dict_values] = cloudpickle_fast._dict_values_reduce
    _dispatch_table[_collections_abc.dict_items] = cloudpickle_fast._dict_items_reduce
    _dispatch_table[type(OrderedDict().keys())] = cloudpickle_fast._odict_keys_reduce
    _dispatch_table[
        type(OrderedDict().values())
    ] = cloudpickle_fast._odict_values_reduce
    _dispatch_table[type(OrderedDict().items())] = cloudpickle_fast._odict_items_reduce

    dispatch_table = ChainMap(_dispatch_table, copyreg.dispatch_table)

    # function reducers are defined as instance methods of CloudPickler
    # objects, as they rely on a CloudPickler attribute (globals_ref)
    def _dynamic_function_reduce(self, func):
        """Reduce a function that is not pickleable via attribute lookup."""
        newargs = self._function_getnewargs(func)
        state = cloudpickle_fast._function_getstate(func)
        import time

        time.sleep(1)
        return (
            types.FunctionType,
            newargs,
            state,
            None,
            None,
            cloudpickle_fast._function_setstate,
        )

    def _function_reduce(self, obj):
        """Reducer for function objects.

        If obj is a top-level attribute of a file-backed module, this
        reducer returns NotImplemented, making the CloudPickler fallback to
        traditional _pickle.Pickler routines to save obj. Otherwise, it reduces
        obj using a custom cloudpickle reducer designed specifically to handle
        dynamic functions.

        As opposed to cloudpickle.py, There no special handling for builtin
        pypy functions because cloudpickle_fast is CPython-specific.
        """
        if cloudpickle_fast._should_pickle_by_reference(obj):
            return NotImplemented
        else:
            return self._dynamic_function_reduce(obj)

    def _function_getnewargs(self, func):
        code = func.__code__

        # base_globals represents the future global namespace of func at
        # unpickling time. Looking it up and storing it in
        # CloudpiPickler.globals_ref allow functions sharing the same globals
        # at pickling time to also share them once unpickled, at one condition:
        # since globals_ref is an attribute of a CloudPickler instance, and
        # that a new CloudPickler is created each time pickle.dump or
        # pickle.dumps is called, functions also need to be saved within the
        # same invocation of cloudpickle.dump/cloudpickle.dumps (for example:
        # cloudpickle.dumps([f1, f2])). There is no such limitation when using
        # CloudPickler.dump, as long as the multiple invocations are bound to
        # the same CloudPickler.
        base_globals = self.globals_ref.setdefault(id(func.__globals__), {})

        if base_globals == {}:
            # Add module attributes used to resolve relative imports
            # instructions inside func.
            for k in ["__package__", "__name__", "__path__", "__file__"]:
                if k in func.__globals__:
                    base_globals[k] = func.__globals__[k]

        # Omit file and path info.
        base_globals['__file__'] = ''
        base_globals['__path__'] = ''

        # Do not bind the free variables before the function is created to
        # avoid infinite recursion.
        if func.__closure__ is None:
            closure = None
        else:
            closure = tuple(
                cloudpickle_fast._make_empty_cell()
                for _ in range(len(code.co_freevars))
            )

        return code, base_globals, None, None, closure

    def dump(self, obj):
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if "recursion" in e.args[0]:
                msg = (
                    "Could not pickle object as excessively deep recursion " "required."
                )
                raise pickle.PicklingError(msg) from e
            else:
                raise

    if pickle.HIGHEST_PROTOCOL >= 5:
        # `CloudPickler.dispatch` is only left for backward compatibility - note
        # that when using protocol 5, `CloudPickler.dispatch` is not an
        # extension of `Pickler.dispatch` dictionary, because CloudPickler
        # subclasses the C-implemented Pickler, which does not expose a
        # `dispatch` attribute.  Earlier versions of the protocol 5 CloudPickler
        # used `CloudPickler.dispatch` as a class-level attribute storing all
        # reducers implemented by cloudpickle, but the attribute name was not a
        # great choice given the meaning of `CloudPickler.dispatch` when
        # `CloudPickler` extends the pure-python pickler.
        # dispatch = dispatch_table

        # Implementation of the reducer_override callback, in order to
        # efficiently serialize dynamic functions and classes by subclassing
        # the C-implemented Pickler.
        # TODO: decorrelate reducer_override (which is tied to CPython's
        # implementation - would it make sense to backport it to pypy? - and
        # pickle's protocol 5 which is implementation agnostic. Currently, the
        # availability of both notions coincide on CPython's pickle and the
        # pickle5 backport, but it may not be the case anymore when pypy
        # implements protocol 5
        def __init__(self, file, protocol=None, buffer_callback=None):
            if protocol is None:
                protocol = cloudpickle_fast.DEFAULT_PROTOCOL
            Pickler.__init__(
                self, file, protocol=protocol, buffer_callback=buffer_callback
            )
            # map functions __globals__ attribute ids, to ensure that functions
            # sharing the same global namespace at pickling time also share
            # their global namespace at unpickling time.
            self.globals_ref = {}
            self.proto = int(protocol)

        def reducer_override(self, obj):
            """Type-agnostic reducing callback for function and classes.

            For performance reasons, subclasses of the C _pickle.Pickler class
            cannot register custom reducers for functions and classes in the
            dispatch_table. Reducer for such types must instead implemented in
            the special reducer_override method.

            Note that method will be called for any object except a few
            builtin-types (int, lists, dicts etc.), which differs from reducers
            in the Pickler's dispatch_table, each of them being invoked for
            objects of a specific type only.

            This property comes in handy for classes: although most classes are
            instances of the ``type`` metaclass, some of them can be instances
            of other custom metaclasses (such as enum.EnumMeta for example). In
            particular, the metaclass will likely not be known in advance, and
            thus cannot be special-cased using an entry in the dispatch_table.
            reducer_override, among other things, allows us to register a
            reducer that will be called for any class, independently of its
            type.


            Notes:

            * reducer_override has the priority over dispatch_table-registered
            reducers.
            * reducer_override can be used to fix other limitations of
              cloudpickle for other types that suffered from type-specific
              reducers, such as Exceptions. See
              https://github.com/cloudpipe/cloudpickle/issues/248
            """
            if sys.version_info[:2] < (
                3,
                7,
            ) and cloudpickle_fast._is_parametrized_type_hint(
                obj
            ):  # noqa  # pragma: no branch
                try:
                    return (
                        cloudpickle_fast._create_parametrized_type_hint,
                        cloudpickle_fast.parametrized_type_hint_getinitargs(obj),
                    )
                except pickle.PicklingError:
                    # There are some false positive cases in '_is_parametrized_type_hint'.
                    # We should not fail early for these false positive cases.
                    pass
            t = type(obj)
            try:
                is_anyclass = issubclass(t, type)
            except TypeError:  # t is not a class (old Boost; see SF #502085)
                is_anyclass = False

            if is_anyclass:
                return _class_reduce(obj)
            elif isinstance(obj, types.FunctionType):
                return self._function_reduce(obj)
            else:
                # fallback to save_global, including the Pickler's
                # dispatch_table
                return NotImplemented


def code_position_independent_dumps(obj, protocol=None):
    """Serialize obj as a string of bytes allocated in memory.

    Note that the serialization omits the code position info such as code
    filename, firstlineno, etc.
    """
    with io.BytesIO() as file:
        cp = CodePositionIndependentCloudPickler(file, protocol=protocol)
        cp.dump(obj)
        return file.getvalue()
