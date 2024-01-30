import pytest

from secretflow.component.ml.nn.core.sandbox import (
    dynamic_sandbox,
    runner,
    static_sandbox,
)
from secretflow.component.ml.nn.core.sandbox.whitelists import (
    applications,
    tensorflow_wrapper,
)

from .model_def import MODELS_CODE


def test_check_and_transform():
    # builtin black list
    code = "exec"
    with pytest.raises(SyntaxError, match="reserved exec is not allowed"):
        static_sandbox.check_and_transform(code)

    # func call
    code = "super(BaseClass).__init__()"
    with pytest.raises(SyntaxError, match="super with args is not allowed"):
        static_sandbox.check_and_transform(code)

    # module white list
    code = "import os"
    with pytest.raises(ImportError, match="import os is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "from os.sys import platform"
    with pytest.raises(ImportError, match="import os.sys is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "from .test import test_api"
    with pytest.raises(ImportError, match="import test is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "from . import _api"
    with pytest.raises(ImportError, match="import None is not allowed"):
        static_sandbox.check_and_transform(code)

    # name check
    code = "_a"
    with pytest.raises(SyntaxError, match="_a is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "a._attr"
    with pytest.raises(AttributeError, match="attribute _attr is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "def _a(): ..."
    with pytest.raises(SyntaxError, match="function _a is not allowed"):
        static_sandbox.check_and_transform(code)

    # baned features
    code = "a = lambda x: x"
    with pytest.raises(SyntaxError, match="lambda is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "f'a={a}'"
    with pytest.raises(SyntaxError, match="joined str is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "del a"
    with pytest.raises(SyntaxError, match="del is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "def a(): yield 1"
    with pytest.raises(SyntaxError, match="yield is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "def a(): yield from 1"
    with pytest.raises(SyntaxError, match="yield is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "async def a(): ..."
    with pytest.raises(SyntaxError, match="async is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "await a()"
    with pytest.raises(SyntaxError, match="await is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "try:\n\ta()\nexcept Exception:\n\t..."
    with pytest.raises(SyntaxError, match="try is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "raise 'error'"
    with pytest.raises(SyntaxError, match="raise is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "assert x, y"
    with pytest.raises(SyntaxError, match="assert is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "with x as y:\n\t a()"
    with pytest.raises(SyntaxError, match="with is not allowed"):
        static_sandbox.check_and_transform(code)

    # class def
    code = "class A(metaclass=B): ..."
    with pytest.raises(SyntaxError, match="metaclass is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "class A(B1, B2): ..."
    with pytest.raises(SyntaxError, match="multi base is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "@proxy\nclass A: ..."
    with pytest.raises(SyntaxError, match="decorator on class is not allowed"):
        static_sandbox.check_and_transform(code)

    code = "class A(B): ..."
    with pytest.raises(
        SyntaxError, match="base class must be one of: object, Layer, Model, Module"
    ):
        static_sandbox.check_and_transform(code)

    # subscript_wrapper
    code = "a[1:2]"
    compiled_code = static_sandbox.check_and_transform(code, "__subscript_wrapper__")
    assert compiled_code.co_names == ("__subscript_wrapper__", "a")


def test_runtime_module_container():
    apps = dynamic_sandbox.runtime_module_container(
        applications, applications.app_whitelist, applications.app_wrapper
    )
    assert apps.BSTBase == applications.app_wrapper[applications.BSTBase]

    with pytest.raises(AttributeError, match="app_wrapper not exists"):
        apps.app_wrapper


def test_subscript_wrapper():
    # callable
    def func():
        ...

    class Obj(object):
        def __getitem__(self, key):
            return func

        def __setitem__(self, key, value):
            pass

    obj = Obj()
    obj = dynamic_sandbox.subscript_wrapper(obj)

    # getitem
    # numbers
    obj[1]
    obj[1.0, 2.0]
    obj[1:2, 1::1, 3]
    obj[(2, 1), 1:3]

    # str
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        obj["str"]
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        obj["str", 1]
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        obj[1:"str", 1]
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        obj[(1, "str"), 1]

    # setitem
    # numbers
    obj[1] = func
    obj[1.0, 2.0] = func
    obj[1:2, 1::1, 3] = func
    obj[(2, 1), 1:3] = func

    # str
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        obj["str"] = func
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        obj["str", 1] = func
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        obj[1:"str", 1] = func
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        obj[(1, "str"), 1] = func

    # str not callable
    obj["str"] = 1
    obj["str", 1] = 1
    obj[1:"str", 1] = 1
    obj[(1, "str"), 1] = 1


def test_prepare_globals():
    global_dict = runner.prepare_globals(apis={"test": "test"})
    assert global_dict["test"] == "test"

    api_list = ["tf", "keras", "layers", "Module", "Model", "Layer", "apps"]
    for api in api_list:
        assert api in global_dict


def test_run_code():
    models = {}

    def fit(client_base, server_base, server_fuse):
        models.update(
            {
                "client_base": client_base,
                "server_base": server_base,
                "server_fuse": server_fuse,
            }
        )

    runner.run_code(code=MODELS_CODE, apis={"fit": fit})

    assert "client_base" in models
    assert isinstance(models["client_base"], tensorflow_wrapper.SequentialWrapper)

    assert "server_base" in models
    assert isinstance(models["server_base"], tensorflow_wrapper.SequentialWrapper)

    assert "server_fuse" in models
    assert isinstance(models["server_fuse"], tensorflow_wrapper.ModelWrapper)


def test_run_code_builtins():
    code = """
class A(Module):
    def __init__(self):
        super().__init__()

    @tf.function
    def __call__(self):
        return self.x()

    def x(self):
        return True

a = A()
a()
"""

    runner.run_code(code=code, apis={})


def test_run_code_subscript_wrapper():
    code = """
obj = Obj()

obj[(1, 2), "a"::] = 1
obj[(1, 2), 3::]
"""

    class Obj:
        def __getitem__(self, key):
            return lambda x: x

        def __setitem__(self, key, value):
            pass

    runner.run_code(code=code, apis={"Obj": Obj})

    code1 = code + """obj[(1, 2), "a"::] = Obj"""
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        runner.run_code(code=code1, apis={"Obj": Obj})

    code2 = code + """obj[(1, 2), "a"::]"""
    with pytest.raises(ValueError, match="string slice of subscript is not allowed"):
        runner.run_code(code=code2, apis={"Obj": Obj})

    code3 = code + """obj[(1, 2), obj::]"""
    with pytest.raises(ValueError, match="object slice of subscript is not allowed"):
        runner.run_code(code=code3, apis={"Obj": Obj})


def test_run_code_with_error():
    code = """
class A(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return self.x(x)

    def x(self, x):
        return x[0]

a = A()
# should error
a(0)
"""

    with pytest.raises(TypeError, match="'int' object is not subscriptable"):
        runner.run_code(code=code, apis={})
