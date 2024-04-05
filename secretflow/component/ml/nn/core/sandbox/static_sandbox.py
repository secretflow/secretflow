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

import ast
import builtins
from typing import Any, List

from .whitelists.common_whitelist import (
    baseclass_whitelist,
    builtins_whitelist,
    module_whitelist,
)


class Transformer(ast.NodeTransformer):
    def __init__(self, subscript_wrapper=""):
        super().__init__()

        self.subscript_wrapper = subscript_wrapper
        self.builtins_blacklist = set(dir(builtins)) - set(builtins_whitelist.keys())

    @staticmethod
    def illegal_general_name(name: str):
        return (
            (name.startswith("_") or name.endswith("_"))
            and name != "__init__"
            and name != "__call__"
            and name != "_"
        )

    def check_module_name(self, names: List[str]):
        for name in names:
            if name.startswith("_") or name.endswith("_"):
                raise ImportError(f"module {name} is not allowed")

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            modules = alias.name.split(".")
            self.check_module_name(modules)

            module = modules[0]
            if module not in module_whitelist:
                raise ImportError(f"import {alias.name} is not allowed")
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        # relative import is not allowed
        if node.level != 0:
            raise ImportError(f"import {node.module} is not allowed")

        modules = node.module.split(".")
        self.check_module_name(modules)

        module = modules[0]
        if module not in module_whitelist:
            raise ImportError(f"import {node.module} is not allowed")
        for alias in node.names:
            modules = alias.name.split(".")
            self.check_module_name(modules)

        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.builtins_blacklist:
            raise SyntaxError(
                f"reserved {node.id} is not allowed", node.lineno, node.col_offset
            )

        if self.illegal_general_name(node.id):
            raise SyntaxError(f"{node.id} is not allowed", node.lineno, node.col_offset)

        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if self.illegal_general_name(node.attr):
            raise AttributeError(f"attribute {node.attr} is not allowed")

        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self.illegal_general_name(node.name):
            raise SyntaxError(
                f"function {node.name} is not allowed", node.lineno, node.col_offset
            )

        return self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        raise SyntaxError("lambda is not allowed", node.lineno, node.col_offset)

    def visit_Call(self, node: ast.Call) -> Any:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "super"
            and len(node.args) > 0
        ):
            raise SyntaxError(
                f"super with args is not allowed", node.lineno, node.col_offset
            )
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        if len(node.keywords) > 0:
            raise SyntaxError("metaclass is not allowed", node.lineno, node.col_offset)
        if len(node.decorator_list) > 0:
            raise SyntaxError(
                "decorator on class is not allowed", node.lineno, node.col_offset
            )
        if len(node.bases) > 1:
            raise SyntaxError("multi base is not allowed", node.lineno, node.col_offset)
        if len(node.bases) == 1:
            base = node.bases[0]
            if not isinstance(base, ast.Name) or base.id not in baseclass_whitelist:
                raise SyntaxError(
                    f"base class must be one of: {', '.join(baseclass_whitelist)}",
                    node.lineno,
                    node.col_offset,
                )
        return self.generic_visit(node)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Any:
        raise SyntaxError(
            "formatted value is not allowed", node.lineno, node.col_offset
        )

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        raise SyntaxError("joined str is not allowed", node.lineno, node.col_offset)

    def visit_Del(self, node: ast.Del) -> Any:
        raise SyntaxError("del is not allowed")

    def visit_Yield(self, node: ast.Yield) -> Any:
        raise SyntaxError("yield is not allowed", node.lineno, node.col_offset)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Any:
        raise SyntaxError("yield is not allowed", node.lineno, node.col_offset)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        raise SyntaxError("async is not allowed", node.lineno, node.col_offset)

    def visit_Await(self, node: ast.Await) -> Any:
        raise SyntaxError("await is not allowed", node.lineno, node.col_offset)

    def visit_Try(self, node: ast.Try) -> Any:
        raise SyntaxError("try is not allowed", node.lineno, node.col_offset)

    def visit_Raise(self, node: ast.Raise) -> Any:
        raise SyntaxError("raise is not allowed", node.lineno, node.col_offset)

    def visit_Assert(self, node: ast.Assert) -> Any:
        raise SyntaxError("assert is not allowed", node.lineno, node.col_offset)

    def visit_With(self, node: ast.With) -> Any:
        raise SyntaxError("with is not allowed", node.lineno, node.col_offset)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        node = self.generic_visit(node)
        if not self.subscript_wrapper:
            return node

        node.value = ast.Call(
            func=ast.Name(id=self.subscript_wrapper, ctx=ast.Load()),
            args=[node.value],
            keywords=[],
        )
        node.value = ast.fix_missing_locations(node.value)
        return node


def check_and_transform(code: str, subscript_wrapper=""):
    if not code.isascii():
        raise SyntaxError("non ascii char is not allowed")
    tree = ast.parse(code)
    transformer = Transformer(subscript_wrapper=subscript_wrapper)
    tree_trans = transformer.visit(tree)
    return compile(tree_trans, "<string>", "exec")
