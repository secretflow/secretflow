# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

workspace(name="secretflow")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name="yacl",
    commit="47dbaf6663a9a80e5a94511a20b44f2350451afe",
    remote="https://github.com/secretflow/yacl.git",
)

load("@yacl//bazel:repositories.bzl", "yacl_deps")

yacl_deps()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name="pybind11_bazel",
    sha256="a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d",
    strip_prefix="pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
    urls=[
        "https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip",
    ],
)

http_archive(
    name="pybind11",
    build_file="@pybind11_bazel//:pybind11.BUILD",
    sha256="eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec",
    strip_prefix="pybind11-2.10.0",
    urls=[
        "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz",
    ],
)

load(
    "@rules_foreign_cc//foreign_cc:repositories.bzl",
    "rules_foreign_cc_dependencies",
)

rules_foreign_cc_dependencies(
    register_built_tools=False,
    register_default_tools=False,
    register_preinstalled_tools=True,
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(name="local_config_python")
