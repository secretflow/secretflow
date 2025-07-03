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

import argparse
import atexit
import os
import platform
import posixpath
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import List

import setuptools
from setuptools import find_packages, setup
from setuptools.command import build_ext

this_directory = os.path.abspath(os.path.dirname(__file__))


def get_commit_id() -> str:
    commit_id = (
        subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    )
    dirty = subprocess.check_output(['git', 'diff', '--stat']).decode('ascii').strip()

    if dirty:
        commit_id = f"{commit_id}-dirty"

    return commit_id


def update_version_file(custom_feature, filepath):
    # Proceed with file modifications
    today = date.today()
    dstr = today.strftime("%Y%m%d")
    with open(filepath, "r") as fp:
        content = fp.read()

    content = content.replace("$$DATE$$", dstr)
    content = content.replace("$$BUILD_TIME$$", time.strftime('%b %d %Y, %X'))
    try:
        content = content.replace("$$COMMIT_ID$$", get_commit_id())
    except:
        pass

    if "SF_BUILD_DOCKER_NAME" in os.environ:
        content = content.replace(
            "$$DOCKER_VERSION$$", os.environ["SF_BUILD_DOCKER_NAME"]
        )
    if custom_feature != "lite":
        content = content.replace('__mode__ = "lite"', '__mode__ = "full"')

    with open(filepath, "w+") as fp:
        fp.write(content)


def complete_version_file(custom_feature, *filepath):
    file_path = os.path.join(".", *filepath)
    backup_path = file_path + ".bak"

    # Create backup if it doesn't exist
    if not os.path.exists(backup_path):
        # Read original content and create backup
        with open(file_path, "r") as fp:
            original_content = fp.read()
        with open(backup_path, "w") as fp:
            fp.write(original_content)

        # Register restore function to run at exit
        def restore_original():
            if os.path.exists(backup_path):
                # Restore original content from backup
                with open(backup_path, "r") as fp:
                    backup_content = fp.read()
                with open(file_path, "w") as fp:
                    fp.write(backup_content)
                # Clean up backup file
                os.remove(backup_path)

        atexit.register(restore_original)

    update_version_file(custom_feature, file_path)


def find_version(custom_feature, *filepath):
    complete_version_file(custom_feature, *filepath)
    # Extract version information from filepath
    with open(os.path.join(".", *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        print("Unable to find version string.")
        exit(-1)


def filter_requirements(requirements: List[str], custom_feature: str):
    """A lite feature in comment should be something like "# FEATURE=[lite]"."""
    filtered_reqs = []
    for r in requirements:
        comment_symbol_idx = r.find("#")
        if comment_symbol_idx == -1:
            continue

        comment = r[comment_symbol_idx + 1 :]
        feature_match = re.search(r"FEATURE=\[([0-9A-Za-z,_]+)\]", comment)
        if not feature_match:
            continue
        features = feature_match.group(1).split(",")
        if custom_feature in features:
            filtered_reqs.append(r[:comment_symbol_idx].strip())

    return filtered_reqs


def read_requirements(custom_feature: str = None):
    requirements = []
    dependency_links = []
    with open("./requirements.txt") as file:
        requirements = file.read().splitlines()
    if custom_feature:
        requirements = filter_requirements(requirements, custom_feature)
    for r in requirements:
        if r.startswith("--extra-index-url"):
            requirements.remove(r)
            dependency_links.append(r)
    print("Requirements: ", requirements)
    print("Dependency: ", dependency_links)
    return requirements, dependency_links


# [ref](https://github.com/perwin/pyimfit/blob/master/setup.py)
# Modified cleanup command to remove dist subdirectory
# Based on: https://stackoverflow.com/questions/1710839/custom-distutils-commands
class CleanCommand(setuptools.Command):
    description = "custom clean command that forcefully removes dist directories"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        directories_to_clean = ["./build"]

        for dir in directories_to_clean:
            if os.path.exists(dir):
                shutil.rmtree(dir)


# [ref](https://github.com/google/trimmed_match/blob/master/setup.py)
class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, bazel_target, ext_name):
        self._bazel_target = bazel_target
        self._relpath, self._target_name = posixpath.relpath(bazel_target, "//").split(
            ":"
        )
        setuptools.Extension.__init__(self, ext_name, sources=[])


class BuildBazelExtension(build_ext.build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        for ext in self.extensions:
            self.bazel_build(ext)
        build_ext.build_ext.run(self)

    def bazel_build(self, ext):
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        bazel_argv = [
            "bazel",
            "build",
            ext._bazel_target + ".so",
            "--symlink_prefix=" + os.path.join(self.build_temp, "bazel-"),
            "--compilation_mode=" + ("dbg" if self.debug else "opt"),
        ]

        if platform.machine() == "x86_64":
            bazel_argv.extend(["--config=avx"])

        self.spawn(bazel_argv)

        shared_lib_suffix = ".so"
        ext_bazel_bin_path = os.path.join(
            self.build_temp,
            "bazel-bin",
            ext._relpath,
            ext._target_name + shared_lib_suffix,
        )

        ext_dest_path = self.get_ext_fullpath(ext.name)
        Path(ext_dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


def plat_name():
    # Default Linux platform tag
    plat_name = "manylinux2014_x86_64"

    if sys.platform == "darwin":
        # macOS platform detection
        if platform.machine() == "x86_64":
            plat_name = "macosx_12_0_x86_64"
        else:
            plat_name = "macosx_12_0_arm64"
    elif sys.platform.startswith("linux"):
        # Linux platform detection
        if platform.machine() == "aarch64":
            plat_name = "manylinux_2_28_aarch64"

    return plat_name


def long_description():
    with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    if os.getcwd() != this_directory:
        print("You must run setup.py from the project root")
        exit(-1)

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--lite", action="store_true", help="Build SecretFlow lite", required=False
    )
    args, unknowns = argparser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknowns

    package_name = "secretflow"
    description = "SecretFlow"
    # Feature is used to filter the requirements.
    custom_feature = None

    pkg_exclude_list = [
        "examples",
        "examples.*",
        "tests",
        "tests.*",
    ]

    entry_points = {
        "console_scripts": ["secretflow = secretflow.cli:cli"],
    }
    package_data = {'secretflow': ['component/translation.json']}
    ext_modules = None

    if args.lite:
        """
        The primary distinction between secretflow-lite and non-lite lies in
        the inclusion of traditional machine learning only. Specifically, lite
        does not incorporate deep learning dependency packages due to their
        significant size.
        """
        package_name = "secretflow-lite"
        custom_feature = "lite"
        description = "SecretFlow Lite"
        pkg_exclude_list.extend(
            ["secretflow_fl", "secretflow_fl.*"]
        )  # exclude secretflow_fl in lite version
    else:
        package_name = "secretflow"
        description = "SecretFlow Full"
        entry_points['secretflow_plugins'] = ['secretflow_fl = secretflow_fl.component']
        package_data["secretflow_fl"] = ['component/translation.json']
        ext_modules = [
            BazelExtension(
                "//secretflow_lib/binding:_lib", "secretflow_fl/security/privacy/_lib"
            ),
        ]

    install_requires, dependency_links = read_requirements(custom_feature)

    setup(
        name=package_name,
        version=find_version(custom_feature, "secretflow", "version.py"),
        license="Apache 2.0",
        description=description,
        long_description=long_description(),
        long_description_content_type="text/markdown",
        author="SCI Center",
        author_email="secretflow-contact@service.alipay.com",
        url="https://github.com/secretflow/secretflow",
        packages=find_packages(exclude=pkg_exclude_list),
        package_data=package_data,
        install_requires=install_requires,
        ext_modules=ext_modules,
        extras_require={"dev": ["pylint"]},
        cmdclass=dict(
            build_ext=BuildBazelExtension, clean=CleanCommand, cleanall=CleanCommand
        ),
        dependency_links=dependency_links,
        options={
            "bdist_wheel": {"plat_name": plat_name()},
        },
        entry_points=entry_points,
    )
