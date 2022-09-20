import os
import posixpath
import shutil
from pathlib import Path

import setuptools
from setuptools import find_packages, setup
from setuptools.command import build_ext

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read_requirements():
    requirements = []
    with open('./requirements.txt') as file:
        requirements = file.read().splitlines()
    with open('./docker/dev-requirements.txt') as file:
        requirements += file.read().splitlines()
    print("Requirements: ", requirements)
    return requirements


# [ref](https://github.com/perwin/pyimfit/blob/master/setup.py)
# Modified cleanup command to remove build subdirectory
# Based on: https://stackoverflow.com/questions/1710839/custom-distutils-commands
class CleanCommand(setuptools.Command):
    description = "custom clean command that forcefully removes dist/build directories"
    user_options = []

    def initialize_options(self):
        self._cwd = None

    def finalize_options(self):
        self._cwd = os.getcwd()

    def run(self):
        assert os.getcwd() == self._cwd, 'Must be in package root: %s' % self._cwd
        os.system('rm -rf ./build ./dist')


# [ref](https://github.com/google/trimmed_match/blob/master/setup.py)
class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, bazel_target, ext_name):
        self._bazel_target = bazel_target
        self._relpath, self._target_name = posixpath.relpath(bazel_target, '//').split(
            ':'
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
            'bazel',
            'build',
            ext._bazel_target + '.so',
            '--symlink_prefix=' + os.path.join(self.build_temp, 'bazel-'),
            '--compilation_mode=' + ('dbg' if self.debug else 'opt'),
        ]

        self.spawn(bazel_argv)

        shared_lib_suffix = '.so'
        ext_bazel_bin_path = os.path.join(
            self.build_temp,
            'bazel-bin',
            ext._relpath,
            ext._target_name + shared_lib_suffix,
        )

        ext_dest_path = self.get_ext_fullpath(ext.name)
        Path(ext_dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


setup(
    name='secretflow',
    version='0.7.7b0',
    license='Apache 2.0',
    description='Secret Flow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='SCI Center',
    author_email='secretflow-contact@service.alipay.com',
    url='https://github.com/secretflow/secretflow',
    packages=find_packages(exclude=('examples', 'examples.*', 'tests', 'tests.*')),
    install_requires=read_requirements(),
    ext_modules=[
        BazelExtension(
            '//secretflow_lib/binding:_lib', 'secretflow/security/privacy/_lib'
        ),
    ],
    extras_require={'dev': ['pylint']},
    cmdclass=dict(
        build_ext=BuildBazelExtension, clean=CleanCommand, cleanall=CleanCommand
    ),
)
