# Copyright 2022 Ant Group Co., Ltd.
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

from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read_requirements():
    requirements = []
    with open('requirements.txt') as file:
        requirements = file.read().splitlines()
    with open('dev-requirements.txt') as file:
        requirements.extend(file.read().splitlines())
    print("Requirements: ", requirements)
    return requirements


setup(
    name='secretflow',
    version='0.5.0',
    description='Secret Flow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ant Group',
    author_email='',
    url='',
    packages=find_packages(exclude=('examples', 'examples.*', 'tests', 'tests.*')),
    install_requires=read_requirements(),
    extras_require={
        'dev': ['pylint'],
    },
)
