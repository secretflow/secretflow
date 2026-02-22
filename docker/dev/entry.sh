#!/bin/bash
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


set -ex

cp -r src src_copied
cd src_copied


source ~/.bashrc
conda create -n build python=3.10 -y
conda activate build

pip install build
python3 -m build --wheel 

# Clean old wheel files before copying new ones
rm -f ../src/docker/dev/*.whl
cp dist/* ../src/docker/dev/
