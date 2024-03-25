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


usage() { echo "Usage: $0 [-l <en|zh_CN>]" 1>&2; exit 1; }


while getopts ":l:" o; do
    case "${o}" in
        l)
            l=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${l}" ]; then
    usage
fi

echo "selected language is: ${l}"

if [[ "$l" != "en" && "$l" != "zh_CN" ]]; then
    usage
fi

rm -rf ./source
SPHINX_APIDOC_OPTIONS=members,autosummary sphinx-apidoc -f -d 2 -t templates -o ./source ../secretflow/
make clean
env PYTHONPATH=$PYTHONPATH:$PWD/.. make SPHINXOPTS="-D language='${l}'" html
