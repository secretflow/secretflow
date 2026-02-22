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


. ./test_config.sh

copy_files="test_new.sh test_config.sh test.py"

for remote_ip in $remote_ips;
do
	ssh root@${remote_ip} "if ! [ -e $test_dir ]; then mkdir -p $test_dir; fi"
done



for file in $copy_files;
do
	for remote_ip in ${remote_ips[*]};
	do
		scp $file root@${remote_ip}:$test_dir/
	done
done