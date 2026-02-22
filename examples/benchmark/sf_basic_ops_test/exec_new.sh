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


if [[ $1 != "semi2k" && $1 != "aby3" && $1 != "cheetah" ]]; then
	echo "Usage: $0 protocol(semi2k, aby3, cheetah) [func_name_pattern(regex for func_name)] [trace_mem]"
	exit 1
fi

. ./test_config.sh

suffix=$(date +%H_%M_%S_%s)
log_name=$test_log_dir/test.log.$suffix
echo "log file: " $log_name

bash -x sync_new.sh $1

exec_cmd="if ! [ -e $test_log_dir ]; then mkdir -p $test_log_dir; fi; cd $test_dir; conda activate $conda_env; ray stop; bash -x test_new.sh party_name $1 $2 $3 >>$log_name 2>&1; ray stop"


ssh root@${remote_ips[0]} "$(echo $exec_cmd | sed "s/party_name/${remote_parties[0]}/g")" &
if [[ $1 == 'aby3' ]]; then
	ssh root@${remote_ips[1]} "$(echo $exec_cmd | sed "s/party_name/${remote_parties[1]}/g")" &
fi

ssh root@${local_ip} "$(echo $exec_cmd | sed "s/party_name/${local_party}/g")"
