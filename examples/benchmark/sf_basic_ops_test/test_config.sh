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

local_party=party_name
local_ip=party_ip
remote_parties=(remote_party_name)
remote_ips=(remote_party_ip)
conda_env=your_conda_env_name
test_dir=/home/your_user_name/secretflow_benchmark_test
test_log_dir=$test_dir/logs

net_rate=100mbit
net_latency=50msec