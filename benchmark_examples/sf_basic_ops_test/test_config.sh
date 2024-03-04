#!/bin/bash
local_party=party_name
local_ip=party_ip
remote_parties=(remote_party_name)
remote_ips=(remote_party_ip)
conda_env=your_conda_env_name
test_dir=/home/your_user_name/secretflow_benchmark_test
test_log_dir=$test_dir/logs

net_rate=100mbit
net_latency=50msec