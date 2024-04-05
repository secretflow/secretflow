#! /bin/bash
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



function print_usage() {
    echo "Exe " $@
    echo "Usage: test_new.sh party_name protocol(aby3,semi2k,cheetah)"
    echo "party_rank required"
    exit 1
}

ray stop --force

sleep 5

. ./test_config.sh

if [ -z "$1" ]; then
	print_usage $@
fi
if [ -z "$2" ]; then
	print_usage $@
fi

echo $@

echo "ray started"
sleep 5

tc qdisc del dev eth0 root
tc qdisc add dev eth0 root handle 1: tbf rate ${net_rate} burst 256kb latency 800ms
tc qdisc add dev eth0 parent 1:1 handle 10: netem delay ${net_latency} limit 8000


if [ -n "$4" ]; then
    trace="--trace_memory true"
fi

export SPU_CTH_ENABLE_EMP_OT=1

if [ -z "$3" ]; then
    python ./test.py --protocol $2 --field 64 --party $1 --func ".*"
else
    python ./test.py --protocol $2 --field 64 --party $1 --func "$3" $trace
fi


ray stop --force

tc qdisc del dev eth0 root