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


mkdir -p sandbox/rootfs sandbox/work

nsjail --config .nsjail/nsjail.cfg --chroot /root/sandbox/rootfs -Mo --rlimit_fsize max --hostname APP \
    --disable_no_new_privs --rlimit_nofile max --disable_clone_newuser --disable_clone_newnet --skip_setsid \
    --rlimit_as max --rlimit_nproc max --pass_fd 256 --keep_env --keep_cap --proc_path /proc \
    -- /usr/local/bin/python -m secretflow.kuscia.entry /etc/kuscia/task-config.conf