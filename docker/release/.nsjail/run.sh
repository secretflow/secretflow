#!/bin/bash

mkdir -p sandbox/rootfs sandbox/work

nsjail --config .nsjail/nsjail.cfg --chroot /root/sandbox/rootfs -Mo --rlimit_fsize max --hostname APP \
    --disable_no_new_privs --rlimit_nofile max --disable_clone_newuser --disable_clone_newnet --skip_setsid \
    --rlimit_as max --rlimit_nproc max --pass_fd 256 --keep_env --keep_cap --proc_path /proc \
    -- /usr/local/bin/python -m secretflow.kuscia.entry /etc/kuscia/task-config.conf