#!/bin/bash

. ./test_config.sh

copy_files="test_new.sh test_config.sh test.py"

for remote_ip in $remote_ips;
do
	ssh root@${remote_ip} "if ! [ -e $test_dir ]; then mkdir -p $test_dir; fi"
done



for file in $copy_files;
do
	for remote_ip in ${remote_ips};
	do
		scp $file root@${remote_ip}:$test_dir/
	done
done