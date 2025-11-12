# Basic Operation Benchamrk

Example to setup a benchmark framework for basic computations.

## Setup Steps

1. prepare 2 or 3 machines
2. set up Secure Shell (SSH) trust between these machines
3. install conda and secretflow on each machine, see README.md in secretflow, assume the conda env name is "sf"
4. create the test folder on each machine: `mkdir /home/your_user_name/secretflow_benchmark_test`
5. create the log folder on each machine:`mkdir /home/your_user_name/secretflow_benchmark_test/logs`
6. copy from this example folder and edit `test_config.sh` file in the test folder: `/home/your_user_name/secretflow_benchmark_test/test_config.sh`
7. copy from this example folder and edit `test.py` file in the test folder: `/home/your_user_name/secretflow_benchmark_test/test.py`
8. copy and paste the `exec_new.sh` , `sync_new.sh` and `test_new.sh` files into the test folder: `/home/your_user_name/secretflow_benchmark_test/`
9. optionally prepare data in `/home/your_user_name/data` folder (The data can also be randomly generated)

## Execution Tests

execute the command in a conda environment which installed secretflow.

```sh
conda activate sf
sh exec_new.sh protocol_type
```

change “protocol_type” to one of the following values:

- semi2k
- cheetah
- aby3

example outputs:

```txt
log file:  /home/your_user_name/secretflow_benchmark_test/logs/test.log.15_34_06_1708932846
+ . ./test_config.sh
++ local_party=alice
++ local_ip=123.45.67.890
++ remote_parties=(bob)
++ remote_ips=(123.45.67.891)
++ conda_env=sf
++ test_dir=/home/your_user_name/secretflow_benchmark_test
++ test_log_dir=/home/your_user_name/secretflow_benchmark_test/logs
```
