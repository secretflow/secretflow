# Benchmarks

We can quickly benchmark via the component test framework.

Currently contained examples:

- WOE
- LR
- XGB

## Steps

1. Prepare 3 Linux machines for testing purpose. We recomment at least 32 CPUs each. Call them test_001, test_002, test_003 for example.
2. Make sure these 3 Linux machines can ssh login into each other without passwords.
3. use root user
4. install docker image secretflow/secretflow-anolis8
5. create conda environment and install secretflow and component test framework for test_001.
6. create a dir to store your code, e.g. /root/sf_test for all machines
7. create logs dir in this dir, e.g. /root/sf_test/logs for all machines
8. copy files in sf_test in this folder to your folder for test_001
9. prepare the test version you want to test, obtain the whl files
10. copy the  *.whl files to /root/sf_test/*.whl for test_001
11. prepare data csvs, e.g. /root/sf_test/data/train_alice.csv and /root/sf_test/data/test_bob.csv
    Note that the test_001 should have all the test files, other machines should have its own data.
    test_002 should have e.g. /root/sf_test/data/test_bob.csv only.
12. change the following file based on your benchmark needs:
    - /root/sf_test/configs/node.json
    - /root/sf_test/configs/data_*.json

## Who should do what?

Developers: make sure cluster, net, and algorithm configs are correct.

Test runners: make sure data, node and version configs are correct.

## How to run the tests?

1. setup data and configuration
    1. data are in data folder
    2. configurations are in configs folder
2. run script

## Run Scripts

### Run all test in one go

``` shell
python run_benchmark.py -c config.json
```

### Run one single test

``` shell
python run_benchmark.py -c config.json  -s woe
```

### Run in test run

Sample 10 features each party and run (for debug only).

``` shell
python run_benchmark.py -c config.json  -s logistic_regression -t
```

## How to Read the results

Summary of memory and time taken: find current folder report json

## Tips

For long test, use nohup

change the output_file to a name you like

```shell
nohup xxxxx  > output_file 2>&1 &
```

## TODO

read auc, ks, f1 results

Currently:

1. Search `biclassification_eval` in terminal or nohup.oout for output_file.
2. Find out a line like this.
`INFO:root:run comp biclassification_eval:ml.eval:biclassification_eval:0.0.1 on node alice with uuid_path 9eddbdba-17a6-4267-853e-e9eed335dcb0`
3. Find log `logs/9eddbdba-17a6-4267-853e-e9eed335dcb0.log`.
4. Search for `auc`.
