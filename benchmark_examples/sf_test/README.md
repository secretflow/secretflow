# Benchmark scrips

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
