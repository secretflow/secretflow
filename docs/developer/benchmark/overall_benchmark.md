# SecretFlow Benchmark Results

Latest update time: 2024/02/21

The benchmarks were conducted on a standard environment on Alibaba ecs.r8i.8xlarge instance, with 32 vCPU cores (16 real CPUs with hyperthreading) and 256G memory.

Each party will use one instance of machine.

- **CPU**: Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz
- **RAM**: 256GB
- **OS**: CentOS 7.9 64bit
- **Disk**: 200GB

Network Condition:

- **Bandwidth**: Limited to 100 Mbps
- **Fixed Delay**: 50 ms

## Basic Operation Benchmark

Each party holds 100000000 size random data

| Task                          | 2 party cheetah Time (s) | 3 party aby3 Time (s) | semi2k(TFP)  Time (s)   |
|-------------------------------|--------------------------|-----------------------|-------------------------|
| element-wise addition         | 283.83696                | 1621.0594             | 292.81814               |
| element-wise multiplication   | 3836.5489                | 2641.75911            | 522.32634               |
| element-wise less comparison  | 3355.8496                | -                     | 813.28057               |

## PSI and PIR Operation Benchmark
### PSI benchmark table

Please refer to [Benchmark](psi_benchmark.md#信通院测试标准下的benchmark-my-target-section) .


### PIR benchmark table (only 2 party setting)

|Query ID Count| Indistinguishability | Latency | Time (s) | Time (h) | Algorithm   |
|--------------|----------------------|---------|--------|---------|-------------|
| 10000        |  Million Level       | 50msec  | 3686   | 1:01:26 | KeywordPIR  |
| 1            |  Million Level       | 50msec  | 51     | 0:00:51 | KeywordPIR  |

## United Statistiсs Operation Benchmark

Each party holds 100000000 size random data

| Task                     | 2 party cheetah Time (s) | 3 party aby3 Time (s) | semi2k(TFP)  Time (s)   |
|--------------------------|--------------------------|-----------------------|-------------------------|
| variance (reveal mean)   | 59.45547                 | 5.78                  | 3.43896                 |
| variance (protect mean)  | OOM                      | 4295.96               | 1607.76037              |
| median                   | 91.70969                 | 1079.87955            | 27.40997                |

Note: variance (reveal mean) allow revealing mean value when computing variance, which is practical for some applications.

Note: cheetah performance may need further optimization.

## Feature Engineering Operation Benchmark

In 2-party setting, each party holds 800000 lines of data with 1500 features and the first party holds the label.

In 3-party setting, each party holds 800000 lines of data with 1000 features and the first party holds the label.

| Task                          | 2 party Time (s)     |3 party Time (s)      |
|-------------------------------|----------------------|----------------------|
| WOE and IV (OU)               | 434.46               | 505.04               |

## Machine Learning Algorithm Benchmark

In 2-party setting, each party holds 800000 lines of data with 1500 features and the first party holds the label.

In 3-party setting, each party holds 800000 lines of data with 1000 features and the first party holds the label.

Classification algorithms are used as examples.

The training time is the best training time for the algorithm to reach convergence compared to the best cleartext model (hyperparameter tuning may affect the result). In addition, train auc must be within 0.02 away from the cleartext model. Refer to benchmark_examples folder in source code for more parameter details.

The inference time is the prediction time on training data.

| Task                          | 2 party cheetah Time (s) | 3 party aby3 Time (s) | semi2k(TFP)  Time (s)   |
|-------------------------------|--------------------------|-----------------------|-------------------------|
| LR Training   (with SS-SGD)   | 1244.53                  | 1155.78               | 693.57                  |
| LR Inference  (with SS-SGD)   | 300                      | 260.7                 | 261.31                  |

| Task                          | 2 party Time (s)         | 3 party Time (s)      |
|-------------------------------|--------------------------|-----------------------|
| XGB Training  (with SGB OU)   | 4176.54                  | 4568.08               |
| XGB Inference (with SGB OU)   | 54.5                     | 37.73                 |
| NN Training   (split learning)| 6844                     | 2477                  |
| NN Inference  (split learning)| 5                        | 3                     |

Note that XGB and NN pipelines are not affected by MPC protocols because they use other privacy computation technologies.

## Benchmark scripts

The replication process can be found in the following folders:

For basic operations and united statistics, see source code `secretflow/benchmark_examples/sf_basic_ops_test` folder for more details.
For feature engineering and machine learning, see source code `secretflow/benchmark_examples/sf_component_test` folder for more details.

## Extra Notes

Semi2k protocol relies on a trusted third party, it should not be considered a pure 2pc MPC protocol.
Semi2k needs a trusted third party (TTP) or trusted first party (TFP).
In general, it should not be used for pure 2 party situation in production.
