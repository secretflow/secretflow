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

## Extra Notes

Semi2k protocol relies on a trusted third party, it should not be considered a MPC safe 2pc protocol.

## Basic Operation Benchmark

Each party holds 100000000 size random data

| Task                          | 2 party semi2k Time (s) |2 party cheetah Time (s) | 3 party aby3 Time (s) |
|-------------------------------|-------------------------|-------------------------|-----------------------|
| element-wise addition         | 292.81814               | 283.83696               | 1621.0594             |
| element-wise multiplication   | 522.32634               | 3836.5489               | 2641.75911            |
| element-wise less comparison  | 813.28057               | 3355.8496               | -                     |

## United Statistiсs Operation Benchmark

Each party holds 100000000 size random data

| Task                          | 2 party semi2k Time (s) |2 party cheetah Time (s) | 3 party aby3 Time (s) |
|-------------------------------|-------------------------|-------------------------|-----------------------|
| variance (reveal mean)        | 3.43896                 | 59.45547                | 5.78                  |
| variance (protect mean)       | 1607.76037              |  -                      | 4295.96               |
| median                        | 27.40997                | 91.70969                | 1079.87955            |

## Feature Engineering Operation Benchmark

In 2-party setting, each party holds 800000 lines of data with 1500 features and the first party holds the label.

In 3-party setting, each party holds 800000 lines of data with 1000 features and the first party holds the label.

| Task                          | 2 party semi2k Time (s) |2 party cheetah Time (s) | 3 party aby3 Time (s) |
|-------------------------------|-------------------------|-------------------------|-----------------------|
| WOE and IV Computation        | 449.15                  | 434.46                  | 505.04                |

## Machine Learning Algorithm Benchmark

In 2-party setting, each party holds 800000 lines of data with 1500 features and the first party holds the label.

In 3-party setting, each party holds 800000 lines of data with 1000 features and the first party holds the label.

Classification algorithms are used as examples.

The training time is the best training time for the algorithm to reach convergence compared to the best cleartext model (hyperparameter tuning may affect the result). In addition, train auc must be within 0.02 away from the cleartext model. Refer to benchmark_examples folder in source code for more parameter details.

The inference time is the prediction time on training data.

| Task                          | 2 party semi2k Time (s) |2 party cheetah Time (s) | 3 party aby3 Time (s) |
|-------------------------------|-------------------------|-------------------------|-----------------------|
| LR Training   (with SS-SGD)   | 693.57                  | 1244.53                 | 1155.78               |
| XGB Training  (with SGB)      | 4175.76                 | 4176.54                 | 4568.08               |
| NN Training   (split learning)| 6844                    | 6844                    | 2477                  |
| LR Inference  (with SS-SGD)   | 261.31                  | 300                     | 260.7                 |
| XGB Inference (with SGB)      | 56.83                   | 54.5                    | 37.73                 |
| NN Inference  (split learning)| 5                       | 5                       | 3                     |
