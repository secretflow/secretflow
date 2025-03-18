# Introduction

The implemention of FedSMP, a differentially private federated learning scheme.

[Federated Learning with Sparsified Model Perturbation (FedSMP): Improving Accuracy under Client-Level Differential Privacy](https://ieeexplore.ieee.org/abstract/document/10360319/)

## Algorithm Description

FedSMP is a novel differentially-private FL scheme which can provide a client-level DP guarantee while maintaining high model accuracy. To mitigate the impact of privacy protection on model accuracy, Fed-SMP leverages a new technique called Sparsified Model Perturbation (SMP) where local models are sparsified first before being perturbed by Gaussian noise.

In each round of FL learning, FedSMP includs the following steps:

1. the server:

    - generates a random mask vector $m\in\{0,1\}^d$ with a compression ratio $p=1-\frac{k}{d}$, where $k$ is the total number of element $1$ in $m\in\{0,1\}^d$

2. the clients:

    - conduct local training to obatin the model updates $g$
    - mask the model updates $g=g*m$
    - scale up the model updats $g=g*\frac{d}{k}$
    - inject Gaussian noise $g=g+\mathcal{N}(0,\frac{\sigma^2C^2I^d}{N})$
    - mask the model updates $g=g*m$

3. the server:
    - aggregate the model updates with DP noise

## Metric

settings:

dataset: MNIST, batchsize: 64, epoch: 10, learning rate: 0.01, clipping threshold: 1.0

| compression ratio | accuracy (\%) | size of grad (MB) | Noise scale |
| ---- | ---- | ---- | ---- |
| $p$=0.001 |0.8960  |0.00127  | $\sigma$=0.05 |
| $p$=0.005| 0.9347 |0.00635  | $\sigma$=0.05 |
| $p$=0.01| 0.9586 |0.0127  |$\sigma$=0.05  |
| $p$=0.1| **0.9787** |0.127  | $\sigma$=0.05 |
| $p$=0.2| 0.9737 |0.254  | $\sigma$=0.05 |
| $p$=0.4|0.9720  |0.508  | $\sigma$=0.05 |
| $p$=0.8| 0.9649 |1.016  | $\sigma$=0.05 |
| $p$=1.0(DP-FedAvg)| 0.9455 |1.27  |$\sigma$=0.05  |
| $p$=1.0(FedAvg)| 0.9938 |1.27 | $\sigma$=0 |

## Test

`pytest --env sim -n auto -v --capture=no tests/ml/nn/fl/strategy/test_fedsmp_torch.py`
