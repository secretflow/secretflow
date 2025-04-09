# KL-FedALA with Secretflow

This document implements KL-FedALA with secretflow. In federation learning, selecting users participating in aggregation based on KL divergence is a popular algorithm.

## FedALA Algorithm
> reffered paper: [FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://arxiv.org/abs/2212.01197)

> reffered codes: [https://github.com/TsingZ0/FedALA](https://github.com/TsingZ0/FedALA)

## User selection based on KL divergence
### KL Divergence Calculation
Each client computes the KL divergence between its local data distribution (e.g., class probabilities) and the global model's output distribution.
### Client Filtering
Clients with low KL divergence (indicating alignment with the global model) are prioritized for aggregation, while those with high divergence (potential outliers or malicious actors) are excluded.

## Experiments Introduction
The testting contains 4 datasets: `mnist`, `cifar10`, `cifar100` and `TinyImage`.

Data partiotion is same to the paper [https://github.com/TsingZ0/PFLlib](https://github.com/TsingZ0/PFLlib). We use Dirichlet Distribution to generrate heterogeneous dataset for each client.

In order to reproduce the experimental results in the paper, the hyperparameter is same to the paer. The main hyperparameter is listed as follows:

`global epochs` = 1000

`local learning rate` = 0.005 for cifar10, cifar100 and TinyImage, 0.1 for mnist

`number of clients` = 20

`eta` = 1

`s` = 80%

`p`  =2

## Quick Start
Before starting experiments, make sure you are in the right working directory.
You need to get the heterogeneous datasets from [https://github.com/TsingZ0/PFLlib] and save it at `/dst`.

```commandline
cd ./FedALA/ALAsys
```

You can use following commands to start your experiments.

```commandline
python main.py  -nc 20 -data cifar10 -et 1 -p 2 -s 80 -sc 20
```
Note that only the cpu version is supported currently, set different to test the optimum performance, and when sc equals to nc, the algorithm will be the same as FedALA.