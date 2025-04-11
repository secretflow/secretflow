# FedALA with Secretflow

This document implements FedALA with secretflow.

> reffered paper: [FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://arxiv.org/abs/2212.01197)

> reffered codes: [https://github.com/TsingZ0/FedALA](https://github.com/TsingZ0/FedALA)

## Experiments Introduction
The testting contains 2 datasets: `mnist` and `cifar10`.

Data partiotion is same to the paper [https://github.com/TsingZ0/PFLlib](https://github.com/TsingZ0/PFLlib). We use Dirichlet Distribution to generrate heterogeneous dataset for each client.

In order to reproduce the experimental results in the paper, the hyperparameter is same to the paer. The main hyperparameter is listed as follows:

`global epochs` = 1000

`local learning rate` = 0.005

`number of clients` = 20

`eta` = 1

`s` = 80%

`p`  =2

## Quick Start
Before starting experiments, make sure you are in the right working directory.
You need to get the heterogeneous datasets from [https://github.com/TsingZ0/PFLlib] and save it at `\dst`.

```commandline
cd /FedALA
```

You can use following commands to start your experiments.

```commandline
python main.py  -nc 20 -data cifar10 -et 1 -p 2 -s 80 
```
Note that only the cpu version is supported currently