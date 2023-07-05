# Vertically Federated XGB (SecureBoost)

In vertical federated learning scenarios, data is partitioned vertically according to features, meaning that each participant's data samples are consistent, but have different columns and types.

## Introduction to SecureBoost

Original paper: [SecureBoost](https://arxiv.org/abs/1901.08755)  

In vertical federated learning scenarios, each party has data with the same samples, but different feature spaces. SecureBoost prioritizes the protection of label holder's information and is designed to be as accurate as the original XGBoost algorithm. 

When compared to its MPC technology-powered counterpart, secret sharing-based XGB, SecureBoost is often faster. More specifically, SecureBoost is computationally more expensive than ss_xgb, but the latter is often bound by network bandwidth. In other words, SecureBoost is much faster when we have more CPU power but less network resources.

Our implementation of SecureBoost offers high performance and cutting-edge speed, supported by HEU devices.

## Tutorial

Please check out this simple [tutorial](../../../tutorial/SecureBoost.ipynb).

## Security Warning

Please note that the federated tree model algorithm [SecureBoost](https://arxiv.org/abs/1901.08755) is not a provably secure algorithm. There exist known [attacks](https://arxiv.org/pdf/2011.09290.pdf) that could lead to data leakage. Therefore, we recommend using [MPC-XGB](https://arxiv.org/abs/2005.08479) instead of SecureBoost when data security is a concern, which is implemented in [Decision Trees](../../mpc_ml/decision_tree.rst).
