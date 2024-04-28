# Performance Analysis: SecureBoost vs XGBoost

This document provides a thorough examination of the performance distinctions between SecureBoost (SGB) and XGBoost. The analysis scrutinizes both the accuracy of the models and the time taken to train them.



## Introduction

The purpose of this analysis is to establish a baseline for the expected performance differential of SGB when juxtaposed with XGBoost. There is a recognized discrepancy between federated learning systems, such as SGB, and traditional localized learning algorithms. Through this study, we will assess the performance of SGB in relation to XGBoost across a spectrum of networking scenarios. It is anticipated that, depending on the conditions—ranging from Local Area Network (LAN) to Wide Area Network (WAN) environments—the time required for training SGB may be considerably greater, stretching from tenfold to even a hundredfold longer, compared to XGBoost.

### SecureBoost (SGB)

SecureBoost is a federated learning framework that enables secure computation of boosted trees. It provides privacy-preserving machine learning solutions while maintaining high accuracy, it was published by [Cheng et al.](https://arxiv.org/abs/1901.08755) and originally implemented in FATE. Secretflow provides a high performance implementation of SecureBoost, and named it as SGB.

### XGBoost

XGBoost is a widely popular open source library for gradient boosting. It has been widely adopted in industry and research.


## Methodology

There are many previous benchmark on SGB on large scale dataset. However, few are quickly replicable and consider the model performance or random factor of the model training process. This analysis will focus on replicability: it serves as a demo method for comparing the performance of SGB vs XGBoost in your case.

- **Data Sets Used**: Open source datasets like creditcard, bank marketing datasets
- **Software Setup**: SecretFlow 1.5.0a2, sf-heu 0.5.0b0, spu 0.9.0.dev20240415
- **Evaluation Metrics**: AUC and time consumption in seconds

Other Federated Learning Setups:

Two physical machines are used for the experiment. The bandwidth and latency are one way limitation applied on each of the two machines.
We modify the ip addresses used in the published secureboost analysis [notebook](../../tutorial/secureboost_analysis.ipynb) and then run the notebook.

The detailed parameters for SGB and XGBoost are included in the above notebook unchanged. Basically, 100 boosting rounds are set and early stopping is enabled.

The features are splitted equally into two parties, with one party holding an additional label column.

## Results

### Network Scenario Comparison

In this section we compare the performance of SGB and XGBoost under different network scenarios.

- **Network Scenarios**: LAN, 100mb/10ms, 100mb/20ms, 50mb/20ms, 10mb/50ms
- **Hardware Setup**: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.70GHz CPU：8C, Memory：16G


Table for XGB performance:

| datasets    | num samples | num features | XGB mean time | XGB mean auc |
|-------------|-------------|--------------|---------------|--------------|
| bank        | 45211       | 17           | 1.694         | 0.92         |
| creditcard  | 284807      | 30           | 1.83          | 0.96         |

Table for SGB performance:

| bandwidth and latency | datasets    | num samples | num features | SGB mean time | SGB mean auc | SGB/XGB time Ratio |
|-----------------------|-------------|-------------|--------------|---------------|--------------|--------------------|
| LAN                   | bank        | 45211       | 17           | 136.11        | 0.92         | 80.35              |
| LAN                   | creditcard  | 284807      | 30           | 106.45        | 0.96         | 58.17              |
| 100Mb/10ms            | bank        | 45211       | 17           | 195.09        | 0.92         | 115.17             |
| 100Mb/10ms            | creditcard  | 284807      | 30           | 183.34        | 0.96         | 100.19             |
| 100Mb/20ms            | bank        | 45211       | 17           | 208.05        | 0.92         | 122.82             |
| 100Mb/20ms            | creditcard  | 284807      | 30           | 185.07        | 0.96         | 101.13             |
| 50Mb/20ms             | bank        | 45211       | 17           | 245.58        | 0.92         | 144.97             |
| 50Mb/20ms             | creditcard  | 284807      | 30           | 257.09        | 0.96         | 140.49             |
| 10Mb/50ms             | bank        | 45211       | 17           | 601.67        | 0.92         | 355.18             |
| 10Mb/50ms             | creditcard  | 284807      | 30           | 851.03        | 0.96         | 465.04             |

### Hardware Scenario Comparison

In this section we compare the performance of SGB and XGBoost under different hardware in LAN condition.

Table for XGB performance:

| Hardware              | datasets    | num samples | num features | XGB mean time | XGB mean auc |
|-----------------------|-------------|-------------|--------------|---------------|--------------|
| 8369B 8C              | bank        | 45211       | 17           | 1.69          | 0.92         |
| 8369B 8C              | creditcard  | 284807      | 30           | 1.83          | 0.96         |
| 5975WX 32C            | bank        | 45211       | 17           | 3.83          | 0.92         |
| 5975WX 32C            | creditcard  | 284807      | 30           | 2.21   	   | 0.96         |

Table for SGB performance:

| Hardware              | datasets    | num samples | num features | SGB mean time | SGB mean auc | SGB/XGB time Ratio |
|-----------------------|-------------|-------------|--------------|---------------|--------------|--------------------|
| 8369B 8C              | bank        | 45211       | 17           | 136.11        | 0.92         | 80.35              |
| 8369B 8C              | creditcard  | 284807      | 30           | 106.45        | 0.96         | 58.17              |
| 5975WX 32C            | bank        | 45211       | 17           | 87.56         | 0.92         | 22.85              |
| 5975WX 32C            | creditcard  | 284807      | 30           | 38.85         | 0.96         | 17.56              |



## Performance Analysis

The analysis evaluates key performance metrics for SecureBoost (SGB) and XGBoost (XGB), focusing on mean training time and model accuracy measured by the Area Under the Receiver Operating Characteristic Curve (AUC).

### Impact of Network Conditions

Network conditions, specifically bandwidth and latency, have a significant impact on SGB's training time. Clear performance degradation is observed as network bandwidth narrows and latency increases. For example, with the 'bank' dataset, SGB's training time under LAN conditions is 136.11 seconds, while it escalates to 601.67 seconds when bandwidth and latency are restricted to 10Mb/50ms, resulting in an SGB/XGB time ratio surge from 80.35 to 355.18. XGB training, conducted locally, remains consistently efficient across all network scenarios.

### Observations Under Optimal Network Conditions

Even under optimal conditions (LAN with infinite bandwidth and zero latency), SGB trails behind XGB in terms of time efficiency. The time ratios persistently favor XGB, indicative of the additional computational load brought by SGB's federated learning approach and the use of Homomorphic Encryption.

### Consistency of Model Performance

Despite the evident differences in training duration, SGB achieves AUC scores comparable to XGB across varying network conditions. This consistent performance suggests SGB's robustness in preserving model quality while offering enhanced data privacy through federated learning techniques.

### Hardware Utilization Efficacy

The use of Intel(R) Xeon(R) Platinum 8369B CPU with performance optimizations such as AVX is better leveraged by XGB than SGB. Contrastingly, when employing the AMD Ryzen Threadripper PRO 5975WX with 32 cores, SGB shows a smaller performance hit and is only 10 times slower than XGB, a considerable improvement over the Intel hardware observations.

In summary, while federated learning through SecureBoost (SGB) introduces a considerable increase in model training time, especially under limited bandwidth and high latency conditions, it retains a similar level of accuracy as traditional local XGB training and provides the benefit of privacy-preserving data analysis across distributed networks.


## Conclusion

In summary, while federated learning through SecureBoost (SGB) introduces a notable increase in model training time, with time ratios ranging from  17 to as high as 465 under varying network and hardware conditions, it consistently maintains a level of accuracy comparable to traditional local XGBoost (XGB) training. Despite the substantial computational overhead in the most constrained network environments, SGB demonstrates the capacity to deliver privacy-preserving data analysis with manageable performance trade-offs between increased training duration and data security.


## References

SecureBoost: https://arxiv.org/abs/1901.08755

XGBoost: https://xgboost.readthedocs.io/en/latest/index.html

bank marketing dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing

creditcard dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data