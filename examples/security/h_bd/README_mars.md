# Introduction

This folder contains the implementations of the MARS defense against backdoor attacks in horizontal Federated Learning.

# Defense Method

MARS from [MARS: A Malignity-Aware Backdoor Defense in Federated Learning](https://openreview.net/pdf?id=O34CXUAZ0E)

MARS is a novel defense approach that identifies and mitigates backdoor models in Federated Learning by focusing on the malignity of neurons within local models. Unlike existing schemes that directly detect abnormal statistical measures based on model parameters, MARS calculates the backdoor energy (BE) of each neuron, reflecting how strongly a neuron is associated with backdoor attacks. Figure 1 gives the overview of MARS.
<img width="1119" alt="image" src="https://github.com/user-attachments/assets/b6af4f0f-6d5b-4ea7-b924-c17a87276667" />


# Algorithm Description

The high-level overview of MARS is illustrated in the following steps:

1. **Backdoor Energy Calculation:**  
   For each local model, calculate the BE of each neuron using the upper bound of BE, which can be easily calculated using only the model parameters.

2. **Concentrated Backdoor Energy Extraction:**  
   Extract the highest BE values from each layer and concatenate them into a CBE vector.

3. **Wasserstein Distance-Based Clustering:**  
   Use K-WMeans (K-Means with Wasserstein distance) to partition the CBEs of all local models into two clusters. Select the cluster with the smaller center norm as the trusted cluster, ensuring that backdoor models are excluded from the aggregation.

# Implementation

- MARS defense implementation: agg_mars.py
- The defense ability of MARS against model replacement backdoor attack: test_torch_mars.py

# Test

1. **Test MARS on CIFAR-10 Dataset:**  
   Use the following command to run MARS defense tests under a simulated federated environment:
   
```bash
pytest --env sim -n auto -v --capture=no tests/ml/nn/fl/defense/test_torch_mars.py

```

# Test results

After 30 epochs of training on CIFAR-10 (backdoor attack starts at 10 epochs, poison_rate is 0.01), the accuracy is 0.5170 and ASR is 0.0990 (close to random probability of 10 classifications, 0.1). The results demonstrate that MARS effectively mitigates the impact of backdoor attacks while maintaining the performance of the aggregated model.

<img width="2416" alt="image" src="https://github.com/user-attachments/assets/fed07201-55d6-4134-9a5c-5888baa76855" />
