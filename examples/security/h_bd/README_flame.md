# Introduction

This folder contains the implementations of the FLAME defense against backdoor attacks in horizontal Federated Learning.

# Defense Method

FLAME from [FLAME: Taming Backdoors in Federated Learning](https://www.usenix.org/system/files/sec22-nguyen.pdf)

FLAME is a dynamic defense approach that identifies and mitigates poisoned model updates in Federated Learning. Unlike fixed-cluster methods, FLAME adapts to the dynamic attack setting where the number of injected backdoors may vary between training rounds. Its key contributions include:

- **Filtering:**  
  FLAME computes the pairwise cosine distances between the weight vectors of local models and applies a dynamic clustering method (based on HDBSCAN) to identify outlier models that deviate significantly from the majority (i.e., models with high angular deviations).

- **Adaptive Clipping:**  
  To limit the impact of adversarial models that have artificially scaled up their updates, FLAME computes the Euclidean distances between the local updates and the previous global model. The median of these distances is used as an adaptive clipping bound. Each model update is then scaled down (clipped) so that its L2-norm does not exceed this bound.

- **Adaptive Noising:**  
  Gaussian noise is added to the aggregated model. The noise level is chosen adaptively based on the clipping bound and theoretical guarantees, ensuring that the noise is sufficient to eliminate backdoor influences while preserving the model's benign performance.

# Algorithm Description

<img width="475" alt="image" src="https://github.com/user-attachments/assets/e9be6556-31fe-4b06-8f48-b2d1ff64a626" />


The high-level overview of FLAME is illustrated in the above figure. The overall algorithm proceeds as follows:

1. **Client Update:**  
   The aggregator sends the current global model \( G_{t-1} \) to all clients, and each client computes a local model update \( W_i \) using its own data.

2. **Dynamic Model Filtering:**  
   The weight vectors of all \( W_i \) are used to compute pairwise cosine distances. FLAME leverages a dynamic clustering method (HDBSCAN) to identify and filter out outliers (potentially poisoned models) that exhibit large angular deviations from the benign majority.

3. **Adaptive Clipping:**  
   The Euclidean distances between each \( W_i \) and the previous global model \( G_{t-1} \) are calculated. The median of these distances is adopted as the adaptive clipping bound \( S_t \). For each admitted model, a clipping factor \( \gamma = \min\{1, S_t/e_i\} \) is applied to scale its update down if necessary:
   
   \[
   W^c_{i} = G_{t-1} + (W_i - G_{t-1}) \cdot \gamma
   \]

4. **Model Aggregation and Noising:**  
   The accepted (clipped) updates are aggregated via Federated Averaging to produce an intermediate global model \( G_t \). Finally, adaptive Gaussian noise \( N(0, \sigma^2) \) is added based on the clipping bound (with \(\sigma = \lambda \cdot S_t\)) to yield the final global model for round \( t \), denoted as \( G^*_t \).

# Implementation

- FLAME defense implementation: `agg_flame.py`
- The defense ability of FLAME against model replacement backdoor attack: `test_torch_flame.py`

# Test

1. **Test FLAME on CIFAR-10 Dataset:**  
   Use the following command to run FLAME defense tests under a simulated federated environment:
   ```bash
   pytest --env sim -n auto -v --capture=no tests/ml/nn/fl/defense/test_torch_flame.py

# Test results

After 30 epoch if training on CIFAR-10 (backdoor attack start at 10 epochs, poison_rate is 0.01), accuracy is 0.5038 and ASR is 0.0960 (close to random probability of 10 classifications, 0.1)

<img width="2416" alt="image" src="https://github.com/user-attachments/assets/036395f2-7fbb-4d4b-902d-82b8152b8658" />
