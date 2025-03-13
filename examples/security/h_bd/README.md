# Introduction

This folder contains the implementions of backdoor attack on horizontal Federal Learning.

## Attack Method

Model Replacement Attack from [How To Backdoor Federated Learning](https://arxiv.org/pdf/1807.00459)

### Algorithm Description

Naive approach. 

The attacker can simply train its model on backdoored inputs. Each training batch should include a mix of correctly labeled inputs and backdoored inputs to help the model learn to recognize the difference. The attacker can also change the local learning rate and the number of local epochs to maximize the overfitting to the backdoored data.
The naive approach does not work against federated learning. Aggregation cancels out most of the backdoored model's contribution and the joint model quickly forgets the backdoor. The attacker needs to be selected often and even then the poisoning is very slow. 

Model replacement. 

In this method, the attacker ambitiously attempts to substitute the new global model $G^{t+1}$ with a malicious model $X$ as follows:

$$
X=G^t+\frac{\eta}{n} \sum_{i=1}^m\left(L_i^{t+1}-G^t\right)
$$

Because of the non-i.i.d. training data, each local model may be far from the current global model. As the global model converges, these deviations start to cancel out, i.e., $\sum_{i=1}^{m-1}\left(L_i^{t+1}-G^t\right) \approx 0$. Therefore, the attacker can solve for the model it needs to submit as follows:

$$
\widetilde{L}_m^{t+1}=\frac{n}{\eta} X-\left(\frac{n}{\eta}-1\right) G^t-\sum_{i=1}^{m-1}\left(L_i^{t+1}-G^t\right) \approx \frac{n}{\eta}\left(X-G^t\right)+G^t
$$

This attack scales up the weights of the backdoored model $X$ by $\gamma=\frac{n}{\eta}$ to ensure that the backdoor survives the averaging and the global model is replaced by $X$. 

An attacker who does not know $n$ and $\eta$ can approximate the scaling factor $\gamma$ by iteratively increasing it every round and measuring the accuracy of the model on the backdoor task. Scaling by $\gamma<\frac{n}{\eta}$ does not fully replace the global model, but the attack still achieves good backdoor accuracy.

# Implemention
  - `fl_model_bd.py`
  - `backdoor_fl_torch.py`
  - Test of Model replacement backdoor attack: `test_torch_backdoor.py`

# Test

1. Test SME attack on MNIST dataset
    - `pytest --env sim -n auto -v --capture=no tests/ml/nn/fl/attack/test_torch_backdoor.py`

# Test results

After 50 epochs of training on CIFAR-10 (backdoor attack start at 30 epochs, poison_rate is 0.1), accuracy is 0.5005 and ASR is 0.8083.