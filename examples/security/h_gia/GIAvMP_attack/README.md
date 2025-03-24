# Introduction

This folder contains the implemention of the gradient inversion attack via model poisoning (GIAvMP).

## Algorithm Description

GIAvMP is an active gradient inversion attack, which poisons the local model of the victim party in FL to improve the attack performance. The overview of GIAvMP is demonstrated in Figure 1.

<p align="center">
    <img width="500" height="300" src="https://github.com/skylineZSS/rep-img/raw/main/overview.png" alt>
</p>
<p align="center">
    <em>Figure 1: The gradient flow in a quadratic model and the construction of a surrogate model. </em>
</p>

The process of GIAvMP:

1. The server (attacker) constructs malicious model parameters, which can mitigate the confusion of aggregated gradients computed over the poisoned model.
2. The server sends malicious model parameters to the victim client.
3. The victim client updates local model and conducts local training.
4. The victim client shares local gradients with the server.
5. The server recovers local training data from the shared gradients.

If the model is a FCNN, the server can directly recover the raw data from the gradients of the 1st FC layer. The raw data $x$ can be obtained by
$$
x=\frac{\partial L}{\partial w_i}/\frac{\partial L}{\partial b_i}
$$
where $\frac{\partial L}{\partial w_i}$ and $\frac{\partial L}{\partial b_i}$ are the gradients w.r.t the weights and bias of the $i^{th}$ neuron. The malicious parameters ensure that the gradients of the $i^{th}$ neuron is only connected with data $x$.

if the model is a CNN, the process of GIAvMP is demonstrated in Figure 2.
<p align="center">
    <img width="750" height="300" src="https://github.com/skylineZSS/rep-img/raw/main/attack_on_CNN.png" alt>
</p>
<p align="center">
    <em>Figure 1: The gradient flow in a quadratic model and the construction of a surrogate model. </em>
</p>
GIAvMP poison the 1st FC layer in CNN model to obtain the feature of the raw data after the convolutional layers. Then GIAvMP uses DLG to recover the raw data from the raw features.

## Attack Metric

- Mean Square Error (MSE)

    MSE is the average L2 difference between the reconstructed image $\hat{x}$ and its corresponding real image $x$. MSE is computed as
    $$
    MSE(x, \hat{x})=\frac{\|\hat{x}-x \Vert_2}{dim(x)},
    $$
    where $dim(x)$ is the dimensions of $x$.

- Peak Signal-to-Noise Ratio (PSNR)

    PSNR is usually applied to measure the quality of the reconstructed images, which is computed as
    $$
    PSNR(\hat{x},x)=10\times\log_{10}(\frac{1}{MSE(\hat{x},x)})
    $$

## Attack performance

Dataset: CIFAR10

settings: "trainMP": True, "model": FCNNmodel, "k": 32, "batchsize": 32, "epochs": 1, "train_lr": 1, "epochs_for_trainMP": 10, "epochs_for_DLGinverse": 8000, "ratio_aux_dataset": 1.0

The attack performance on the FCNN model is shown below. The $B$ is the batchsize used in the victim's local training. A larger $B$ can increase the gradient confusion, which mitigates the attack performance.

| B | PSNR(dB)| mse |
| :-:  |  :-: |  :-: |
| 16  |  67 |  0.00014 |
| 32  |  59 |  0.0024 |
| 64  |  44.4 |  0.0086 |
| 128  |  25.93 |  0.019 |

The comparison of the raw images and the recovered images is shown in Figure 3.
<p align="center">
    <img width="750" height="200" src="https://github.com/skylineZSS/rep-img/raw/main/FCNN_res.png" alt>
</p>
<p align="center">
    <em>Figure 3: GIAvMP attack on a FCNN model. The comparison of the raw images (odd column) and the recovered images (even column).  </em>
</p>

The attack performance of GIAvMP on the CNN model is shown in Figure 4. The batchsize is 32.
<p align="center">
    <img width="400" height="300" src="https://github.com/skylineZSS/rep-img/raw/main/CNN_res.png" alt>
</p>
<p align="center">
    <em>Figure 4: GIAvMP attack on a CNN model. The comparison of the raw images (odd column) and the recovered images (even column).  </em>
</p>

The attack performance of DLG on the CNN model is shown in Figure 5. The batchsize is 32.
<p align="center">
    <img width="400" height="300" src="https://github.com/skylineZSS/rep-img/raw/main/dlgB32CNN.png" alt>
</p>
<p align="center">
    <em>Figure 5: DLG attack on a CNN model. The comparison of the raw images (odd column) and the recovered images (even column).  </em>
</p>

## Defense

### Differential Privacy

Add DP noise into the shared gradients can defend against GIAvMP, but the model accuracy will decrease.

The GIAvMP attack performance under different DP noise scale $\sigma$ is shown below.

| $\sigma$ | PSNR(dB)| mse | model acc(%)|
| :-:  |  :-: |  :-: | :-: |
| 0.05  |  -17.01 |  63.74 |79.42 |
| 0.01  |  -1.31 |  4.25 |79.81|
| 0.001 |  -1.14 |  1.31 |80.01|
| 0.0  |  59 |  0.0024 |80.29|

The attack performance of GIAvMP when applying DP in FL.
<p align="center">
    <img width="750" height="200" src="https://github.com/skylineZSS/rep-img/raw/main/FCNN_res_DP.png" alt>
</p>
<p align="center">
    <em>Figure 6: GIAvMP attack on a FCNN model when applying DP. The comparison of the raw images (odd column) and the recovered images (even column).  </em>
</p>

### Detect the malicious parameters

The malicious parameters will lead to a sharp decrease of the model accuracy. The victim can leverage this phenomenon to detect if the model is poisoned.
