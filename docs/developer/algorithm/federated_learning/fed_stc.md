# Strategy: FedSTC
## Overview
||Sparse method|Quant method|Residual|Encoding|Upstream|Downstream|
|---|---|---|---|---|---|---|
|FedSTC|topk|binarization|Yes|Golomb|Yes|Yes|
||Handle Non-IID|Handle Dropping/Skipping||Generality|||
||Fine //TODO|Caching and synchronizing||General||

The main motivation of FedSTC is to compress the communication between client and server. The main contributions are as follows:
1. Compared with the previous sparse work on upstream (client 2 server), FedSTC also sparses on downstream (server 2 client);
2. When only some clients participate in each round, a Weight Update Caching mechanism is provided on the server side. Each client must synchronize the latest model before participating in the next round of training, or lag behind global weights. updates; (I understand such motivation is that if only part of the updates are updated, the content to be transmitted can be sparse);
3. Quantization is added while sparse. The quantization method is Binarization. Only 3 numbers will appear in the final matrix, {![](https://latex.codecogs.com/svg.image?\inline&space;\small&space;\mu), 0, ![](https://latex.codecogs.com/svg.image?\inline&space;\small&space;\mu)};
4. Lossless Golomb Encoding is used on the sparse + quantized matrix;
## Design
### Sparsity（topk）
Only upstream sparse:
![math1](resources/fedstc_math_1.jpg)
Add downstream：
![math2](resources/fedstc_math_2.jpg)
A is the Residual status on the server side of the previous round;
### Caching
The server keeps the most recent historical updates:
![math3](resources/fedstc_math_3.jpg)
The latest global weights can be expressed as:
![math4](resources/fedstc_math_4.jpg)
When a client joins training again, it must update the corresponding ![](https://latex.codecogs.com/svg.image?\inline&space;\small&space;P^{(s)}) or ![](https://latex.codecogs.com/svg.image?\inline&space;\small&space;W) ;
### Binarization (quant -> ternary tensor) 
![](https://latex.codecogs.com/svg.image?\inline&space;\small&space;e'&space;\in&space;\{-\mu,0,\mu\},&space;\&space;\mu&space;=&space;mean(abs(e)))

Assuming that mu is the sum of the absolute values of all elements in the matrix after sparse, the non-zero elements in the matrix are binarized to ![](https://latex.codecogs.com/svg.image?\inline&space;\small&space;\mu) or ![](https://latex.codecogs.com/svg.image?\inline&space;\small&space;-\mu) according to the sign;

### Pseudo Code on Compression
![algo](resources/fedstc_algo_1.jpg)
### Lossless Encoding
Golomb Encoding
## Experiment
Experiment on different models + datasets:
|model|dataset|
|---|---|
|VGG11|CIFAR|
|CNN|KWS|
|LSTM|Fashion-MNIST|
|Logistic R|MNIST|
FedAvg is one of the baselines. In order to compare the transmission cost horizontally with FedSTC, FedAvg uses a delay period. For example, for FedSTC with sparse rate = 1/400, the delay period is 400 iterations;
**Experimental conclusion: FedSTC is obviously better than FedAvg in the case of (a) non-iid, (b) small batch size, (c) large number of participating clients but low participation in each round**
### on Non-iidness
#### outperforms FedAvg
![exp_1](resources/fedstc_exp_1.jpg)
#### on batch size
![exp_2](resources/fedstc_exp_2.jpg)
#### on drop rate
![exp_3](resources/fedstc_exp_3.jpg)
#### on data amount unbalanced
![exp_4](resources/fedstc_exp_4.jpg)
#### on convergence
![exp_5](resources/fedstc_exp_5.jpg)
## Implementation
1. The sparse+binarization in upstream and downstream has been implemented;
2. Caching is not implemented;
3. golomb/ encoding is not implemented;

## Reference
[Robust and Communication-Efficient Federated Learning From Non-i.i.d. Data](https://ieeexplore.ieee.org/document/8889996)