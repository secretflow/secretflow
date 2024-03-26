The implementation of the data reconstruction attack in Split learning -- FSHA and our defense -- Gradients Scrutinizer is under this path.

## Requirements
Our code is implemented and tested on python 3.6 and TensorFlow.

Install all the requirements by:

`pip install requirements.txt`

or install with conda by:

`conda env create -f requirements.yml`

`conda activate GradientsScrutinizer`

The script `test_FSHA.py` can be used to reproduct FSHA attack.

The script `test_FSHA_DP.py` can be used to reproduct FSHA attack under differential privacy defense.

The script `example.py` can be used to reproduct our experiments of detecting FSHA training from honest SL training with CIFAR 10 dataset or MNIST dataset. It can be done by:

`python example.py`

This script includes the training of FSHA client and server models and honest client and server models, and the detction score calculation.
