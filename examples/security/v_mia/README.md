# SplitFedLearning

This folder contains a split federated learning demo. Two clients and a server was implemented. Further, the [training-based Model Inversion Attack (MIA)](https://doi.org/10.1145/3319535.3354261) is implemented.

## Usage

Install Pytorch >=1.8.1

Run main.py to train the classifier using split federated learning. The model is trained for `train_epoches `epoches, and the parameters are saved every `log_epoch `epoches as `client_{epoch_id}.pkl` and `server_{epoch_id}.pkl`, respectively.

Run attack.py to perform MIA.

Run export.py to export both the private images and auxiliary images.

## Datasets

The MNIST dataset is used in this demo. We use the official MNIST handle in torchvision, which will automatically download the dataset. Other datasets can also be used by simply changing the handle into `ImageFolder`.

