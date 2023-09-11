# Introduction
This folder contains example of GAN-based LAtent Space Search (GLASS) aglorithm in paper ''GAN You See Me? Enhanced Data Reconstruction Attacks against Split Inference'' (under review).

The example in this repository are designed to demonstrate the basic attack process of GLASS.

# Brief
 - The folder `op` contains the operator implementation of StyleGAN2.
 - The folder `tmp` is used to store experimental data.
 - `model_resnet.py` and `model_stylegan2.py` are the implementations of ResNet and StyleGAN2.
 - `test.py` is the implementation of attack example.

# Download
To download the `target_model.pt`, `attack_model.pt` and `data.npz`, please use the following Baidu Netdisk link:
```
https://pan.baidu.com/s/1mXPpy2c68WPYCfAg6bJ2xw
```
The extraction code is **8bg7**.

Store the downloaded model and data files to the `tmp` folder.

# Run Example
```
python test.py
```

The attack result is shown in the `tmp` folder.

`truth_{i}.png` in `tmp` folder is the ground truth image.

`recon_{i}.png` in `tmp` folder is the reconstructed image.