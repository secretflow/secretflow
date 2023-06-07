# Installation

The simplest way to try SecretFlow is to use [offical docker image](#option-2-from-docker) which ships with SecretFlow binary.

Or you could [install SecretFlow via Python Package Index](#option-1-from-pypi).

For advanced users, you could [install SecretFlow from source](#option-3-from-source).

For Windows users, you could [install SecretFlow base WSL2](##Option-4-from-WSL).

After installation, don't forget to [have a quick try](#a-quick-try) to check if SecretFlow is good to go.


## Environment

Python：3.8

pip: >= 19.3

OS: CentOS 7, Ubuntu 18.04

CPU/Memory: recommended minimum requirement is 8C16G.

## Option 1: from pypi
For users who want to try SecretFlow, you can install [the current release](https://pypi.org/project/secretflow/).

Note that it requires python version == 3.8, you can create a virtual environment with conda if not satisfied.

```
conda create -n sf python=3.8
conda activate sf
```

After that, please use pip to install SecretFlow.

```bash
pip install -U secretflow
```

## Option 2: from docker
You can also use SecretFlow Docker image to give SecretFlow a quick try.

The latest version can be obtained from [secretflow tags](https://hub.docker.com/r/secretflow/secretflow-anolis8/tags).

```
export version={SecretFlow version}
```

for example
```bash
export version=0.6.13b1
```

then run the image.
```bash
docker run -it secretflow/secretflow-anolis8:${version}

```

## Option 3: from source

1. Download code and set up Python virtual environment.

```sh
git clone https://github.com/secretflow/secretflow.git
cd secretflow

conda create -n secretflow python=3.8
conda activate secretflow
```

2. Install SecretFlow
```sh

python setup.py bdist_wheel

pip install dist/*.whl
```

## Option 4: from WSL

SecretFlow does not support Windows directly now, however, a Windows user can use secretFlow by WSL(Windows Subsystem for Linux).

1. Install WSL2 in Windows

- You are supposed to follow the [guide_zh](https://learn.microsoft.com/zh-cn/windows/wsl/install) or [guide_en](https://learn.microsoft.com/en-us/windows/wsl/install) to install WSL(Windows Subsystem for Linux) in your Windows and make sure that the version of WSL is 2.
- As for the distribution of GNU/Linux, Ubuntu is recommended.

2. Install Anaconda in WSL

Just follow the installation of anaconda in GNU/Linux to install anaconda in your WSL.

3. Install secretflow

- create conda environment

```shell
conda create -n sf python=3.8
```

- activate the environment

```shell
conda activate sf
```

- use pip to install SecretFlow.

```
pip install -U secretflow
```

4. Use WSL to develop your application


After set up of SecretFlow in WSL, you can use [Pycharm Professional to Configure an interpreter using WSL](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html) or [Visual Studio Code with WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode) to use SecretFlow in Windows Operating System.


## GPUs support

### Before you read

if you don't need the GPU support, please skip this session to [a quick try](#a-quick-try).

### Introduction

NVIDIA's CUDA and cuDNN are typically used to accelerate the training and testing of Tensoflow and PyTorch deep learning models. Tensoflow and PyTorch are both deep learning backends for SecretFlow.

[JAX](https://github.com/google/jax) is the frontend highly recommended by the SecretFlow team. JAX can compile and run your NumPy code on accelerators, such as GPUs and TPUs.Now we supply GPUs supports to JAX in SecretFlow.


if you want to use GPU acceleration in SecretFlow, you use offical GPU docker image or build the GPU docker image by yourself, However, whichever way you choose, you need to finish the [Preparations](###Preparations).Next, chose [option 1](###Option-1-get-the-GPU-docker-image-from-the-SecretFlow-repository) or [option 2](Option-2-build-the-GPU-docker-image-by-yourself) to get the GPU image. Finally, based on the image to run your container of GPU.

### Preparations
1. Make sure your NVIDIA driver is available and meet the version requirements as [JAX recommend](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).

- The version requirements:

 the driver on the host must be version >= 525.60.13 for CUDA 12 and >= 450.80.02 for CUDA 11 on Linux.

- Run NVIDIA System Management Interface (nvidia-smi) to make sure your NVIDIA driver is available and meet the version requirements.

```bash
nvidia-smi
```
- Since the GPU packages of PyTorch and TensorFlow are not available now, we only supply the GPU Docker image based on the CUDA11. When the GPU packages of PyTorch and TensorFlow based on the CUDA12 are available, we will supply the GPU Docker image based on the CUDA12. 

2. Follow the [NVIDIA official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to setup NVIDIA Container Toolkit on your distributions.

### Option 1: get the GPU docker image from the SecretFlow repository

The GPU Docker image of SecretFlow is available on the SecretFlow repository at Dockerhub and you can run the following command to get the latest GPU docker image.

```bash
docker pull secretflow/secretflow-gpu
```
For more information, please visit [the GPU docker images at Dockerhub](https://hub.docker.com/r/secretflow/secretflow-gpu).

### Option 2: build the GPU docker image by yourself
You also can build the Docker image by yourself.

1. Use a dockerfile file to construct an image

- Download code

```bash
git clone https://github.com/secretflow/secretflow.git
cd secretflow/docker
```

- Construct an image

```bash
docker build -f  secretflow-gpu.Dockerfile -t secretflow-gpu .
```

### Run an container and Check GPU
1. Run an container

```bash
docker container run --runtime=nvidia  -it --gpus all secretflow-gpu bash
```

- `--gpus all`: This parameter is essential.
- `--runtime=nvidia`: This parameter is essential.

2. After the container is running, you can use the jupyter notebook [GPU Check](../tutorial/GPU_check.ipynb) to check the callability of JAX, Tensorflow and PyTorch for NVIDIA GPUs inside the container.

## A quick try

Try your first SecretFlow program.

```python
>>> import secretflow as sf
>>> sf.init(['alice', 'bob', 'carol'], address='local')
>>> dev = sf.PYU('alice')
>>> import numpy as np
>>> data = dev(np.random.rand)(3, 4)
>>> data
<secretflow.device.device.pyu.PYUObject object at 0x7fdec24a15b0>
```
