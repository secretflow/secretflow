# Installation

Secretflow is available in two editions: Lite and Full. The Lite edition is optimized for minimal size by excluding deep-learning related dependencies, making it more compact. On the other hand, the Full edition encompasses the complete set of dependencies for users requiring the full functionality of deep learning integration. Select the edition that best aligns with your specific requirements.

The simplest way to try SecretFlow is to use [offical docker image](#option-2-from-docker) which ships with SecretFlow binary.

Or you could [install SecretFlow via Python Package Index](#option-1-from-pypi).

For advanced users, you could [install SecretFlow from source](#option-3-from-source).

For Windows users, you could [install SecretFlow base WSL2](#option-4-from-wsl).

After installation, don't forget to [have a quick try](#a-quick-try) to check if SecretFlow is good to go.

> Additional: For users with available GPU devices, you could [try GPU support](#gpus-support).

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

- Full edition
```bash
pip install -U secretflow
```

- Lite edition
```bash
pip install -U secretflow-lite
```

## Option 2: from docker
You can also use SecretFlow Docker image to give SecretFlow a quick try.

- Full edition
```bash
docker run -it secretflow/secretflow-anolis8:latest
```

- Lite edition
```bash
docker run -it secretflow/secretflow-lite-anolis8:latest
```

More versions can be obtained from [secretflow tags](https://hub.docker.com/r/secretflow/secretflow-anolis8/tags).

## Option 3: from source

1. Download code and set up Python virtual environment.

```sh
git clone https://github.com/secretflow/secretflow.git
cd secretflow

conda create -n secretflow python=3.8
conda activate secretflow
```

2. Install SecretFlow

- Full edition
```sh

python setup.py bdist_wheel

pip install dist/*.whl
```

- Lite edition
```sh

python setup.py bdist_wheel --lite

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
    
    - Full edition
    ```
    pip install -U secretflow
    ```
    - Lite edition
    ```
    pip install -U secretflow-lite
    ```

4. Use WSL to develop your application


After set up of SecretFlow in WSL, you can use [Pycharm Professional to Configure an interpreter using WSL](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html) or [Visual Studio Code with WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode) to use SecretFlow in Windows Operating System.

## A quick try

Try your first SecretFlow program.

Import secretflow package.

```python
>>> import secretflow as sf
```

Create a local cluster with parties alice, bob and carol.

```python
>>> sf.init(parties=['alice', 'bob', 'carol'], address='local')
```

Create alice's PYU device, which can process alice's data.

```python
>>> alice_device = sf.PYU('alice')
```

Let alice say hello world.
```python
>>> message_from_alice = alice_device(lambda x:x)("Hello World!")
```

Print the message.
```python
>>> message_from_alice
<secretflow.device.device.pyu.PYUObject object at 0x7fdec24a15b0>
```

We see that the message on alice device is a PYU Object at deriver program.

Print the text at the driver by revealing the message.

```python
>>> print(sf.reveal(message_from_alice))
Hello World!
```

## GPU support

### Before you read

If you don't need to use GPU, please ignore this section and refer to [quick try](#a-quick-try).

### Introduction

NVIDIA's CUDA and cuDNN are typically used to accelerate the training and inference of machine learning models. Tensoflow and PyTorch, two widely-used machine learning frameworks, both intergrate the GPU support. In SecretFlow, PyTorch and Tensorflow are adopted as the backends for Federated Learning, of which the performance can be boosted with GPU support.

If you want to use GPU acceleration in SecretFlow, you need to complete the [Preparations](#preparations) first to set up the environment.

In the following, there are two options to run the GPU-version SecretFlow:

1. Use the [offical GPU docker image](#option-1-get-the-gpu-docker-image-from-the-secretflow-repository)

2. [Build the GPU docker image by yourself](#option-2-build-the-gpu-docker-image-by-yourself).

After the image is ready, you could [run the container and try GPU support](#run-a-container-and-check-gpu).

### Preparations
1. Make sure your NVIDIA driver is available and meet the version requirements:

 Driver version must be >= 525.60.13 for CUDA 12 and >= 450.80.02 for CUDA 11 on Linux.

You could run NVIDIA System Management Interface (nvidia-smi) to make sure your NVIDIA driver is available and meet the version requirements.

```bash
nvidia-smi
```
> **NOTE**: We currently only supply the GPU Docker image based on CUDA11. When the GPU packages of PyTorch and TensorFlow based on CUDA12 are available, we will supply the GPU Docker image based on CUDA12.

2. Follow the [NVIDIA official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to setup NVIDIA Container Toolkit on your distributions.

After the environment is set up, you could build/run the image.

### Option 1: Get the GPU docker image from the SecretFlow repository

The GPU Docker image of SecretFlow is available on the SecretFlow repository at Dockerhub and you can run the following command to get the latest GPU docker image.

```bash
docker pull secretflow/secretflow-gpu
```
For more information, please visit [the GPU docker images at Dockerhub](https://hub.docker.com/r/secretflow/secretflow-gpu).

### Option 2: Build the GPU docker image by yourself
You could also build the Docker image by yourself.

1. Download code

```bash
git clone https://github.com/secretflow/secretflow.git
cd secretflow/docker
```

2. Use a dockerfile file to construct the image

```bash
docker build -f  secretflow-gpu.Dockerfile -t secretflow-gpu .
```

### Run a container and Check GPU

1. Run a container

```bash
docker container run --runtime=nvidia  -it --gpus all secretflow-gpu bash
```

> **NOTE**: The following two parameters are necessary:
> - `--runtime=nvidia`
> - `--gpus all`

2. After the container is running, you can use the jupyter notebook [GPU Check](../tutorial/GPU_check.ipynb) to check the access of Tensorflow and PyTorch for NVIDIA GPUs inside the container.