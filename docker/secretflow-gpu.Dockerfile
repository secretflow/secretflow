FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN  apt-get update \
     && apt-get install -y libcudnn8=8.6.0.163-1+cuda11.8 --allow-downgrades --allow-change-held-packages  \
     && apt-get install -y python3.8 --allow-downgrades --allow-change-held-packages   \
     && apt-get install -y python3-pip --allow-downgrades --allow-change-held-packages 

RUN if [ ! -e /usr/bin/python ]; then ln -sf /usr/bin/python3.8 /usr/bin/python; fi

RUN if [ ! -e /usr/bin/python3 ]; then ln -sf /usr/bin/python3.8 /usr/bin/python3; fi

#install the dependencies of cuda11
#you are supposed to  add the mirror source of pypi to accelerate installation of nvidia packages of cuda11, 
#if not, the building of images are prone to fail very much 
RUN pip install nvidia-cublas-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvcc-cu11 \
    nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 install nvidia-cudnn-cu11 \
    nvidia-cufft-cu11  nvidia-curand-cu11  nvidia-cusolver-cu11 \
    nvidia-cusparse-cu11 nvidia-nccl-cu11  nvidia-nvtx-cu11 \
    && rm -rf  ~/.cache/pip \
    && rm -rf /tmp/*

# install the gpu version of jax and jaxlib based cuda11
# the site of https://storage.googleapis.com/jax-releases/jax_cuda_releases.html is very necessary
# ref to https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier
RUN pip install --upgrade "jax[cuda11_pip]"==0.4.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip install --upgrade "jaxlib[cuda11_pip]"==0.4.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && rm -rf  ~/.cache/pip \
    && rm -rf /tmp/*

#install secretflows and its dependencies.
#you are supposed to add the mirror source of pypi to accelerate installation of SecretFlow and accelerate the building of images
#if not, the building of images are prone to fail very much
# Now, based on the CUDA11, the best match of TensorFlow, PyTorch and Jax are
# tensorflow==2.12.0, due to the version of TensorFlow which secretflow  requires is 2.11.0, so we install tensorflow==2.12.0 manually.
# torch==2.0.0
# jax==0.4.1
RUN pip install -U secretflow \
    && pip install tensorflow==2.12.0 \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install protobuf==3.20.3 \
    && rm -rf  ~/.cache/pip \
    && rm -rf /tmp/*
    
COPY secretflow_entrypoint.sh /opt/secretflow/

COPY secretflow_entrypoint.py /opt/secretflow/



ENTRYPOINT ["sh","/opt/secretflow/secretflow_entrypoint.sh"]
