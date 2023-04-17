FROM tensorflow/tensorflow:2.10.1-gpu
#install the dependencies of cuda11
#you are supposed to  add the mirror source of pypi to accelerate installation of nvidia packages of cuda11, 
#if not, the building of images are prone to fail very much 
RUN pip install nvidia-cublas-cu11 \
    && pip install nvidia-cuda-cupti-cu11 \
    && pip install nvidia-cuda-nvcc-cu11 \
    && pip install nvidia-cuda-nvrtc-cu11 \
    && pip install nvidia-cuda-runtime-cu11 \
    && pip install nvidia-cudnn-cu11 \
    && pip install nvidia-cufft-cu11 \
    && pip install nvidia-curand-cu11 \
    && pip install nvidia-cusolver-cu11 \
    && pip install nvidia-cusparse-cu11 \
    && pip install nvidia-nccl-cu11 \
    && pip install nvidia-nvtx-cu11 \
# install the gpu version of jax and jaxlib based cuda11
# the site of https://storage.googleapis.com/jax-releases/jax_cuda_releases.html is very necessary
# ref to https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier
RUN pip install --upgrade "jax[cuda11_pip]"==0.4.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip install --upgrade "jaxlib[cuda11_pip]"==0.4.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#install secretflow
#you are supposed to add the mirror source of pypi to accelerate installation of SecretFlow and accelerate the building of images
#if not, the building of images are prone to fail very much 
RUN pip install -U secretflow
