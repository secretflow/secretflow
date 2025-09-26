# secretflow/anolis8-python:3.10.13
FROM openanolis/anolisos:8.8

ARG TARGETPLATFORM

RUN yum install -y  glibc wget && \
    yum clean all

LABEL maintainer="secretflow-contact@service.alipay.com"

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    fi && \
    wget -O /root/Miniconda3.sh "$MINICONDA_URL" && \
    mkdir ~/tmpconda; TMPDIR=~/tmpconda bash /root/Miniconda3.sh -b && \
    ln -s /root/miniconda3/bin/conda /usr/bin/conda && \
    rm -f /root/Miniconda3.sh

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create --name secretflow python==3.10.13