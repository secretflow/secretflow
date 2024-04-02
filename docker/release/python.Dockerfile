# secretflow/anolis8-python:3.10.13
FROM openanolis/anolisos:8.8

LABEL maintainer="secretflow-contact@service.alipay.com"

RUN  ARCH=$(uname -m) && \
     if [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    elif [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH"; \
        exit 1; \
    fi && \
    wget $MINICONDA_URL -O /tmp/Miniconda3.sh && \
    bash /tmp/Miniconda3.sh -b && ln -s /root/miniconda3/bin/conda /usr/bin/conda && \
    rm -f /tmp/Miniconda3.sh

RUN conda create --name secretflow python==3.10.13