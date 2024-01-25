# secretflow/anolis8-python:3.8.18
FROM anolis-registry.cn-zhangjiakou.cr.aliyuncs.com/openanolis/anolisos:8.6

LABEL maintainer="secretflow-contact@service.alipay.com"

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /tmp/Miniconda3.sh

RUN bash /tmp/Miniconda3.sh -b \
    && ln -s /root/miniconda3/bin/conda /usr/bin/conda \
    && rm -f /tmp/Miniconda3.sh

RUN conda create --name secretflow python==3.8.18