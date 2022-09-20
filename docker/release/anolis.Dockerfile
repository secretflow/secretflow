FROM openanolis/anolisos:8.4-x86_64

LABEL maintainer="secretflow-contact@service.alipay.com"

RUN yum install -y git wget unzip which vim \
    && yum clean all 

COPY Miniconda3.sh /tmp/Miniconda3.sh

RUN bash /tmp/Miniconda3.sh -b \
    && ln -s /root/miniconda3/bin/conda /usr/bin/conda \
    && rm -f /tmp/Miniconda3.sh

COPY .condarc /root/.condarc
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && rm -f /tmp/environment.yml \
    && conda clean --all -f --yes \
    && rm -rf /root/.cache

RUN echo "source /root/miniconda3/bin/activate secretflow" > ~/.bashrc

WORKDIR /root

CMD ["/bin/bash"]