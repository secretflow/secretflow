FROM openanolis/anolisos:8.8 AS builder

RUN yum install -y \
    wget gcc gcc-c++ autoconf bison flex git protobuf-devel libnl3-devel \
    libtool make pkg-config protobuf-compiler \
    && yum clean all

RUN cd / && git clone https://github.com/google/nsjail.git \
    && cd /nsjail && git checkout 3.3 -b v3.3 \
    && make && mv /nsjail/nsjail /bin

FROM secretflow/anolis8-python:3.10.13 AS python

FROM openanolis/anolisos:8.8

LABEL maintainer="secretflow-contact@service.alipay.com"

COPY --from=builder /bin/nsjail /usr/local/bin/
COPY --from=python /root/miniconda3/envs/secretflow/bin/ /usr/local/bin/
COPY --from=python /root/miniconda3/envs/secretflow/lib/ /usr/local/lib/

RUN yum install -y protobuf libnl3 libgomp && yum clean all

RUN grep -rl '#!/root/miniconda3/envs/secretflow/bin' /usr/local/bin/ | xargs sed -i -e 's/#!\/root\/miniconda3\/envs\/secretflow/#!\/usr\/local/g'

ARG sf_version

ENV version $sf_version

RUN pip install secretflow==${version} --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://test.pypi.org/simple/ && rm -rf /root/.cache

COPY .nsjail /root/.nsjail

COPY *.whl /tmp/
RUN pip install /tmp/*.whl --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://test.pypi.org/simple/ --no-cache-dir && rm -rf /tmp/*.whl

ARG config_templates=""
LABEL kuscia.secretflow.config-templates=$config_templates

ARG deploy_templates=""
LABEL kuscia.secretflow.deploy-templates=$deploy_templates

WORKDIR /root

COPY anolis_entrypoint.sh /opt/secretflow/

COPY anolis_entrypoint.py /opt/secretflow/

ENTRYPOINT ["sh","/opt/secretflow/anolis_entrypoint.sh"]
