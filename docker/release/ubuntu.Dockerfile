FROM secretflow/ubuntu-base-ci:20250228 AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
        wget gcc g++ autoconf bison flex git \
        libprotobuf-dev libnl-3-dev libnl-route-3-dev libnl-genl-3-dev \
        libtool make pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Clone and build nsjail
RUN cd / && git clone https://github.com/google/nsjail.git \
    && cd /nsjail && git checkout 3.3 -b v3.3 \
    && make && mv /nsjail/nsjail /bin

FROM secretflow/ubuntu-base-ci:20250228

LABEL maintainer="secretflow-contact@service.alipay.com"

COPY --from=builder /bin/nsjail /usr/local/bin/

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y libprotobuf-dev libnl-3-200 libnl-genl-3-200 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ARG sf_version

ENV version $sf_version

RUN pip install secretflow==${version} --extra-index-url https://test.pypi.org/simple/ && rm -rf /root/.cache

COPY .nsjail /root/.nsjail

ARG config_templates=""
LABEL kuscia.secretflow.config-templates=$config_templates

ARG deploy_templates=""
LABEL kuscia.secretflow.deploy-templates=$deploy_templates

WORKDIR /root

COPY entrypoint.sh /opt/secretflow/

COPY entrypoint.py /opt/secretflow/

ENTRYPOINT ["sh","/opt/secretflow/entrypoint.sh"]