# Conda does not support CentOS 7 on Linux aarch64, so use CentOS 8
FROM centos:centos8

RUN cd /etc/yum.repos.d/ \
    && sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* \
    && sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-* \
    && yum update -y \
    && yum install -y dnf-plugins-core \
    && yum config-manager --set-enabled powertools \
    && yum clean all

RUN yum install -y \
    git vim-common wget unzip which java-11-openjdk-devel \
    libtool autoconf make ninja-build perl-IPC-Cmd patch \
    && yum clean all

# install conda
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b \
    && rm -f Miniforge3-$(uname)-$(uname -m).sh \
    && /root/miniforge3/bin/conda init

# Add conda to path
ENV PATH="/root/miniforge3/bin:${PATH}"

# Install lld
RUN /root/miniforge3/bin/conda install -c conda-forge lld nasm cmake gxx==11.4.0 clangxx -y \
    && /root/miniforge3/bin/conda clean -afy

# install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# install go
ARG GO_VERSION=1.22.4
ARG GO_SHA256SUM=a8e177c354d2e4a1b61020aca3562e27ea3e8f8247eca3170e3fa1e0c2f9e771
RUN url="https://golang.google.cn/dl/go${GO_VERSION}.linux-arm64.tar.gz"; \
    wget --no-check-certificate -O go.tgz "$url"; \
    echo "${GO_SHA256SUM} *go.tgz" | sha256sum -c -; \
    tar -C /usr/local -xzf go.tgz; \
    rm go.tgz;

ENV GOPATH="/usr/local"
ENV PATH="/usr/local/go/bin:${GOPATH}/bin:${PATH}"

# install bazel 
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-arm64 \
    && mv bazelisk-linux-arm64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel 

# run as root for now
WORKDIR /home/admin/

ENTRYPOINT [ "/bin/bash", "-lc" ]