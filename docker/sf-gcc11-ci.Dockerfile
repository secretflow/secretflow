FROM openanolis/anolisos:8.6-x86_64


LABEL maintainer="secretflow-contact@service.alipay.com"

# GCC version
ARG DEVTOOLSET_VERSION=11

RUN yum makecache

# install devtools and [enable it](https://access.redhat.com/solutions/527703)
RUN yum install -y \
    gcc-toolset-${DEVTOOLSET_VERSION}-gcc \
    gcc-toolset-${DEVTOOLSET_VERSION}-gcc-c++ \
    gcc-toolset-${DEVTOOLSET_VERSION}-binutils \
    gcc-toolset-${DEVTOOLSET_VERSION}-libatomic-devel \
    gcc-toolset-${DEVTOOLSET_VERSION}-libasan-devel \
    gcc-toolset-${DEVTOOLSET_VERSION}-libubsan-devel \
    git wget unzip which libtool autoconf make \
    cmake ninja-build nasm python38 python38-devel vim-common java-11-openjdk-devel \
    && echo "source scl_source enable gcc-toolset-${DEVTOOLSET_VERSION}" > /etc/profile.d/enable_gcc_toolset.sh \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && yum clean all

# NOTE, ant aci will not source /etc/profile.d, so add scl to `/etc/profile.d/` does not work
ENV PATH /opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk/

# install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# rust 替换成国内源
RUN echo -e "\
[source.crates-io]\n\
registry = \"https://github.com/rust-lang/crates.io-index\"\n\
replace-with = \'ustc\'\n\
[source.ustc]\n\
registry = \"git://mirrors.ustc.edu.cn/crates.io-index\"\n\
" > /root/.cargo/config

RUN echo -e "\
[CentOSAppStream] \n\
name=CentOS-8.5.2111 - AppStream - mirrors.aliyun.com \n\
baseurl=http://mirrors.aliyun.com/centos-vault/8.5.2111/AppStream/\$basearch/os/ \n\
gpgcheck=0 \n\
gpgkey=http://mirrors.aliyun.com/centos/RPM-GPG-KEY-CentOS-Official \n\
" > /etc/yum.repos.d/CentOS-AppStream.repo

RUN dnf install -y epel-release \
    && yum install -y lcov \
    && yum clean all

# install bazel 
RUN wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh \
    && chmod +x ./bazel-5.1.1-installer-linux-x86_64.sh && ./bazel-5.1.1-installer-linux-x86_64.sh && rm -f ./bazel-5.1.1-installer-linux-x86_64.sh

# install python packages
COPY dev-requirements.txt /tmp/dev-requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN sed -i "s/tensorflow==/tensorflow-cpu==/g" /tmp/requirements.txt
RUN grep -v -E "^(spu==|sf-heu==)" /tmp/requirements.txt > /tmp/requirements2.txt && mv /tmp/requirements2.txt /tmp/requirements.txt
RUN python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple/ \
    && python3 -m pip config set global.extra-index-url "https://pypi.tuna.tsinghua.edu.cn/simple https://download.pytorch.org/whl/cpu" \
    && python3 -m pip config set install.trusted-host "pypi.tuna.tsinghua.edu.cn download.pytorch.org mirrors.bfsu.edu.cn" \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install wheel \
    && python3 -m pip install -r /tmp/requirements.txt \
    && python3 -m pip install -r /tmp/dev-requirements.txt \
    && python3 -m pip cache purge \
    && rm -f /tmp/requirements.txt \
    && rm -f /tmp/dev-requirements.txt

# run as root for now
WORKDIR /home/admin/


CMD ["/bin/bash"]
