FROM secretflow/anolis8-python:3.8.15 as python

FROM anolis-registry.cn-zhangjiakou.cr.aliyuncs.com/openanolis/anolisos:8.6

LABEL maintainer="secretflow-contact@service.alipay.com"

COPY --from=python /root/miniconda3/envs/secretflow/bin/ /usr/local/bin/
COPY --from=python /root/miniconda3/envs/secretflow/lib/ /usr/local/lib/

RUN grep -rl '#!/root/miniconda3/envs/secretflow/bin' /usr/local/bin/ | xargs sed -i -e 's/#!\/root\/miniconda3\/envs\/secretflow/#!\/usr\/local/g'

ARG sf_version

ENV version $sf_version

RUN pip install secretflow==${version}

# For security reason.
# Since onnx-1.13.1's protobuf conflicts with TensorFlow-2.10.1's, 
# so we upgrade it manually.
RUN pip install onnx==1.13.1 protobuf==3.20.3 && rm -rf /root/.cache

WORKDIR /root

CMD ["/bin/bash"]