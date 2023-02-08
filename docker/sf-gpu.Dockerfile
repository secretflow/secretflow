FROM tensorflow/tensorflow:2.10.1-gpu
RUN pip install -U secretflow -i https://pypi.doubanio.com/simple/
RUN pip install jupyter -i https://pypi.doubanio.com/simple/

VOLUME  /tf/
EXPOSE 8888
