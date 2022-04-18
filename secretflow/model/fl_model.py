# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""FedModel定义

"""

from typing import Callable, List, Union
import math
import os

import tensorflow as tf
from torch import nn as nn
from loguru import logger

from secretflow.data.ndarray import FedNdarray
from secretflow.device import PYU
from secretflow.data.ndarray import FedNdarray
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.model.fl_base import PYUTFModel

from secretflow import reveal


class FLTFModel:
    def __init__(self, device_list: List[PYU] = [], model: Callable[[], tf.keras.Model] = None, aggregator=None):

        self._workers = {device: PYUTFModel(
            model, device=device) for device in device_list}
        self._aggregator = aggregator
        self.steps_per_epoch = 0

    def handle_data(self, train_x: Union[HDataFrame, FedNdarray],
                    train_y: Union[HDataFrame, FedNdarray],
                    batch_size=32,
                    shuffle=False,
                    epochs=1,
                    sample_weight:  Union[FedNdarray, HDataFrame] = None):
        if isinstance(train_x, HDataFrame):
            train_x = train_x.values
        if isinstance(train_y, HDataFrame):
            train_y = train_y.values
        if isinstance(sample_weight, HDataFrame):
            sample_weight = sample_weight.values
        parties_length = train_x.length()
        max_data_size = max(
            [length for device, length in parties_length.items()])

        self.steps_per_epoch = math.ceil(max_data_size / batch_size)
        for device, worker in self._workers.items():
            # 当多方数据量不一致时，steps_per_epoch由数据量最大的一方决定，其他方将会重复取数据，repeat count为需要重复取的轮数
            repeat_count = epochs * \
                math.ceil(max_data_size*1.0/parties_length[device])
            if sample_weight is not None:
                sample_weight_partition = sample_weight.partitions[device]
            else:
                sample_weight_partition = None
            worker.build_dataset(train_x.partitions[device],
                                 train_y.partitions[device],
                                 s_w=sample_weight_partition,
                                 batch_size=batch_size,
                                 buffer_size=batch_size * 8,
                                 shuffle=shuffle,
                                 repeat_count=repeat_count)

    def scatter(self, x):
        return {device: device(lambda x: x)(x) for device, worker in self._workers.items()}

    def fit(self, x: Union[HDataFrame, FedNdarray], y: Union[HDataFrame, FedNdarray], batch_size=32, epochs=1, verbose='auto',
            callbacks=None, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, validation_freq=1, aggregate_freq=1):
        """水平联邦训练接口

        Args:
               x: feature，接受FedNdArray 或者 HDataFrame
               y: label，接受FedNdArray 或 或者 HDataFrame
               batch_size: 批处理大小，接受整数
               epochs: 训练回合数
               verbose: 是否在过程中显示性能指标, 0为不显示, 1为显示
               callbacks:
               validation_data: 验证集数据，不提供fit内部进行数据验证集拆分的功能
               validation_freq: 验证频率, 默认为1, 表示每个epoch后进行验证
               sample_weight: 接受 1D Array sample的权值，含义同keras, 表示在sample上获得的梯度进行加权聚合
               class_weight: 接受字典，用来映射类别和相应的权重，用于计算加权损失函数（仅训练阶段）
        Returns:
            本地训练后的最新参数
        """
        # sanity check
        assert isinstance(validation_freq, int) and validation_freq >= 1
        assert isinstance(aggregate_freq, int) and aggregate_freq >= 1
        report_list = []
        # 构建dataset
        if isinstance(x, HDataFrame) and isinstance(y, HDataFrame):
            train_x, train_y = x.values, y.values
        else:
            train_x, train_y = x, y

        if validation_data is not None:
            valid_x, valid_y = validation_data[0], validation_data[1]
        else:
            valid_x, valid_y = None, None

        self.handle_data(train_x, train_y, sample_weight=sample_weight, batch_size=batch_size,
                         shuffle=shuffle, epochs=epochs)

        # 初始化 weights
        current_weights = {
            device: worker.get_weights() for device, worker in self._workers.items()}

        for epoch in range(epochs):
            # do train
            report_list.append(f"epoch {epoch}")
            for step in range(0, self.steps_per_epoch, aggregate_freq):
                weights, sample_nums = [], []
                for device, worker in self._workers.items():
                    weight, sample_num = worker.train_step(current_weights[device],
                                                           epoch*self.steps_per_epoch+step,
                                                           aggregate_freq if step + aggregate_freq < self.steps_per_epoch else self.steps_per_epoch - step)
                    weights.append(weight)
                    sample_nums.append(sample_num)
                current_weight = self._aggregator.average(
                    weights, axis=0, weights=sample_nums)
                current_weights = self.scatter(current_weight)

            if epoch == 0:
                metric_names = reveal([worker.get_metric_name()
                                       for device, worker in self._workers.items()][0])
            metrics = [worker.get_metrics()
                       for device, worker in self._workers.items()]
            global_metrics = self._aggregator.average(
                metrics, axis=0)
            for name, metric in zip(metric_names, global_metrics):
                report_list.append(f"train-{name}:{metric} ")
            if epoch % validation_freq == 0 and valid_x is not None:
                valid_eval = self.evaluate(
                    valid_x, valid_y, batch_size=batch_size, verbose=0, sample_weight=sample_weight)
                for name, metric in zip(metric_names, valid_eval):
                    report_list.append(f"valid-{name}:{metric} ")
            report = " ".join(report_list)
            logger.info(report)

    def predict(self, x: Union[HDataFrame, FedNdarray], batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False):
        """水平联邦离线预测接口

        Args:
               x: feature，接受FedNdArray 或者 HDataFrame
               batch_size: 批处理大小，接受整数
               verbose: 是否在过程中显示性能指标, 0为不显示, 1为显示
               callbacks:
               max_queue_size: 验证集数据，不提供fit内部进行数据验证集拆分的功能
               validation_freq: 验证频率, 默认为1, 表示每个epoch后进行验证
               sample_weight: sample的权值，含义同keras, 表示在sample上获得的梯度进行加权聚合
        Returns:
            离线预测后的结果，numpy.array
        """
        y_pred = [worker.predict(x.partitions[device].data, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size,
                                 workers=workers, use_multiprocessing=use_multiprocessing)
                  for device, worker in
                  self._workers.items()]
        return y_pred

    def evaluate(self, x: Union[HDataFrame, FedNdarray], y: Union[HDataFrame, FedNdarray] = None, batch_size: int = None, sample_weight: Union[HDataFrame, FedNdarray] = None, verbose=0, steps=None):
        """水平联邦离线评估接口

        Args:
               x: feature，接受FedNdArray 或者 HDataFrame
               y: label，接受FedNdArray 或 或者 HDataFrame
               batch_size: 批处理大小，接受整数
               sample_weight: sample的权值，含义同keras, 表示在sample上获得的梯度进行加权聚合
               verbose: 是否在过程中显示性能指标, 0为不显示, 1为显示
               steps: 需要评估哪一个step的指标
        Returns:
            离线预测后的结果，numpy.array
        """
        if isinstance(x,  HDataFrame):
            x = x.values
        if isinstance(y, HDataFrame):
            y = y.values
        if isinstance(sample_weight, HDataFrame):
            sample_weight = sample_weight.values
        metrics = []
        for device, worker in self._workers.items():
            if sample_weight is not None:
                s_w_partition = sample_weight.partitions[device]
            else:
                s_w_partition = None
            metric = worker.evaluate(x.partitions[device],
                                     y.partitions[device],
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     sample_weight=s_w_partition,
                                     steps=steps)
            metrics.append(metric)

        # TODO: global agg for each evaluation
        global_metrics = self._aggregator.average(metrics, axis=0)
        return global_metrics

    def save_model(self, model_path: str = None, is_test=False):
        """联邦学习保存模型接口

            Args:
                model_path: model路径
                is_test: 是否本机模拟测试
            """
        assert model_path is not None, "model path cannot be empty"
        res = {}
        for device, worker in self._workers.items():
            if is_test:
                model_path_test = os.path.join(
                    model_path, device.__str__().strip("_"))
                res[device] = worker.save_model(model_path_test)
            else:
                res[device] = worker.save_model(model_path)
        reveal(res)

    def load_model(self, model_path: str = None, is_test=False):
        """联邦学习加载模型接口

            Args:
                model_path: model路径
                is_test: 是否本机模拟测试
            """
        assert model_path is not None, "model path cannot be empty"
        res = {}
        for device, worker in self._workers.items():
            if is_test:
                model_path_test = os.path.join(
                    model_path, device.__str__().strip("_"))
                res[device] = worker.load_model(model_path_test)
            else:
                res[device] = worker.load_model(model_path)
        reveal(res)
