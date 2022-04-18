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

"""SLModel定义

"""

from typing import Callable, Union, Dict, List
import os
import random
import math

import tensorflow as tf
import numpy as np
from loguru import logger
# from tqdm import tqdm

from secretflow.data.ndarray import FedNdarray
from secretflow.device import PYU
from secretflow.data.ndarray import FedNdarray
from secretflow.device.device.base import Device, reveal
from secretflow.model.sl_base import PYUSLTFModel
from secretflow.data.vertical import VDataFrame


class SLModelTF:
    def __init__(self, base_model_dict: Dict[Device, Callable[[], tf.keras.Model]] = {},
                 device_y: PYU = None,
                 model_fuse: Callable[[], tf.keras.Model] = None):

        self._workers = {device: PYUSLTFModel(
            model, None if device != device_y else model_fuse, device=device) for device, model in base_model_dict.items()}

        self.device_y = device_y

    def handle_data(self, x: Union[FedNdarray, VDataFrame, List],
                    y:  Union[FedNdarray, VDataFrame] = None,
                    sample_weight:  Union[FedNdarray, VDataFrame] = None,
                    batch_size=32, shuffle=False, epochs=1, stage="train", random_seed=1234):
        # 将VDataFrame转成FedNdarray
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        if isinstance(sample_weight, VDataFrame):
            sample_weight = sample_weight.values
        if isinstance(x, List):
            if isinstance(x[0], VDataFrame):
                x = [xi.values for xi in x]
        # 计算steps_per_epoch
        if isinstance(x, FedNdarray):
            parties_length = x.length()
        elif isinstance(x, List):
            parties_length = x[0].length()
        else:
            raise ValueError(
                f"Only can be FedNdarray or List, but got {type(x)} ")
        lengths = [length for device, length in parties_length.items()]
        assert len(set(lengths)) == 1, "length of all parties must be same"
        steps_per_epoch = math.ceil(lengths[0] / batch_size)
        for device, worker in self._workers.items():
            if device == self.device_y and y is not None:
                y_partitions = y.partitions[device]
                if sample_weight is not None:
                    s_w_partitions = sample_weight.partitions[device]
                else:
                    s_w_partitions = None
            else:
                y_partitions = None
                s_w_partitions = None
            if isinstance(x, List):
                worker.build_dataset(*[xi.partitions[device] for xi in x],
                                     y=y_partitions,
                                     s_w=s_w_partitions,
                                     batch_size=batch_size,
                                     buffer_size=batch_size * 8,
                                     shuffle=shuffle,
                                     repeat_count=epochs,
                                     stage=stage,
                                     random_seed=random_seed)
            else:
                worker.build_dataset(*[x.partitions[device]],
                                     y=y_partitions,
                                     s_w=s_w_partitions,
                                     batch_size=batch_size,
                                     buffer_size=batch_size * 8,
                                     shuffle=shuffle,
                                     repeat_count=epochs,
                                     stage=stage,
                                     random_seed=random_seed)
        return steps_per_epoch

    def fit(self, x: Union[VDataFrame, FedNdarray, List], y: Union[FedNdarray, VDataFrame], batch_size=32, epochs=1, verbose=0,
            callbacks=None, validation_data=None, shuffle=False, sample_weight=None, validation_freq=1):
        """拆分学习训练接口

        Args:
               x: feature，接受FedNdArray，或List[FedNdarray]
               y: label，接受FedNdArray
               batch_size: 批处理大小，接受整数
               epochs: 训练回合数
               verbose: 是否在过程中显示性能指标, 0为不显示, 1为显示
               callbacks:
               validation_data: 验证集数据，不提供fit内部进行数据验证集拆分的功能
               validation_freq: 验证频率, 默认为1, 表示每个epoch后进行验证
               sample_weight: sample的权值，含义同keras, 表示在sample上获得的梯度进行加权聚合

        """
        # sanity check
        assert isinstance(validation_freq, int) and validation_freq >= 1
        assert len(self._workers) == 2, "split learning only support 2 parties"
        assert isinstance(validation_freq, int) and validation_freq >= 1

        # 构建dataset
        train_x, train_y = x, y
        if validation_data is not None:
            logger.debug("validation_data provided")
            if len(validation_data) == 2:
                valid_x, valid_y = validation_data
                valid_sample_weight = None
            else:
                valid_x, valid_y, valid_sample_weight = validation_data
        else:
            valid_x, valid_y, valid_sample_weight = None, None, None
        random_seed = random.randint(0, 100000)
        steps_per_epoch = self.handle_data(train_x, train_y, sample_weight=sample_weight,
                                           batch_size=batch_size, shuffle=shuffle, epochs=epochs, stage="train", random_seed=random_seed)
        validation = False

        if valid_x is not None and valid_y is not None:
            validation = True
            valid_steps = self.handle_data(valid_x, valid_y, sample_weight=valid_sample_weight,
                                           batch_size=batch_size, epochs=epochs, stage="eval")
        res = {}
        for epoch in range(epochs):
            for step in range(0, steps_per_epoch):
                hiddens = []  # driver端
                for device, worker in self._workers.items():
                    hidden = worker.base_forward(stage="train")
                    hiddens.append(hidden.to(self.device_y))

                # 只有 y worker 要执行
                gradients = self._workers[self.device_y].fuse_net(*hiddens)

                idx = 0
                for device, worker in self._workers.items():
                    gradient = gradients[idx].to(device)
                    res[device] = worker.base_backward(gradient)
                    idx += 1
            if validation and epoch % validation_freq == 0:
                # validation
                for step in range(0, valid_steps):
                    hiddens = []  # driver端
                    for device, worker in self._workers.items():
                        hidden = worker.base_forward("eval")
                        hiddens.append(hidden.to(self.device_y))
                metrics = self._workers[self.device_y].evaluate(*hiddens)
                if verbose > 0:
                    logger.info(f"valid evaluate={reveal(metrics)}")
        reveal(res)

    @reveal
    def predict(self, x: Union[FedNdarray, List[FedNdarray]], batch_size=32, verbose=0):
        """拆分学习离线预测接口

        Args:
               x: feature，接受FedNdArray，或List[FedNdarray]
               y: label，接受FedNdArray
               batch_size: 批处理大小，接受整数
               verbose: 是否在过程中显示性能指标, 0为不显示, 1为显示
        """
        predict_steps = self.handle_data(
            x, None, batch_size=batch_size, stage="eval", epochs=1)
        # if verbose > 0:
        #     pbar = tqdm(total=predict_steps)
        #     pbar.set_description('Predict Processing:')
        for step in range(0, predict_steps):
            hiddens = []  # driver端
            for device, worker in self._workers.items():
                hidden = worker.base_forward(stage="eval", step=step)
                hiddens.append(hidden.to(self.device_y))
            # if verbose > 0:
            #     pbar.update(1)
        y_pred = self._workers[self.device_y].predict(*hiddens)
        return y_pred

    @reveal
    def evaluate(self, x: Union[FedNdarray, List[FedNdarray]], y: FedNdarray = None, batch_size: int = 32, sample_weight=None, verbose=1, steps=None):
        """拆分学习离线评估接口

        Args:
            x: feature，接受FedNdArray，或List[FedNdarray]
            y: label，接受FedNdArray
            batch_size: 批处理大小，接受整数
            sample_weight: 测试样本的可选 Numpy 权重数组，用于对损失函数进行加权。
            verbose: 是否在过程中显示性能指标, 0为不显示, 1为显示
            steps: 声明评估结束之前的总步数（批次样本）。默认值 None
        Returns:
            metrics: 返回联合评估结果
        """

        evaluate_steps = self.handle_data(
            x, y, sample_weight=sample_weight, batch_size=batch_size, stage="eval", epochs=1)
        metrics = None
        # if verbose > 0:
        #     pbar = tqdm(total=evaluate_steps)
        #     pbar.set_description('Evaluate Processing:')
        for step in range(0, evaluate_steps):
            hiddens = []  # driver端
            for device, worker in self._workers.items():
                hidden = worker.base_forward(stage="eval")
                hiddens.append(hidden.to(self.device_y))
            # if verbose > 0:
            #     pbar.update(1)
            metrics = self._workers[self.device_y].evaluate(*hiddens)
        return metrics

    def save_model(self, base_model_path: str = None, fuse_model_path: str = None, is_test=False):
        """拆分学习保存模型接口

        Args:
               base_model_path: base model路径
               fuse_model_path: fuse model路径
               is_test: 是否本机模拟测试
        """
        assert base_model_path is not None, "model path cannot be empty"
        assert fuse_model_path is not None, "model path cannot be empty"

        for device, worker in self._workers.items():
            if is_test:
                base_model_path_test = os.path.join(
                    base_model_path, device.__str__().strip("_"))
                worker.save_base_model(base_model_path_test)
            else:
                worker.save_base_model(base_model_path)
        res = self._workers[self.device_y].save_fuse_model(fuse_model_path)
        reveal(res)

    def load_model(self, base_model_path: str = None, fuse_model_path: str = None, is_test=False):
        """拆分学习加载模型接口

        Args:
               base_model_path: base model路径
               fuse_model_path: fuse model路径
               is_test: 是否本机模拟测试
        """
        assert base_model_path is not None, "model path cannot be empty"
        assert fuse_model_path is not None, "model path cannot be empty"

        for device, worker in self._workers.items():
            if is_test:
                base_model_path_test = os.path.join(
                    base_model_path, device.__str__().strip("_"))
                if not os.path.exists(base_model_path_test):
                    raise FileNotFoundError(
                        f"base model not exist, path={base_model_path_test}")
                worker.load_base_model(base_model_path_test)
            else:
                if not os.path.exists(base_model_path):
                    raise FileNotFoundError(
                        f"base model not exist, path={base_model_path}")
                worker.load_base_model(base_model_path)
        if not os.path.exists(fuse_model_path):
            raise FileNotFoundError(
                f"fuse model not exist, path={fuse_model_path}")
        res = self._workers[self.device_y].load_fuse_model(fuse_model_path)
        reveal(res)
