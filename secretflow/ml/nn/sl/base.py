# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from secretflow.utils.communicate import ForwardData


class SLBaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def base_forward(self, stage: str, **kwargs):
        pass

    @abstractmethod
    def base_backward(self, gradient):
        pass

    @abstractmethod
    def fuse_net(self, hiddens):
        pass

    @abstractmethod
    def build_dataset_from_numeric(
        self,
        *x: Union[List[np.ndarray], List[pd.DataFrame], np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        batch_size=32,
        buffer_size=128,
        shuffle=False,
        repeat_count=1,
        stage="train",
        random_seed=1234,
    ):
        pass

    @abstractmethod
    def build_dataset_from_builder(
        self,
        *x: Union[List[np.ndarray], List[pd.DataFrame], str],
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        batch_size=-1,
        shuffle=False,
        buffer_size=256,
        random_seed=1234,
        stage="train",
        dataset_builder: Callable = None,
    ):
        pass

    @abstractmethod
    def build_dataset_from_csv(
        self,
        file_path: str,
        label: str = None,
        s_w: Optional[np.ndarray] = None,
        batch_size=-1,
        shuffle=False,
        repeat_count=1,
        random_seed=1234,
        buffer_size=None,
        na_value='?',
        label_decoder=None,
        stage="train",
    ):
        pass

    @abstractmethod
    def set_steps_per_epoch(self, steps_per_epoch):
        pass

    @abstractmethod
    def get_basenet_output_num(self):
        pass

    @abstractmethod
    def get_stop_training(self):
        pass

    @abstractmethod
    def set_sample_weight(self, sample_weight, stage="train"):
        pass

    @abstractmethod
    def reset_metrics(self):
        pass

    @abstractmethod
    def evaluate(
        self,
        forward_data: Union[List[ForwardData], ForwardData],
    ):
        pass

    @abstractmethod
    def predict(
        self,
        forward_data: Union[List[ForwardData], ForwardData],
    ):
        pass

    @abstractmethod
    def save_base_model(self, base_model_path: str, **kwargs):
        pass

    @abstractmethod
    def save_fuse_model(self, fuse_model_path: str, **kwargs):
        pass

    @abstractmethod
    def load_base_model(self, base_model_path: str, **kwargs):
        pass

    @abstractmethod
    def load_fuse_model(self, fuse_model_path: str, **kwargs):
        pass

    @abstractmethod
    def export_base_model(self, model_path: str, save_format: str = "onnx", **kwargs):
        pass

    @abstractmethod
    def export_fuse_model(self, model_path: str, save_format: str = "onnx", **kwargs):
        pass

    @abstractmethod
    def get_privacy_spent(self, step: int, orders=None):
        pass

    @abstractmethod
    def get_skip_gradient(self):
        pass
