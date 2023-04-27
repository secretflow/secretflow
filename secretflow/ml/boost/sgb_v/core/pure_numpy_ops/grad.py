# Copyright 2023 Ant Group Co., Ltd.
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
import numpy as np
from .pred import sigmoid
from typing import Tuple


def compute_gh_linear(y: np.ndarray, pred: np.ndarray):
    return pred - y, np.ones(pred.shape)


def compute_gh_logistic(y: np.ndarray, pred: np.ndarray):
    yhat = sigmoid(pred)
    return yhat - y, yhat * (1 - yhat)


def split_GH(x) -> Tuple[np.ndarray, np.ndarray]:
    return x[:, 0].reshape(1, -1), x[:, 1].reshape(1, -1)
