#!/usr/bin/env python3
# *_* coding: utf-8 *_*

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


"""CustomSLModel

"""

import logging
import math
import os
from typing import Callable, Dict, Iterable, List, Tuple, Union

from multiprocess import cpu_count
from tqdm import tqdm

from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, Device, reveal, wait
from secretflow.device.device.pyu import PYUObject
from secretflow.ml.nn.sl.strategy_dispatcher import dispatch_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.compressor import Compressor
from secretflow.utils.random import global_random

# from secretflow.ml.nn import SLModel
from .sl_model import SLModel
import pdb


class CustomSLModel(SLModel):
    def __init__(
        self,
        base_model_dict: Dict[Device, Callable[[], "tensorflow.keras.Model"]] = {},
        device_y: PYU = None,
        model_fuse: Callable[[], "tensorflow.keras.Model"] = None,
        compressor: Compressor = None,
        dp_strategy_dict: Dict[Device, DPStrategy] = None,
        random_seed: int = None,
        device_strategy_dict: Dict[Device, str] = {},  # modify: custom device strategy
        **kwargs,
    ):
        """Custom Interface for vertical split learning, support different strategies
        Attributes:
            base_model_dict: Basemodel dictionary, key is PYU, value is the Basemodel defined by party.
            device_y: Define which model have label.
            model_fuse:  Fuse model defination.
            compressor: Define strategy tensor compression algorithms to speed up transmission.
            dp_strategy_dict: Dp strategy dictionary.
            random_seed: If specified, the initial value of the model will remain the same, which ensures reproducible.
            device_strategy_dict: strategy dictionary of split learning.
        """
        super().__init__()

        self.device_y = device_y
        self.has_compressor = compressor is not None
        self.dp_strategy_dict = dp_strategy_dict
        self.simulation = kwargs.get("simulation", False)
        self.num_parties = len(base_model_dict)

        # modify: custom split learning dictionary
        self.device_strategy_dict = device_strategy_dict
        attack_args = kwargs.get("attack_args", {})
        defense_args = kwargs.get("defense_args", {})

        # TODO: add argument `backend`
        import secretflow.ml.nn.sl.backend.tensorflow.strategy  # noqa

        self._workers = {}
        for device, model in base_model_dict.items():
            self._workers[device], self.check_skip_grad = dispatch_strategy(
                # strategy,
                device_strategy_dict.get(device, "split_nn"),
                backend=kwargs.get("backend", "tensorflow"),
                device=device,
                builder_base=model,
                builder_fuse=None if device != device_y else model_fuse,
                compressor=compressor,
                random_seed=random_seed,
                dp_strategy=dp_strategy_dict.get(device, None)
                if dp_strategy_dict
                else None,
                base_local_steps=kwargs.get("base_local_steps", 1),
                fuse_local_steps=kwargs.get("fuse_local_steps", 1),
                bound_param=kwargs.get("bound_param", 0.0),
                loss_thres=kwargs.get("loss_thres", 0.01),
                split_steps=kwargs.get("split_steps", 1),
                max_fuse_local_steps=kwargs.get("max_fuse_local_steps", 1),
                attack_args=attack_args.get(device, {}),
                defense_args=defense_args.get(device, {}),
            )
