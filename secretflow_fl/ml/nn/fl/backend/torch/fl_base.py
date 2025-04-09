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
"""Torch model on worker side
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchmetrics

from secretflow.utils.io import rows_count
from secretflow_fl.ml.nn.core.torch import BuilderType, module
from secretflow_fl.ml.nn.fl.backend.torch.sampler import sampler_data
from secretflow_fl.ml.nn.metrics import Default, Mean, Precision, Recall


class BaseTorchModel(ABC):
    def __init__(
        self,
        builder_base: BuilderType,
        random_seed: int = None,
        skip_bn: bool = False,
        **kwargs,
    ):
        self.train_data_loader = None
        self.eval_data_loader = None
        self.callbacks = None
        self.logs = None
        self.training_logs = {}
        self.epoch = []
        self.epoch_logs = None
        self.wrapped_metrics = []  # store wrapped metrics for global aggregation
        self.history = {}
        self.train_set = None
        self.eval_set = None
        self.skip_bn = skip_bn
        if random_seed is not None:
            torch.manual_seed(random_seed)
        assert builder_base is not None, "Builder_base cannot be none"
        self.use_gpu = kwargs.get("use_gpu", False)
        self.exe_device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.model = module.build(builder_base, self.exe_device)

    def build_dataset_from_csv(
        self,
        csv_file_path: str,
        label: str,
        sampling_rate=None,
        shuffle=False,
        random_seed=1234,
        na_value="?",
        repeat_count=1,
        sample_length=0,
        buffer_size=None,
        ignore_errors=True,
        prefetch_buffer_size=None,
        stage="train",
        label_decoder=None,
    ):
        """build torch.dataloader

        Args:
            csv_file_path: Dict of csv file path
            label: label column name
            sampling_rate: Sampling rate of a batch
            shuffle: A bool that indicates whether the input should be shuffled
            random_seed: Randomization seed to use for shuffling.
            na_value: Additional string to recognize as NA/NaN.
            repeat_count: num of repeats
            sample_length: num of sample length
            buffer_size: shuffle size
            ignore_errors: if `True`, ignores errors with CSV file parsing,
            prefetch_buffer_size: An int specifying the number of feature batches to prefetch for performance improvement.
            stage: the stage of the datset
            label_decoder: callable function for label preprocess
        """
        raise Exception("CSV incremental loader is not supported yet")

    def build_dataset(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        sampling_rate=None,
        buffer_size=None,
        shuffle=False,
        random_seed=1234,
        repeat_count=1,
        sampler_method="batch",
        stage="train",
    ):
        """build torch.dataloader

        Args:
            x: feature, FedNdArray or HDataFrame
            y: label, FedNdArray or HDataFrame
            s_w: sample weight of this dataset
            sampling_rate: Sampling rate of a batch
            buffer_size: shuffle size
            shuffle: A bool that indicates whether the input should be shuffled
            random_seed: Prg seed for shuffling
            repeat_count: num of repeats
            sampler: method of sampler
        """
        if x is None or len(x.shape) == 0:
            raise Exception("Data 'x' cannot be None")

        if y is not None:
            assert (
                x.shape[0] == y.shape[0]
            ), "The samples of feature is different with label"

        assert sampling_rate is not None, "Sampling rate cannot be None"
        data_set = sampler_data(
            sampler_method,
            x,
            y,
            s_w,
            sampling_rate,
            buffer_size,
            shuffle,
            repeat_count,
            random_seed,
        )
        if stage == "train":
            self.train_set = data_set
        elif stage == "eval":
            self.eval_set = data_set
        else:
            raise Exception(f"Illegal argument stage={stage}")

    def build_dataset_from_builder(
        self,
        dataset_builder: Callable,
        x: Union[pd.DataFrame, str],
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        repeat_count=1,
        stage="train",
    ):
        """build tf.data.Dataset

        Args:
            dataset_builder: Function of how to build dataset, must return dataset and step_per_epoch
            x: A pandas Dataframe or A string representing the path to a CSV file or data folder containing the input data.
            y: label, An optional NumPy array containing the labels for the dataset. Defaults to None.
            s_w: An optional NumPy array containing the sample weights for the dataset. Defaults to None.
            repeat_count: An integer specifying the number of times to repeat the dataset. This is useful for increasing the effective size of the dataset.
            stage: A string indicating the stage of the dataset (either "train", "eval"). Defaults to "train".

        Returns:
            A tensorflow dataset
        """
        data_set = None
        assert dataset_builder is not None, "Dataset builder cannot be none"
        if isinstance(x, str):
            data_set, step_per_epoch = dataset_builder(x, stage=stage)
        else:
            if y is not None:
                x.append(y)
                if s_w is not None and len(s_w.shape) > 0:
                    x.append(s_w)

            data_set, step_per_epoch = dataset_builder(x, stage=stage)
        if stage == "train":
            self.train_set = data_set
        elif stage == "eval":
            self.eval_set = data_set
        else:
            raise Exception(f"Illegal argument stage={stage}")
        return step_per_epoch

    def next_batch(self, stage="train"):
        if stage == "train":
            iter_data = next(self.train_iter)
        elif stage == "eval":
            iter_data = next(self.eval_iter)
        else:
            raise Exception(f"Illegal argument stage={stage}")

        if len(iter_data) == 2:
            x, y = iter_data
            s_w = None
        elif len(iter_data) == 3:
            x, y, s_w = iter_data
        else:
            raise ValueError(
                f"Len of batch data must be 2 or 3 but got {len(iter_data)}"
            )

        if self.use_gpu:
            x = x.to(self.exe_device)
            y = y.to(self.exe_device)
            if s_w is not None:
                s_w = s_w.to(self.exe_device)

        return x, y, s_w

    def get_rows_count(self, filename):
        return int(rows_count(filename=filename)) - 1  # except header line

    def get_weights(self, return_numpy=True):
        if self.skip_bn:
            return self.model.get_weights_not_bn(return_numpy=return_numpy)
        else:
            return self.model.get_weights(return_numpy=return_numpy)

    def set_weights(self, weights):
        """set weights of client model"""
        if self.skip_bn:
            self.model.update_weights_not_bn(weights)
        else:
            self.model.update_weights(weights)

    def set_validation_metrics(self, global_metrics):
        self.epoch_logs.update(global_metrics)

    def reset_metrics(self):
        for m in self.model.metrics:
            m.reset()

    def wrap_local_metrics(self, stage="train"):
        # TODO: use pytorch to rewrite
        wraped_metrics = []
        for m in self.model.metrics:
            if isinstance(m, (torchmetrics.Accuracy)):
                tp, fp, tn, fn = map(lambda x: x.cpu(), m._get_final_stats())
                name = m._get_name().lower()
                name = (f"val_{name}" if stage == "val" else name,)

                correct = float((tp + tn).numpy().sum())
                total = float((tp + tn + fp + fn).numpy().sum())

                wraped_metrics.append(Mean(name, correct, total))

            elif isinstance(m, torchmetrics.Precision):
                tp, fp, tn, fn = map(lambda x: x.cpu(), m._get_final_stats())
                threshold = m.threshold

                wraped_metrics.append(
                    Precision(
                        (
                            f"val_{m._get_name().lower()}"
                            if stage == "val"
                            else m._get_name().lower()
                        ),
                        [threshold],
                        [float(tp.numpy().sum())],
                        [float(fp.numpy().sum())],
                    )
                )
            elif isinstance(m, torchmetrics.Recall):
                tp, fp, tn, fn = map(lambda x: x.cpu(), m._get_final_stats())
                threshold = m.threshold
                wraped_metrics.append(
                    Recall(
                        (
                            f"val_{m._get_name().lower()}"
                            if stage == "val"
                            else m._get_name().lower()
                        ),
                        [threshold],
                        tp.numpy().sum(),
                        fn.numpy().sum(),
                    )
                )
            else:
                # only do naive aggregate
                metrics_value = m.cpu().compute()
                wraped_metrics.append(
                    Default(
                        (
                            f"val_{m._get_name().lower()}"
                            if stage == "val"
                            else m._get_name().lower()
                        ),
                        total=metrics_value,
                        count=1,
                    )
                )
        return wraped_metrics

    def evaluate(self, evaluate_steps=0):
        assert evaluate_steps > 0, "Evaluate_steps must greater than 0"
        assert self.model is not None, "Model cannot be none, please give model define"
        assert (
            len(self.model.metrics) > 0
        ), "Metric cannot be none, please give metric by 'TorchModel'"
        self.model.eval()

        # reset all metrics
        self.eval_iter = iter(self.eval_set)
        self.reset_metrics()
        with torch.no_grad():
            for step in range(evaluate_steps):
                x, y, s_w = self.next_batch(stage="eval")
                self.model.validation_step((x, y), step, sample_weight=s_w)
            result = {}
            self.transform_metrics(result, stage="eval")
        if self.logs is None:
            self.wrapped_metrics.extend(self.wrap_local_metrics())
            return self.wrap_local_metrics()
        else:
            val_result = {}
            for k, v in result.items():
                val_result[f"val_{k}"] = v
            self.logs.update(val_result)
            self.wrapped_metrics.extend(self.wrap_local_metrics(stage="val"))
            return self.wrap_local_metrics(stage="val")

    def predict(
        self,
        predict_steps=0,
    ):
        assert (
            self.model is not None
        ), "Please do training first or provide a trained model"
        pred_result = []
        self.eval_iter = iter(self.eval_set)
        assert self.eval_iter is not None, "self.eval_set must be initialized"
        for _ in range(predict_steps):
            iter_data = next(self.eval_iter)
            x = iter_data[0]
            if self.use_gpu:
                x = x.to(self.exe_device)
            y_pred = self.model(x)
            pred_result.extend(y_pred)
        return pred_result

    def _reset_data_iter(self):
        self.reset_metrics()
        self.train_iter = iter(self.train_set)

    def get_local_metrics(self):
        return self.wrapped_metrics

    def get_logs(self):
        print(self.logs)
        return self.logs

    def init_training(self, callbacks, epochs=1, steps=0, verbose=0):
        assert self.model is not None, "model cannot be none, please give model define"
        if callbacks is not None:
            raise Exception("Callback is not supported yet")

    def on_train_begin(self):
        self.training_logs = {}
        self.epoch = []

    def on_epoch_begin(self, epoch):
        self._current_epoch = epoch
        self.epoch_logs = {}
        self.reset_metrics()
        if self.train_set is not None:
            self.train_iter = iter(self.train_set)
        if self.eval_set is not None:
            self.eval_iter = iter(self.eval_set)

    def on_epoch_end(self, epoch):
        self.epoch.append(epoch)
        for k, v in self.epoch_logs.items():
            self.history.setdefault(k, []).append(v)
        self.training_logs = self.epoch_logs

        return self.epoch_logs

    def transform_metrics(self, logs, stage="train"):
        for m in self.model.metrics:
            result = m.compute()
            logs[f"{stage}_{m._get_name().lower()}"] = result
        return logs

    def on_train_end(self):
        return self.history

    def get_stop_training(self):
        return False  # is not supported

    @abstractmethod
    def train_step(self, weights, cur_steps, train_steps, **kwargs):
        pass

    @abstractmethod
    def apply_weights(self, weights, **kwargs):
        pass

    def save_model(self, model_path: str):
        """For compatibility reasons it is recommended to instead save only its state dict Ref:https://pytorch.org/docs/master/notes/serialization.html#id5"""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        assert model_path is not None, "model path cannot be empty"
        check_point = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.optimizers_state_dict(),
            "epoch": self.epoch[-1] if self.epoch else 0,
        }
        torch.save(check_point, model_path)

    def load_model(self, model_path: str, **kwargs):
        """load model from state dict, model structure must be defined before load"""
        assert model_path is not None, "model path cannot be empty"
        assert self.model is not None, "model structure must be defined before load"
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.load_optimizers_state_dict(checkpoint["optimizer_state_dict"])
        if self.use_gpu:
            self.model.to(self.exe_device)
        return checkpoint["epoch"]
