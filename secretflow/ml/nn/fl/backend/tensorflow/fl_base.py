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


import collections
import math
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from secretflow.ml.nn.fl.backend.tensorflow.sampler import sampler_data
from secretflow.ml.nn.metrics import AUC, Mean, Precision, Recall
from secretflow.utils.io import rows_count


class BaseTFModel:
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        random_seed: int = None,
    ):
        self.train_set = None
        self.eval_set = None
        self.callbacks = None
        self.logs = None
        self.epoch_logs = None
        self.training_logs = None
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
        self.model = builder_base() if builder_base else None

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
        """build tf.data.Dataset

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
        assert sample_length > 0, "Sample_length cannot be zero!"
        data_set = None
        batch_size = math.floor(sample_length * sampling_rate)
        data_set = tf.data.experimental.make_csv_dataset(
            csv_file_path,
            batch_size=batch_size,
            label_name=label,
            na_value=na_value,
            header=True,
            num_epochs=1,
            ignore_errors=ignore_errors,
            prefetch_buffer_size=prefetch_buffer_size,
            shuffle=shuffle,
            shuffle_seed=random_seed,
        )
        data_set = data_set.repeat(repeat_count)
        if shuffle:
            if buffer_size is None:
                buffer_size = batch_size * 8
            data_set = data_set.shuffle(buffer_size, seed=random_seed)
        if label_decoder is not None:
            data_set = data_set.map(label_decoder)
        if stage == 'train':
            self.train_set = iter(data_set.repeat(repeat_count))
        elif stage == "eval":
            self.eval_set = iter(data_set.repeat(repeat_count))
        else:
            raise Exception("Unknow stage={stage}")

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
        """build tf.data.Dataset

        Args:
            x: feature, FedNdArray or HDataFrame
            y: label, FedNdArray or HDataFrame
            s_w: sample weight of this dataset
            sampling_rate: Sampling rate of a batch
            buffer_size: shuffle size
            shuffle: A bool that indicates whether the input should be shuffled
            random_seed: Prg seed for shuffling
            repeat_count: num of repeats
            sampler_method: method of sampler
        """
        data_set = None
        # construct train_set
        if x is None or len(x.shape) == 0:
            raise Exception("Data 'x' cannot be None")

        assert sampling_rate is not None, "Sampling rate cannot be None"
        if x is not None and y is not None:
            assert (
                x.shape[0] == y.shape[0]
            ), "The samples of feature is different with label"
        data_set = sampler_data(
            sampler_method=sampler_method,
            x=x,
            y=y,
            s_w=s_w,
            sampling_rate=sampling_rate,
            buffer_size=buffer_size,
            shuffle=shuffle,
            repeat_count=repeat_count,
            random_seed=random_seed,
        )
        if stage == "train":
            self.train_set = iter(data_set)
        elif stage == "eval":
            self.eval_set = iter(data_set)
        else:
            raise Exception(f"Illegal argument stage={stage}")

    def get_rows_count(self, filename):
        return int(rows_count(filename=filename)) - 1  # except header line

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        """set weights of client model"""
        self.model.set_weights(weights)

    def set_validation_metrics(self, global_metrics):
        self.epoch_logs.update(global_metrics)

    def wrap_local_metrics(self):
        wraped_metrics = []
        for m in self.model.metrics:
            if isinstance(m, tf.keras.metrics.Mean):
                wraped_metrics.append(Mean(m.name, m.total.numpy(), m.count.numpy()))
            elif isinstance(m, tf.keras.metrics.AUC):
                wraped_metrics.append(
                    AUC(
                        m.name,
                        m.thresholds,
                        m.true_positives.numpy(),
                        m.true_negatives.numpy(),
                        m.false_positives.numpy(),
                        m.false_negatives.numpy(),
                        m.curve,
                    )
                )
            elif isinstance(m, tf.keras.metrics.Precision):
                wraped_metrics.append(
                    Precision(
                        m.name,
                        m.thresholds,
                        m.true_positives.numpy(),
                        m.false_positives.numpy(),
                    )
                )
            elif isinstance(m, tf.keras.metrics.Recall):
                wraped_metrics.append(
                    Recall(
                        m.name,
                        m.thresholds,
                        m.true_positives.numpy(),
                        m.false_negatives.numpy(),
                    )
                )
            else:
                raise NotImplementedError(
                    f'Unsupported global metric {m.__class__.__qualname__} for now, please add it.'
                )
        return wraped_metrics

    def evaluate(self, evaluate_steps=0):
        assert evaluate_steps > 0, "evaluate_steps must greater than 0"
        assert self.model is not None, "model cannot be none, please give model define"

        self.model.compiled_metrics.reset_state()
        self.model.compiled_loss.reset_state()
        for _ in range(evaluate_steps):
            iter_data = next(self.eval_set)
            if len(iter_data) == 2:
                x, y = iter_data
                s_w = None
            elif len(iter_data) == 3:
                x, y, s_w = iter_data
            if isinstance(x, collections.OrderedDict):
                x = tf.stack(list(x.values()), axis=1)
            # Step 1: forward pass
            y_pred = self.model(x)
            # Step 2: update metrics
            self.model.compiled_metrics.update_state(y, y_pred)
            # Step 3: update loss
            self.model.compiled_loss(y, y_pred, sample_weight=s_w)
            result = {}
            for m in self.model.metrics:
                result[m.name] = m.result().numpy()
        return self.wrap_local_metrics()

    def predict(self, predict_steps=0):
        assert (
            self.model is not None
        ), "Please do training first or provide a trained model"
        pred_result = []
        assert self.eval_set is not None, "self.eval_set must be initialized"

        for _ in range(predict_steps):
            x = next(self.eval_set)
            if isinstance(x, collections.OrderedDict):
                x = tf.stack(list(x.values()), axis=1)
            y_pred = self.model(x)
            pred_result.extend(y_pred)
        return pred_result

    def init_training(self, callbacks, epochs=1, steps=0, verbose=0):
        assert self.model is not None, "model cannot be none, please give model define"

        from tensorflow.python.keras import callbacks as tf_callbacks

        if not isinstance(callbacks, tf_callbacks.CallbackList):
            self.callbacks = tf_callbacks.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self.model,
                verbose=verbose,
                epochs=epochs,
                steps=steps,
            )
        else:
            raise NotImplementedError

    def on_train_begin(self):
        self.callbacks.on_train_begin()

    def on_epoch_begin(self, epoch):
        self.callbacks.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        self.callbacks.on_epoch_end(epoch, self.epoch_logs)
        self.training_logs = self.epoch_logs
        return self.epoch_logs

    def on_train_end(self):
        self.callbacks.on_train_end(logs=self.training_logs)
        return self.model.history.history

    def get_stop_training(self):
        return self.model.stop_training

    @abstractmethod
    def train_step(self, weights, cur_steps, train_steps, **kwargs):
        pass

    def save_model(self, model_path: str):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        assert model_path is not None, "model path cannot be empty"
        self.model.save(model_path)

    def load_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        self.model = tf.keras.models.load_model(model_path)
        return self.model.get_weights()[0].sum()
