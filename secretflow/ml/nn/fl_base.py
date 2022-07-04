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
import copy
import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.keras import callbacks as callbacks_module
from torch import nn
from torch.nn.modules.loss import _Loss as BaseTorchLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import secretflow.device as ft
from secretflow.utils.io import rows_count
from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.metrics import AUC, Mean, Precision, Recall
from secretflow.data.horizontal import PoissonDataSampler

# 抽象model类


class BaseModel(ABC):
    def __init__(self, builder_base: Callable, builder_fuse: Callable = None):
        self.model_base = builder_base() if builder_base is not None else None
        self.model_fuse = builder_fuse() if builder_fuse is not None else None

    @abstractmethod
    def build_dataset(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size=32,
        buffer_size=128,
        repeat_count=1,
    ):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def evaluate(
        self, x, y, batch_size=None, verbose=1, sample_weight=None, steps=None
    ):
        pass


class BaseTFModel(BaseModel):
    def __init__(self, builder_base: Callable[[], tf.keras.Model]):
        super().__init__(builder_base)
        self.model = builder_base() if builder_base else None
        self.train_set = None
        self.valid_set = None
        self.has_s_w = False
        self.callbacks = None
        self.logs = None
        self.epoch_logs = None
        self.training_logs = None

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
        # construct train_set
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
        elif stage == 'valid':
            self.valid_set = data_set
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
        sampler="batch",
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
            sampler: method of sampler
        """
        data_set = None
        self.has_s_w = False
        # construct train_set
        if x is None or len(x.shape) == 0:
            raise Exception("Data 'x' cannot be None")
        if y is None or len(y.shape) == 0:
            raise Exception("Data 'y' cannot be None")
        assert sampling_rate is not None, "Sampling rate cannot be None"
        assert (
            x.shape[0] == y.shape[0]
        ), "The samples of feature is different with label"

        if sampler == "batch":
            data_set = self.batch_sampler(
                x,
                y,
                s_w,
                sampling_rate,
                buffer_size,
                shuffle,
                random_seed,
                repeat_count,
            )
        elif sampler == "possion":
            data_set = self.possion_sampler(x, y, s_w, sampling_rate)
        else:
            logging.error(f'Unvalid sampler {sampler} during building local dataset')

        self.train_set = iter(data_set)

    def get_rows_count(self, filename):
        return int(rows_count(filename=filename)) - 1  # except header line

    def batch_sampler(
        self, x, y, s_w, sampling_rate, buffer_size, shuffle, random_seed, repeat_count
    ):
        batch_size = math.floor(x.shape[0] * sampling_rate)
        assert batch_size > 0, "Unvalid batch size"
        if s_w is not None:
            self.has_s_w = True
            data_set = (
                tf.data.Dataset.from_tensor_slices(
                    (x, y.astype(np.float64), s_w.astype(np.float64))
                )
                .batch(batch_size)
                .repeat(repeat_count)
            )
        else:
            data_set = (
                tf.data.Dataset.from_tensor_slices((x, y.astype(np.float64)))
                .batch(batch_size)
                .repeat(repeat_count)
            )

        if shuffle:
            if buffer_size is None:
                buffer_size = batch_size * 8
            data_set = data_set.shuffle(buffer_size, seed=random_seed)
        return data_set

    def possion_sampler(self, x, y, s_w, sampling_rate):
        gen = PoissonDataSampler(x, y, s_w, sampling_rate)
        x_shape = list(x.shape)
        x_shape[0] = None
        y_shape = list(y.shape)
        y_shape[0] = None
        if s_w is not None:
            self.has_s_w = True
            s_w_shape = list(s_w.shape)
            s_w_shape.shape[0] = None
            data_set = tf.data.Dataset.from_generator(
                lambda: gen,
                output_signature=(
                    tf.TensorSpec(shape=x_shape, dtype=x.dtype),
                    tf.TensorSpec(shape=y_shape, dtype=y.dtype),
                    tf.TensorSpec(shape=s_w_shape, dtype=s_w.dtype),
                ),
            )
        else:
            data_set = tf.data.Dataset.from_generator(
                lambda: gen,
                output_signature=(
                    tf.TensorSpec(shape=x_shape, dtype=x.dtype),
                    tf.TensorSpec(shape=y_shape, dtype=y.dtype),
                ),
            )
        data_set = data_set.repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return data_set

    # TODO compute_gradients():
    def get_weights(self):
        return self.model.get_weights()

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

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose=0,
        sample_weight=None,
        steps=None,
    ):
        assert (
            self.model is not None
        ), "Please do training first or provide a trained model"
        if self.valid_set is None:
            assert type(x) == type(y), "X and y must have same type!"
            local_metric = self.model.evaluate(
                x,
                y,
                batch_size=batch_size,
                verbose=verbose,
                sample_weight=sample_weight,
                steps=steps,
            )
        else:
            assert (
                x is None and y is None
            ), "Something is Wrong, x and y must be None,when use csv reader"
            local_metric = self.model.evaluate(
                self.valid_set, verbose=verbose, steps=steps
            )
        self.valid_set = None
        return np.array(local_metric)

    def predict(
        self,
        x,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        assert (
            self.model is not None
        ), "Please do training first or provide a trained model"

        local_pred = self.model.predict(
            x,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        return local_pred

    def init_training(self, callbacks, epochs=1, steps=0, verbose=0):
        assert self.model is not None, "model cannot be none, please give model define"
        if not isinstance(callbacks, callbacks_module.CallbackList):
            self.callbacks = callbacks_module.CallbackList(
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

    def train_step(self, weights, cur_steps, train_steps) -> Tuple[np.ndarray, int]:
        """Accept ps model params,then do local train

        Args:
            weights: global weight from params server
            cur_steps: current train step
            train_steps: local training steps

        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        if weights is not None:
            self.model.set_weights(weights)
        num_sample = 0
        self.callbacks.on_train_batch_begin(cur_steps)
        logs = {}
        for _ in range(train_steps):
            if self.has_s_w:
                x, y, s_w = next(self.train_set)
            else:
                x, y = next(self.train_set)
                s_w = None
            if isinstance(x, collections.OrderedDict):
                num_sample += int(x[list(x.keys())[0]].shape[0])
            else:
                num_sample += x.shape[0]

            with tf.GradientTape() as tape:
                # Step 1: forward pass
                y_pred = self.model(x, training=True)
                # Step 2: loss calculation, the loss function is configured in `compile()`.
                loss = self.model.compiled_loss(
                    y,
                    y_pred,
                    regularization_losses=self.model.losses,
                    sample_weight=s_w,
                )
            # Step 3: back propagation
            trainable_vars = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Step4: update metrics
            self.model.compiled_metrics.update_state(y, y_pred)
        for m in self.model.metrics:
            logs[m.name] = m.result().numpy()
        self.callbacks.on_train_batch_end(cur_steps + train_steps, logs)
        self.logs = logs
        self.epoch_logs = copy.deepcopy(self.logs)

        return self.model.get_weights(), num_sample

    def save_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        self.model.save(model_path)

    def load_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        self.model = tf.keras.models.load_model(model_path)


class BaseModule(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self, parameters=None):
        if parameters is None:
            parameters = self.parameters()
        grads = []
        for p in parameters:
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(
        self,
        gradients: List[Union[torch.Tensor, np.ndarray]],
        parameters: Optional[List[torch.Tensor]] = None,
    ):
        if parameters is None:
            parameters = self.parameters()
        for g, p in zip(gradients, parameters):
            if g is not None:
                p.grad = torch.from_numpy(g.copy())


class ModelPartition(object):
    def __init__(self, model_fn, optim_fn, loss_fn, dataloader_fn):
        self.model: BaseModule = model_fn()
        self.optimizer: Optimizer = optim_fn(self.model.parameters())
        self.loss: BaseTorchLoss = loss_fn()
        self._dataloader: Dict[str, DataLoader] = {
            k: dl_fn() for k, dl_fn in dataloader_fn.items()
        }
        self._dataiter: Dict[str, Iterator] = {
            k: iter(_dl) for k, _dl in self._dataloader.items()
        }

    def get_one_batch(self, name='train'):
        try:
            x, y = next(self._dataiter[name])
        except StopIteration:
            self._dataiter = iter(self._dataloader[name])
            x, y = next(self._dataiter)
        return x, y

    def forward(
        self, used_name='train', external_input=None
    ) -> (torch.Tensor, torch.Tensor):
        if external_input is None:
            external_input = {}
        x, y = self.get_one_batch(used_name)
        y_pred = self.model(x, **external_input)
        return y_pred, y

    def zero_grad(self):
        self.optimizer.zero_grad()

    def backward(self, used_name='train', gradients=None, external_input: Dict = None):
        if gradients is not None:
            self.model.set_gradients(gradients)
        else:
            if external_input is None:
                external_input = {}
            y_pred, y = self.forward(used_name, external_input)
            loss = self.loss(y_pred, y)
            loss.backward()
        return self.model.get_gradients()

    def apply_gradients(self, gradients=None):
        if gradients is not None:
            self.model.set_gradients(gradients)
        self.optim_step()

    def optim_step(self):
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def call_model_fn(self, fn_name, *args, **kwargs):
        # TODO: a temporary utils
        return getattr(self.model, fn_name)(*args, **kwargs)


@proxy(PYUObject)
class PYUTFModel(BaseTFModel):
    pass


@proxy(ft.PYUObject)
class PYUModel(ModelPartition):
    pass
