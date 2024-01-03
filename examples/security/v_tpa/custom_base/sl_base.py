#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# # Copyright 2022 Ant Group Co., Ltd.
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


"""sl model base
"""
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import callbacks as callbacks_module

from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.metrics import AUC, Mean, Precision, Recall
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.communicate import ForwardData


class SLBaseModel(ABC):
    def __init__(self, builder_base: Callable, builder_fuse: Callable = None):
        self.model_base = builder_base() if builder_base is not None else None
        self.model_fuse = builder_fuse() if builder_fuse is not None else None

    @abstractmethod
    def base_forward(self):
        pass

    @abstractmethod
    def base_backward(self):
        pass

    @abstractmethod
    def fuse_net(self, hiddens):
        pass


class SLBaseTFModel(SLBaseModel):
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        builder_fuse: Callable[[], tf.keras.Model],
        dp_strategy: DPStrategy,
        random_seed: int = None,
        **kwargs,
    ):
        self.dp_strategy = dp_strategy
        self.embedding_dp = (
            self.dp_strategy.embedding_dp if dp_strategy is not None else None
        )
        self.label_dp = self.dp_strategy.label_dp if dp_strategy is not None else None

        self.train_set = None
        self.eval_set = None
        self.valid_set = None
        self.tape = None
        self.h = None
        self.train_x, self.train_y = None, None
        self.eval_x, self.eval_y = None, None
        self.kwargs = {}
        self.train_has_x = False
        self.train_has_y = False
        self.train_has_s_w = False
        self.eval_has_x = False
        self.eval_has_y = False
        self.eval_has_s_w = False
        self.train_sample_weight = None
        self.eval_sample_weight = None
        self.fuse_callbacks = None
        self.predict_callbacks = None
        self.logs = None
        self.epoch_logs = None
        self.training_logs = None
        self.steps_per_epoch = None
        self.skip_gradient = False
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
        super().__init__(builder_base, builder_fuse)

    @staticmethod
    @tf.custom_gradient
    def fuse_op(x, y):
        def grad(upstream):
            return upstream * y, upstream * y

        return x, grad

    def init_data(self):
        self.train_x, self.train_y = None, None
        self.eval_x, self.eval_y = None, None
        self.train_sample_weight = None
        self.eval_sample_weight = None

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch

    def get_basenet_output_num(self):
        if self.model_base:
            if (
                hasattr(self.model_base, "outputs")
                and self.model_base.outputs is not None
            ):
                return len(self.model_base.outputs)
            else:
                if hasattr(self.model_base, "output_num"):
                    return self.model_base.output_num()
                else:
                    raise Exception(
                        "Please define the output_num function in basemodel and return the number of basenet outputs, then try again"
                    )
        else:
            return 0

    def build_dataset_from_numeric(
        self,
        *x: List[np.ndarray],
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        batch_size=32,
        buffer_size=128,
        shuffle=False,
        repeat_count=1,
        stage="train",
        random_seed=1234,
    ):
        """build tf.data.Dataset

        Args:
            x: feature, FedNdArray or HDataFrame
            y: label, FedNdArray or HDataFrame
            s_w: sample weight, FedNdArray or HDataFrame
            batch_size: Number of samples per gradient update
            buffer_size: buffer size for shuffling
            shuffle: whether shuffle the dataset or not
            repeat_count: num of repeats
            stage: stage of this datset
            random_seed: Prg seed for shuffling
        """
        assert (
            x is not None or y is not None
        ), f"At least one of feature(x) and label(y) is not None."
        data_tuple = []
        has_x = False
        if x is not None:
            x = [xi for xi in x if xi is not None]
            if x:
                has_x = True
            data_tuple.extend(x)
        has_y = False
        has_s_w = False
        if y is not None and len(y.shape) > 0:
            has_y = True
            data_tuple.append(y)
            if s_w is not None and len(s_w.shape) > 0:
                has_s_w = True
                data_tuple.append(s_w)

        # convert pandas.DataFrame to numpy.ndarray

        data_tuple = [
            t.values if isinstance(t, pd.DataFrame) else t for t in data_tuple
        ]

        # https://github.com/tensorflow/tensorflow/issues/20481
        data_tuple = data_tuple[0] if len(data_tuple) == 1 else tuple(data_tuple)
        if len(data_tuple) > 0:
            data_set = tf.data.Dataset.from_tensor_slices(data_tuple)

            if shuffle:
                data_set = data_set.shuffle(buffer_size, seed=random_seed)

            data_set = data_set.batch(batch_size).repeat(repeat_count)

            self.set_dataset_stage(
                data_set=data_set,
                stage=stage,
                has_x=has_x,
                has_y=has_y,
                has_s_w=has_s_w,
            )
        else:
            data_set = None

    def build_dataset_from_builder(
        self,
        *x: List[np.ndarray],
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        batch_size=-1,
        shuffle=False,
        buffer_size=256,
        random_seed=1234,
        stage="train",
        dataset_builder: Callable = None,
    ):
        """build tf.data.Dataset

        Args:
            x: feature, FedNdArray or HDataFrame
            y: label, FedNdArray or HDataFrame
            s_w: sample weight, FedNdArray or HDataFrame
            batch_size: Number of samples per gradient update.
            shuffle: Whether to shuffle dataset
            buffer_size: buffer size for shuffling
            random_seed: Prg seed for shuffling
            stage: stage of this datset
            dataset_builder: dataset build callable function of worker
        """
        assert (
            x is not None or y is not None
        ), f"At least one of feature(x) and label(y) is not None."
        if not dataset_builder:
            return -1
        data_tuple = []
        has_x = False

        #  x is (None,) if dont have feature
        if x is not None and x[0] is not None:
            has_x = True
            x = [xi for xi in x]
            data_tuple.extend(x)

        has_y = False
        has_s_w = False
        if y is not None and len(y.shape) > 0:
            has_y = True
            data_tuple.append(y)
            if s_w is not None and len(s_w.shape) > 0:
                has_s_w = True
                data_tuple.append(s_w)

        data_set = dataset_builder(data_tuple)
        # Compatible with existing gnn databuilder
        if hasattr(data_set, "steps_per_epoch"):
            return data_set.steps_per_epoch

        if shuffle:
            data_set = data_set.shuffle(buffer_size, seed=random_seed)
        # Infer batch size
        batch_data = next(iter(data_set))
        if isinstance(batch_data, Tuple):
            batch_data = batch_data[0]
        if isinstance(batch_data, Dict):
            batch_data = list(batch_data.values())[0]

        if isinstance(batch_data, tf.Tensor):
            batch_size_inf = batch_data.shape[0]
            if batch_size > 0:
                assert (
                    batch_size_inf == batch_size
                ), f"The batchsize from 'fit' is {batch_size}, but the batchsize derived from datasetbuilder is {batch_size_inf}, please check"
            else:
                batch_size = batch_size_inf
        else:
            raise Exception(
                f"Unable to get batchsize from dataset, please spcify batchsize in 'fit'"
            )

        self.set_dataset_stage(
            data_set=data_set,
            stage=stage,
            has_x=has_x,
            has_y=has_y,
            has_s_w=has_s_w,
        )
        if isinstance(data_set, tf.data.Dataset):
            import math

            total_size = len(x[0]) if x is not None else 0
            return math.ceil(total_size / batch_size)  # use ceil to avoid drop_last
        else:
            raise Exception("Unknown databuilder")

    def set_dataset_stage(
        self,
        data_set,
        stage="train",
        has_x=None,
        has_y=None,
        has_s_w=None,
    ):
        data_set = iter(data_set)
        if stage == "train":
            self.train_set = data_set
            self.train_has_x = has_x
            self.train_has_y = has_y
            self.train_has_s_w = has_s_w
        elif stage == "eval":
            self.eval_set = data_set
            self.eval_has_x = has_x
            self.eval_has_y = has_y
            self.eval_has_s_w = has_s_w
        else:
            raise Exception(f"Illegal argument stage={stage}")

    @tf.function
    def _base_forward_internal(self, data_x, training=True):
        h = self.model_base(data_x, training=training)

        # Embedding differential privacy
        if self.embedding_dp is not None:
            if isinstance(h, List):
                h = [self.embedding_dp(hi) for hi in h]
            else:
                h = self.embedding_dp(h)
        return h

    def unpack_dataset(self, data, has_x, has_y, has_s_w):
        data_x, data_y, data_s_w = None, None, None
        # case: only has x or has y, and s_w is none
        if has_x and not has_y and not has_s_w:
            data_x = data
        elif not has_x and not has_s_w and has_y:
            data_y = data
        elif not has_x and not has_y:
            raise Exception("x and y can not be none at same time")
        elif not has_y and has_s_w:
            raise Exception("Illegal argument: s_w is illegal if y is none")
        else:
            # handle data isinstance of list
            if has_y:
                if has_s_w:
                    data_x = data[:-2] if has_x else None
                    data_y = data[-2]
                    data_s_w = data[-1]
                else:
                    data_x = data[:-1] if has_x else None
                    data_y = data[-1]
                if self.label_dp is not None:
                    data_y = self.label_dp(data_y.numpy())
                    data_y = tf.convert_to_tensor(data_y)

        return data_x, data_y, data_s_w

    def base_forward(self, stage="train") -> ForwardData:
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
        Returns: hidden embedding
        """
        data_x = None
        self.init_data()
        training = True
        if stage == "train":
            train_data = next(self.train_set)

            self.train_x, self.train_y, self.train_sample_weight = self.unpack_dataset(
                train_data, self.train_has_x, self.train_has_y, self.train_has_s_w
            )
            data_x = self.train_x
        elif stage == "eval":
            training = False
            eval_data = next(self.eval_set)
            self.eval_x, self.eval_y, self.eval_sample_weight = self.unpack_dataset(
                eval_data, self.eval_has_x, self.eval_has_y, self.eval_has_s_w
            )
            data_x = self.eval_x
        else:
            raise Exception("invalid stage")

        # model_base is none equal to x is none
        if not self.model_base:
            return None

        # Strip tuple of length one, e.g: (x,) -> x
        data_x = data_x[0] if isinstance(data_x, Tuple) and len(data_x) == 1 else data_x

        self.tape = tf.GradientTape(persistent=True)
        with self.tape:
            self.h = self._base_forward_internal(data_x, training=training)
        self.data_x = data_x

        forward_data = ForwardData()
        if len(self.model_base.losses) > 0:
            forward_data.losses = tf.add_n(self.model_base.losses)
        # The compressor can only recognize np type but not tensor.
        forward_data.hidden = self.h.numpy() if tf.is_tensor(self.h) else self.h
        return forward_data

    def _base_backward_internal(self, gradients, trainable_vars):
        self.model_base.optimizer.apply_gradients(zip(gradients, trainable_vars))

    def base_backward(self, gradient):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
        """

        return_hiddens = []

        with self.tape:
            if len(gradient) == len(self.h):
                for i in range(len(gradient)):
                    return_hiddens.append(self.fuse_op(self.h[i], gradient[i]))
            else:
                gradient = gradient[0]
                return_hiddens.append(self.fuse_op(self.h, gradient))
            # add model.losses into graph
            return_hiddens.append(self.model_base.losses)

        trainable_vars = self.model_base.trainable_variables
        gradients = self.tape.gradient(return_hiddens, trainable_vars)

        self._base_backward_internal(gradients, trainable_vars)

        # clear intermediate results
        self.tape = None
        self.h = None
        self.kwargs = {}

    def get_base_losses(self):
        return self.model_base.losses

    def get_base_weights(self):
        return self.model_base.get_weights()

    def get_fuse_weights(self):
        return self.model_fuse.get_weights() if self.model_fuse is not None else None

    def init_training(self, callbacks, epochs=1, steps=0, verbose=0):
        if not isinstance(callbacks, callbacks_module.CallbackList):
            self.fuse_callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self.model_fuse,
                verbose=verbose,
                epochs=epochs,
                steps=steps,
                steps_per_epoch=self.steps_per_epoch,
            )
        else:
            raise NotImplementedError

    def init_predict(self, callbacks, steps=1, verbose=0):
        if not isinstance(callbacks, callbacks_module.CallbackList):
            self.predict_callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self.model_base,
                verbose=verbose,
                epochs=1,
                steps=steps,
            )
        else:
            raise NotImplementedError

    def get_stop_training(self):
        return self.model_fuse.stop_training

    def on_train_begin(self):
        if self.fuse_callbacks:
            self.fuse_callbacks.on_train_begin()

    def on_epoch_begin(self, epoch):
        if self.fuse_callbacks:
            self.fuse_callbacks.on_epoch_begin(epoch)

    def on_train_batch_begin(self, step=None):
        assert step is not None, "Step cannot be none"
        if self.fuse_callbacks:
            self.fuse_callbacks.on_train_batch_begin(step)

    def on_train_batch_end(self, step=None):
        assert step is not None, "Step cannot be none"
        self.epoch_logs = copy.deepcopy(self.logs)
        if self.fuse_callbacks:
            self.fuse_callbacks.on_train_batch_end(step, self.logs)

    def on_validation(self, val_logs):
        val_logs = {"val_" + name: val for name, val in val_logs.items()}
        self.epoch_logs.update(val_logs)

    def on_epoch_end(self, epoch):
        if self.fuse_callbacks:
            self.fuse_callbacks.on_epoch_end(epoch, self.epoch_logs)
        self.training_logs = self.epoch_logs
        return self.epoch_logs

    def on_train_end(self):
        if self.fuse_callbacks:
            self.fuse_callbacks.on_train_end(logs=self.training_logs)
        if self.model_fuse is not None:
            return self.model_fuse.history.history
        return None

    def on_predict_batch_begin(self, batch):
        if self.predict_callbacks:
            self.predict_callbacks.on_predict_batch_begin(batch)

    def on_predict_batch_end(self, batch):
        if self.predict_callbacks:
            self.predict_callbacks.on_predict_batch_end(batch)

    def on_predict_begin(self):
        if self.predict_callbacks:
            self.predict_callbacks.on_predict_begin()

    def on_predict_end(self):
        if self.predict_callbacks:
            self.predict_callbacks.on_predict_end()

    def set_sample_weight(self, sample_weight, stage="train"):
        if stage == "train":
            self.train_sample_weight = sample_weight
        elif stage == "eval":
            self.eval_sample_weight = sample_weight
        else:
            raise Exception("Illegal Argument")

    def fuse_net(
        self,
        forward_data: Union[List[ForwardData], ForwardData],
        _num_returns: int = 2,
    ):
        """Fuses the hidden layer and calculates the reverse gradient
        only on the side with the label

        Args:
            forward_data: A list of ForwardData containing hidden layers, losses, etc.
                that are uploaded by each party for computation.
        Returns:
            gradient Of hiddens
        """
        assert (
            self.model_fuse is not None
        ), "Fuse model cannot be none, please give model define"
        if isinstance(forward_data, ForwardData):
            forward_data = [forward_data]
        forward_data = list(forward_data)
        forward_data[:] = (h for h in forward_data if h is not None)
        for i, h in enumerate(forward_data):
            assert h.hidden is not None, f"hidden cannot be found in forward_data[{i}]"
            if isinstance(h.losses, List) and h.losses[0] is None:
                h.losses = None
        # get reg losses:
        losses = [h.losses for h in forward_data if h.losses is not None]
        hidden_features = [h.hidden for h in forward_data]
        hiddens = []
        for h in hidden_features:
            # h will be list, if basenet is multi output
            if isinstance(h, List):
                for i in range(len(h)):
                    hiddens.append(tf.convert_to_tensor(h[i]))
            else:
                hiddens.append(tf.convert_to_tensor(h))

        logs = {}

        gradient = self._fuse_net_train(hiddens, losses)

        for m in self.model_fuse.metrics:
            logs["train_" + m.name] = m.result().numpy()
        self.logs = logs
        # In some strategies, we don't need to return gradient.
        if self.skip_gradient:
            return [None] * _num_returns
        return gradient

    def _fuse_net_train(self, hiddens, losses=[]):
        return self._fuse_net_internal(
            hiddens,
            losses,
            self.train_y,
            self.train_sample_weight,
        )

    @tf.function
    def _fuse_net_internal(self, hiddens, losses, train_y, train_sample_weight):
        with tf.GradientTape(persistent=True) as tape:
            for h in hiddens:
                tape.watch(h)

            # Step 1: forward pass
            y_pred = self.model_fuse(hiddens, training=True, **self.kwargs)
            # Step 2: loss calculation, the loss function is configured in `compile()`.
            # if isinstance(self.model_fuse.loss, tfutils.custom_loss):
            #     self.model_fuse.loss.with_kwargs(kwargs)
            loss = self.model_fuse.compiled_loss(
                train_y,
                y_pred,
                sample_weight=train_sample_weight,
                regularization_losses=self.model_fuse.losses + losses,
            )

        # Step3: compute gradients
        trainable_vars = self.model_fuse.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.model_fuse.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Step4: update metrics
        self.model_fuse.compiled_metrics.update_state(
            train_y, y_pred, sample_weight=train_sample_weight
        )

        return tape.gradient(loss, hiddens)

    def reset_metrics(self):
        self.model_fuse.compiled_metrics.reset_state()
        self.model_fuse.compiled_loss.reset_state()

    @tf.function
    def _evaluate_internal(self, hiddens, eval_y, eval_sample_weight, losses=None):
        # Step 1: forward pass
        y_pred = self.model_fuse(hiddens, training=False, **self.kwargs)

        # Step 2: update loss
        # custom loss will be re-open in the next version
        # if isinstance(self.model_fuse.loss, tfutils.custom_loss):
        #     self.model_fuse.loss.with_kwargs(kwargs)
        self.model_fuse.compiled_loss(
            eval_y,
            y_pred,
            sample_weight=eval_sample_weight,
            regularization_losses=self.model_fuse.losses + losses,
        )
        # Step 3: update metrics
        self.model_fuse.compiled_metrics.update_state(
            eval_y, y_pred, sample_weight=eval_sample_weight
        )

        result = {}
        for m in self.model_fuse.metrics:
            result[m.name] = m.result()
        return result

    def evaluate(self, forward_data: Union[List[ForwardData], ForwardData]):
        """Returns the loss value & metrics values for the model in test mode.

        Args:
            forward_data: A list of data dictionaries containing hidden layers, losses, etc.
                that are uploaded by each party for computation.
        Returns:
            map of model metrics.
        """

        assert (
            self.model_fuse is not None
        ), "model cannot be none, please give model define"
        if isinstance(forward_data, ForwardData):
            forward_data = [forward_data]
        forward_data = list(forward_data)
        forward_data[:] = (h for h in forward_data if h is not None)
        for i, h in enumerate(forward_data):
            assert h.hidden is not None, f"hidden cannot be found in forward_data[{i}]"
            if isinstance(h.losses, List) and h.losses[0] is None:
                h.losses = None
        # get reg losses:
        losses = [h.losses for h in forward_data if h.losses is not None]
        hidden_features = [h.hidden for h in forward_data]
        hiddens = []
        for h in hidden_features:
            if isinstance(h, List):
                for i in range(len(h)):
                    hiddens.append(tf.convert_to_tensor(h[i]))
            else:
                hiddens.append(tf.convert_to_tensor(h))
        metrics = self._evaluate_internal(
            hiddens=hiddens,
            eval_y=self.eval_y,
            eval_sample_weight=self.eval_sample_weight,
            losses=losses,
        )
        result = {}
        for k, v in metrics.items():
            result[k] = v.numpy()
        return result

    def wrap_local_metrics(self):
        wraped_metrics = []
        for m in self.model_fuse.metrics:
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
                    f"Unsupported global metric {m.__class__.__qualname__} for now, please add it."
                )
        return wraped_metrics

    def metrics(self):
        return self.wrap_local_metrics()

    @tf.function
    def _predict_internal(self, hiddens):
        y_pred = self.model_fuse(hiddens, training=False)
        return y_pred

    def predict(self, forward_data: Union[List[ForwardData], ForwardData]):
        """Generates output predictions for the input hidden layer features.

        Args:
            forward_data: A list of data dictionaries containing hidden layers,
                that are uploaded by each party for computation.
        Returns:
            Array(s) of predictions.
        """
        assert (
            self.model_fuse is not None
        ), "Fuse model cannot be none, please give model define"
        if isinstance(forward_data, ForwardData):
            forward_data = [forward_data]
        forward_data = list(forward_data)
        forward_data[:] = (h for h in forward_data if h is not None)
        for i, h in enumerate(forward_data):
            assert h.hidden is not None, f"hidden cannot be found in forward_data[{i}]"
            if isinstance(h.losses, List) and h.losses[0] is None:
                h.losses = None
        hidden_features = [h.hidden for h in forward_data]

        hiddens = []
        for h in hidden_features:
            if isinstance(h, List):
                for i in range(len(h)):
                    hiddens.append(tf.convert_to_tensor(h[i]))
            else:
                hiddens.append(tf.convert_to_tensor(h))
        y_pred = self._predict_internal(hiddens)
        return y_pred

    def save_base_model(self, base_model_path: str, **kwargs):
        Path(base_model_path).parent.mkdir(parents=True, exist_ok=True)
        assert base_model_path is not None, "model path cannot be empty"
        self.model_base.save(base_model_path, **kwargs)

    def save_fuse_model(self, fuse_model_path: str, **kwargs):
        Path(fuse_model_path).parent.mkdir(parents=True, exist_ok=True)
        assert fuse_model_path is not None, "model path cannot be empty"
        self.model_fuse.save(fuse_model_path, **kwargs)

    def load_base_model(self, base_model_path: str, **kwargs):
        self.init_data()
        assert base_model_path is not None, "model path cannot be empty"
        self.model_base = tf.keras.models.load_model(base_model_path, **kwargs)

    def load_fuse_model(self, fuse_model_path: str, **kwargs):
        assert fuse_model_path is not None, "model path cannot be empty"
        self.model_fuse = tf.keras.models.load_model(fuse_model_path, **kwargs)

    def export_base_model(self, model_path: str, save_format: str = "onnx", **kwargs):
        return self._export_model(self.model_base, model_path, save_format, **kwargs)

    def export_fuse_model(self, model_path: str, save_format: str = "onnx", **kwargs):
        return self._export_model(self.model_fuse, model_path, save_format, **kwargs)

    def _export_model(
        self, model, model_path: str, save_format: str = "onnx", **kwargs
    ):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        assert model_path is not None, "model path cannot be empty"
        assert save_format in ["onnx", "tf"], "save_format must be 'onnx' or 'tf'"
        if save_format == "onnx":
            return self._export_onnx(model, model_path, **kwargs)
        elif save_format == "tf":
            return self._export_tf(model, model_path, **kwargs)
        else:
            raise Exception("invalid save_format")

    def _export_onnx(self, model, model_path, **kwargs):
        import tf2onnx

        from .utils import wrap_onnx_input_output

        model_proto, _ = tf2onnx.convert.from_keras(
            model, output_path=model_path, **kwargs
        )
        return {
            "inputs": wrap_onnx_input_output(model_proto.graph.input),
            "outputs": wrap_onnx_input_output(model_proto.graph.output),
        }

    def _export_tf(self, model, model_path, **kwargs):
        kwargs["save_format"] = "tf"  # only SavedModel format is supported
        model.save(model_path, **kwargs)

        from tensorflow.python.tools import saved_model_utils

        from .utils import wrap_tf_input_output

        tag_set = "serve"
        signature_def_key = "serving_default"
        meta_graph_def = saved_model_utils.get_meta_graph_def(model_path, tag_set)
        if signature_def_key not in meta_graph_def.signature_def:
            raise ValueError(
                f'Could not find signature "{signature_def_key}". Please choose from: '
                f'{", ".join(meta_graph_def.signature_def.keys())}'
            )
        inputs = meta_graph_def.signature_def[signature_def_key].inputs
        outputs = meta_graph_def.signature_def[signature_def_key].outputs
        return {
            "inputs": wrap_tf_input_output(inputs),
            "outputs": wrap_tf_input_output(outputs),
        }

    def get_privacy_spent(self, step: int, orders=None):
        """Get accountant of dp mechanism.

        Args:
            step: The current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """
        privacy_dict = self.dp_strategy.get_privacy_spent(step, orders)
        return privacy_dict

    def get_skip_gradient(self):
        return False


"""
@register_strategy(strategy_name='split_nn', backend='tensorflow')
@proxy(PYUObject)
class PYUSLTFModel(SLBaseTFModel):
    pass
"""
