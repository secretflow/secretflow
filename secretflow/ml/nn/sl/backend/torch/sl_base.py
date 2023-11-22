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

import random
from abc import ABC
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data
import torchmetrics

from secretflow.ml.nn.metrics import AUC, Mean, Precision, Recall
from secretflow.ml.nn.sl.base import SLBaseModel
from secretflow.ml.nn.utils import TorchModel
from secretflow.security.privacy import DPStrategy
from secretflow.utils.communicate import ForwardData


class FuseOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(y)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        return y * grad_output, y * grad_output


class SLBaseTorchModel(SLBaseModel, ABC):
    def __init__(
        self,
        builder_base: Callable[[], TorchModel],
        builder_fuse: Callable[[], TorchModel],
        dp_strategy: DPStrategy,
        random_seed: int = None,
        *args,
        **kwargs,
    ):
        num_gpus = kwargs.get("num_gpus", 0)
        self.use_gpu = num_gpus > 0
        self.dp_strategy = dp_strategy
        self.embedding_dp = (
            self.dp_strategy.embedding_dp if dp_strategy is not None else None
        )
        self.label_dp = self.dp_strategy.label_dp if dp_strategy is not None else None

        self.train_set = None
        self.eval_set = None
        self.valid_set = None
        self.train_iter = None
        self.eval_iter = None
        self.valid_iter = None
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
        # record all logs of training on workers
        self.logs = None
        self.steps_per_epoch = None
        self.shuffle = False
        if random_seed is not None:
            torch.manual_seed(random_seed)
            self.random_seed = random_seed
        # used in backward propagation gradients from fuse model to base model
        self.fuse_op = FuseOp()
        self.model_base = (
            builder_base.model_fn(**builder_base.kwargs)
            if builder_base and builder_base.model_fn
            else None
        )
        self.model_fuse = (
            builder_fuse.model_fn(**builder_fuse.kwargs)
            if builder_fuse and builder_fuse.model_fn
            else None
        )

        self.loss_base = (
            builder_base.loss_fn() if builder_base and builder_base.loss_fn else None
        )
        self.loss_fuse = (
            builder_fuse.loss_fn() if builder_fuse and builder_fuse.loss_fn else None
        )

        self.optim_base = (
            builder_base.optim_fn(self.model_base.parameters())
            if builder_base and builder_base.optim_fn
            else None
        )
        self.optim_fuse = (
            builder_fuse.optim_fn(self.model_fuse.parameters())
            if builder_fuse and builder_fuse.optim_fn
            else None
        )

        self.metrics_fuse = (
            [m() for m in builder_fuse.metrics]
            if builder_fuse and builder_fuse.metrics
            else None
        )
        super().__init__()

    def init_data(self):
        self.train_x, self.train_y = None, None
        self.eval_x, self.eval_y = None, None
        self.train_sample_weight = None
        self.eval_sample_weight = None

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch

    # FIXME: 这里需要修改为torch manner
    def get_basenet_output_num(self):
        if self.model_base:
            if (
                hasattr(self.model_base, 'outputs')
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
                    data_y = torch.tensor(data_y)

        return data_x, data_y, data_s_w

    def get_batch_data(self, stage="train"):
        data_x = None
        self.init_data()

        # init model stat to train

        if stage == "train":
            if self.model_base:
                self.model_base.train()
            train_data = next(self.train_iter)
            (
                self.train_x,
                self.train_y,
                self.train_sample_weight,
            ) = self.unpack_dataset(
                train_data, self.train_has_x, self.train_has_y, self.train_has_s_w
            )
            data_x = self.train_x

        elif stage == "eval":
            if self.model_base:
                self.model_base.eval()
            eval_data = next(self.eval_iter)

            self.eval_x, self.eval_y, self.eval_sample_weight = self.unpack_dataset(
                eval_data, self.eval_has_x, self.eval_has_y, self.eval_has_s_w
            )
            data_x = self.eval_x
        else:
            raise Exception("invalid stage")

        # Strip tuple of length one, e.g: (x,) -> x
        data_x = (
            data_x[0]
            if isinstance(data_x, (Tuple, List)) and len(data_x) == 1
            else data_x
        )

        return data_x

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
        """build torch.data.Dataset

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
        if shuffle and random_seed is not None:
            self.shuffle = shuffle
            random.seed(random_seed)
            torch.manual_seed(random_seed)  # set random seed for cpu
            torch.cuda.manual_seed(random_seed)  # set random seed for cuda
            torch.backends.cudnn.deterministic = True

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

        # convert pandas.DataFrame to torch.tensor
        data_tuple = [
            t.values if isinstance(t, pd.DataFrame) else t for t in data_tuple
        ]
        x_copy = [torch.tensor(t.copy()) for t in data_tuple]
        data_set = torch_data.TensorDataset(*x_copy)
        dataloader = torch_data.DataLoader(
            dataset=data_set,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        self.set_dataset_stage(
            data_set=dataloader,
            stage=stage,
            has_x=has_x,
            has_y=has_y,
            has_s_w=has_s_w,
        )

    def build_dataset_from_builder(
        self,
        *x: Union[List[np.ndarray], List[pd.DataFrame], str],
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        batch_size=-1,
        random_seed=1234,
        stage="train",
        dataset_builder: Callable = None,
    ):
        """build torch.data.Dataset

        Args:
            x: feature, FedNdArray or HDataFrame
            y: label, FedNdArray or HDataFrame
            s_w: sample weight, FedNdArray or HDataFrame
            batch_size:  An integer representing the size of the batches to be used.
            random_seed: An integer representing the random seed used for shuffling.
            stage: stage of this datset
            dataset_builder: dataset build callable function of worker
        """
        assert (
            x is not None or y is not None
        ), f"At least one of feature(x) and label(y) is not None."
        if not dataset_builder:
            return -1

        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)  # set random seed for cpu
            torch.cuda.manual_seed(random_seed)  # set random seed for cuda
            torch.backends.cudnn.deterministic = True

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
        steps_per_epoch = -1
        if isinstance(data_set, tuple):
            assert len(data_set) == 2, (
                f"If a dataset builder return more than 1 value, "
                f"it must return 2, one is dataset, another is steps_per_epoch"
            )
            steps_per_epoch = data_set[1]
            data_set = data_set[0]

        self.set_dataset_stage(
            data_set=data_set,
            stage=stage,
            has_x=has_x,
            has_y=has_y,
            has_s_w=has_s_w,
        )

        if steps_per_epoch != -1:
            return steps_per_epoch
        # Compatible with existing gnn databuilder
        if callable(getattr(data_set, '__len__', None)):
            return len(data_set)
        if hasattr(data_set, 'steps_per_epoch'):
            return data_set.steps_per_epoch

        # Infer batch size
        batch_data = next(iter(data_set))

        if isinstance(batch_data, Tuple):
            batch_data = batch_data[0]
        if isinstance(batch_data, Dict):
            batch_data = list(batch_data.values())[0]

        # FIXME: the batch size handle has problem.
        # Besides, the steps_per_epoch should use ceil rather than floor
        if isinstance(batch_data, torch.Tensor):
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

        if isinstance(data_set, torch.utils.data.DataLoader):
            import math

            total_size = len(x[0]) if x is not None else 0
            return math.ceil(total_size / batch_size)
        else:
            raise Exception("Unknown databuilder")

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
        raise NotImplementedError("Does not support build from csv in torch yet.")

    def set_dataset_stage(
        self,
        data_set,
        stage="train",
        has_x=None,
        has_y=None,
        has_s_w=None,
    ):
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

    def base_forward_internal(self, data_x):
        h = self.model_base(data_x)

        # Embedding differential privacy
        if self.embedding_dp is not None:
            if isinstance(h, List):
                h = [self.embedding_dp(hi) for hi in h]
            else:
                h = self.embedding_dp(h)

        return h

    def fuse_net_internal(self, hiddens, train_y, train_sample_weight, logs):
        # Step 1: forward pass
        self.model_fuse.train()
        if isinstance(hiddens, List):
            hiddens = [h.requires_grad_() for h in hiddens]
            if len(hiddens) == 1:
                hiddens = hiddens[0]
        else:
            hiddens = hiddens.requires_grad_()

        y_pred = self.model_fuse(hiddens, **self.kwargs)

        # Step 2: loss calculation.
        # NOTE: Refer to https://stackoverflow.com/questions/67730325/using-weights-in-crossentropyloss-and-bceloss-pytorch to use sample weight
        # custom loss will be re-open in the next version
        loss = self.loss_fuse(
            y_pred,
            train_y,
        )

        logs["train_loss"] = loss.detach().numpy()

        if isinstance(hiddens, List):
            for h in hiddens:
                h.retain_grad()
        else:
            hiddens.retain_grad()
        self.optim_fuse.zero_grad()
        loss.backward()
        self.optim_fuse.step()

        # Step4: update metrics
        for m in self.metrics_fuse:
            if len(train_y.shape) > 1 and train_y.shape[1] > 1:
                m.update(y_pred, train_y.int().argmax(-1))
            else:
                m.update(y_pred, train_y.int())

        # Step5: calculate the gradients for the inputs
        if isinstance(hiddens, List):
            hiddens_grad = [h.grad for h in hiddens]
        else:
            hiddens_grad = hiddens.grad

        return hiddens_grad

    def fuse_net(
        self,
        forward_data: Union[List[ForwardData], ForwardData],
        _num_returns=2,
    ):
        """Fuses the hidden layer and calculates the reverse gradient
        only on the side with the label

        Args:
            hidden_features: A list of hidden layers for each party to compute
        Returns:
            gradient Of hiddens
        """
        pass

    def reset_metrics(self):
        for m in self.metrics_fuse:
            m.reset()

    def staging_metric_states(self):
        # TODO(caibei)
        raise NotImplementedError()

    def recover_metric_states(self):
        # TODO(caibei)
        raise NotImplementedError()

    def _evaluate_internal(self, hiddens, eval_y, eval_sample_weight, logs):
        # Step 1: forward pass
        self.model_fuse.eval()
        if isinstance(hiddens, List) and len(hiddens) == 1:
            hiddens = hiddens[0]

        output = self.model_fuse(hiddens, **self.kwargs)
        if isinstance(output, Tuple) and len(output) > 1:
            y_pred = output[0]
        else:
            y_pred = output

        # Step 2: update loss
        # custom loss will be re-open in the next version
        loss = self.loss_fuse(
            y_pred,
            eval_y,
        )
        logs["loss"] = loss.detach().numpy()

        # Step 3: update metrics
        for m in self.metrics_fuse:
            if (
                len(eval_y.shape) > 1 and eval_y.shape[1] > 1
            ):  # in case eval_y is of shape [batch_size, 1]
                m.update(y_pred, eval_y.argmax(-1))
            else:
                m.update(y_pred, eval_y.int())

        for m in self.metrics_fuse:
            logs[m.__class__.__name__] = m.compute().numpy()

    def evaluate(
        self,
        forward_data: Union[List[ForwardData], ForwardData],
    ):
        """Returns the loss value & metrics values for the model in test mode.

        Args:
            hidden_features: A list of hidden layers for each party to compute
        Returns:
            map of model metrics.
        """

        assert (
            self.model_fuse is not None
        ), "model cannot be none, please give model define"

        if isinstance(forward_data, ForwardData):
            forward_data = [forward_data]
        forward_data[:] = (h for h in forward_data if h is not None)

        hidden_features = [h.hidden for h in forward_data]

        hiddens = []

        for h in hidden_features:
            # h will be list, if basenet is multi output
            if isinstance(h, List):
                for i in range(len(h)):
                    hiddens.append(torch.tensor(h[i]))
            else:
                hiddens.append(torch.tensor(h))
        eval_y = self.eval_y[0] if len(self.eval_y) == 1 else self.eval_y
        metrics = {}
        self._evaluate_internal(
            hiddens=hiddens,
            eval_y=eval_y,
            eval_sample_weight=self.eval_sample_weight,
            logs=metrics,
        )
        if self.logs is None:
            self.logs = metrics.copy()
            return metrics
        else:
            val_metrics = {}
            for k, v in metrics.items():
                val_metrics[f"val_{k}"] = v
            self.logs.update(val_metrics)
            return metrics

    # TODO: rewrite in torch way
    def wrap_local_metrics(self):
        wraped_metrics = []
        for m in self.metrics_fuse:
            if isinstance(m, torchmetrics.MeanMetric):
                wraped_metrics.append(Mean(m.__class__.__name__, m.total, m.count))
            elif isinstance(m, torchmetrics.AUC):
                wraped_metrics.append(
                    AUC(
                        m.__class__.__name__,
                        m.thresholds,
                        m.true_positives,
                        m.true_negatives,
                        m.false_positives,
                        m.false_negatives,
                        m.curve,
                    )
                )
            elif isinstance(m, torchmetrics.Precision):
                wraped_metrics.append(
                    Precision(
                        m.__class__.__name__,
                        m.thresholds,
                        m.true_positives,
                        m.false_positives,
                    )
                )
            elif isinstance(m, torchmetrics.Recall):
                wraped_metrics.append(
                    Recall(
                        m.__class__.__name__,
                        m.thresholds,
                        m.true_positives,
                        m.false_negatives,
                    )
                )
            else:
                raise NotImplementedError(
                    f'Unsupported global metric {m.__class__.__qualname__} for now, please add it.'
                )
        return wraped_metrics

    def metrics(self):
        return self.wrap_local_metrics()

    def _predict_internal(self, hiddens):
        self.model_fuse.eval()

        if isinstance(hiddens, List) and len(hiddens) == 1:
            hiddens = hiddens[0]

        output = self.model_fuse(hiddens, **self.kwargs)
        if isinstance(output, Tuple) and len(output) > 1:
            y_pred = output[0]
        else:
            y_pred = output
        return y_pred

    def predict(
        self,
        forward_data: Union[List[ForwardData], ForwardData],
    ):
        """Generates output predictions for the input hidden layer features.

        Args:
            hidden_features: A list of hidden layers for each party to compute
        Returns:
            Array(s) of predictions.
        """
        assert (
            self.model_fuse is not None
        ), "Fuse model cannot be none, please give model define"

        if isinstance(forward_data, ForwardData):
            forward_data = [forward_data]
        forward_data[:] = (h for h in forward_data if h is not None)

        hidden_features = [h.hidden for h in forward_data]

        hiddens = []
        for h in hidden_features:
            if isinstance(h, List):
                for i in range(len(h)):
                    hiddens.append(torch.tensor(h[i]))
            else:
                hiddens.append(torch.tensor(h))
        y_pred = self._predict_internal(hiddens)
        return y_pred

    def save_base_model(self, base_model_path: str, **kwargs):
        Path(base_model_path).parent.mkdir(parents=True, exist_ok=True)
        assert base_model_path is not None, "model path cannot be empty"
        check_point = {
            'model_state_dict': self.model_base.state_dict(),
            'optimizer_state_dict': self.optim_base.state_dict(),
        }
        torch.save(check_point, base_model_path, **kwargs)

    def save_fuse_model(self, fuse_model_path: str, **kwargs):
        Path(fuse_model_path).parent.mkdir(parents=True, exist_ok=True)
        assert fuse_model_path is not None, "model path cannot be empty"
        check_point = {
            'model_state_dict': self.model_fuse.state_dict(),
            'optimizer_state_dict': self.optim_fuse.state_dict(),
        }
        torch.save(check_point, fuse_model_path, **kwargs)

    def load_base_model(self, base_model_path: str, **kwargs):
        self.init_data()
        assert base_model_path is not None, "model path cannot be empty"
        assert (
            self.model_base is not None
        ), "model structure must be defined before load"

        checkpoint = torch.load(base_model_path)
        self.model_base.load_state_dict(checkpoint['model_state_dict'])
        self.optim_base.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_fuse_model(self, fuse_model_path: str, **kwargs):
        assert fuse_model_path is not None, "model path cannot be empty"
        assert (
            self.model_fuse is not None
        ), "model structure must be defined before load"

        checkpoint = torch.load(fuse_model_path)
        self.model_fuse.load_state_dict(checkpoint['model_state_dict'])
        self.optim_fuse.load_state_dict(checkpoint['optimizer_state_dict'])

    def export_base_model(self, model_path: str, save_format: str = "onnx", **kwargs):
        return self._export_model(self.model_base, model_path, save_format, **kwargs)

    def export_fuse_model(self, model_path: str, save_format: str = "onnx", **kwargs):
        return self._export_model(self.model_fuse, model_path, save_format, **kwargs)

    def _export_model(
        self, model, model_path: str, save_format: str = "onnx", **kwargs
    ):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        assert model_path is not None, "model path cannot be empty"
        assert save_format in ["onnx", "torch"], "save_format must be 'onnx' or 'torch'"
        if save_format == "onnx":
            return self._export_onnx(model, model_path, **kwargs)
        elif save_format == "torch":
            return self._export_torch(model, model_path, **kwargs)
        else:
            raise Exception("invalid save_format")

    def _export_onnx(self, model, model_path, **kwargs):
        raise NotImplementedError(f'Export to ONNX is not supported')

    def _export_torch(self, model, model_path, **kwargs):
        raise NotImplementedError(f'Export to Torch is not supported')

    def get_privacy_spent(self, step: int, orders=None):
        """Get accountant of dp mechanism.

        Args:
            step: The current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """
        privacy_dict = self.dp_strategy.get_privacy_spent(step, orders)
        return privacy_dict

    def apply(self, func, *args, **kwargs):
        if callable(func):
            return func(self, *args, **kwargs)
        else:
            raise Exception("applyed method must be callable")

    def get_base_weights(self):
        return self.model_base.get_weights()

    def get_fuse_weights(self):
        return self.model_fuse.get_weights() if self.model_fuse is not None else None

    def get_stop_training(self):
        return False  # currently not supported

    def _reset_data_iter(self, stage):
        if self.shuffle:
            # FIXME: need a better way to handle global random state
            torch.manual_seed(self.random_seed)
        if stage == "train" and self.train_set is not None:
            self.train_iter = iter(self.train_set)

        if stage == "eval" and self.eval_set is not None:
            self.eval_iter = iter(self.eval_set)

    def get_logs(self):
        return self.logs

    def set_sample_weight(self, sample_weight, stage="train"):
        if stage == "train":
            self.train_sample_weight = sample_weight
        elif stage == "eval":
            self.eval_sample_weight = sample_weight
        else:
            raise Exception("Illegal Argument")

    def get_skip_gradient(self):
        raise NotImplementedError()
