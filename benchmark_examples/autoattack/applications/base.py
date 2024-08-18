# Copyright 2023 Ant Group Co., Ltd.
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

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.base import AutoBase
from benchmark_examples.autoattack.global_config import is_simple_test
from benchmark_examples.autoattack.utils.config import read_tune_config
from benchmark_examples.autoattack.utils.data_utils import (
    CustomTensorDataset,
    reveal_data,
    reveal_part_data,
    sample_ndarray,
)
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow import PYU
from secretflow.data import FedNdarray
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.core.torch import TorchModel


class ModelType(Enum):
    """The model types"""

    DNN = 'dnn'
    DEEPFM = 'deepfm'
    RESNET18 = 'resnet18'
    VGG16 = 'vgg16'
    RESNET20 = 'resnet20'
    CNN = 'cnn'
    OTHER = 'other'


class ClassficationType(Enum):
    """The classfication types"""

    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    MULTILABEL = 'multilabel'


class InputMode(Enum):
    SINGLE = 'single'
    MULTI = 'multi'


class DatasetType(Enum):
    TABLE = 'table'
    RECOMMENDATION = 'recommendation'
    IMAGE = 'image'


class ApplicationBaseAPI(AutoBase, ABC):
    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    @abstractmethod
    def dataset_name(self):
        pass

    @abstractmethod
    def train(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        """
        Public API for implement trainning on any applications.
        This function has a default implementation.
        Args:
            callbacks: for attack callbacks, if set, an attack will perform.
            **kwargs
        Returns:
            The trainbing history.
        """
        pass

    @abstractmethod
    def _train(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        """
        Implementation of trainning, need implemented by subclass.
        Args:
            callbacks: for attack callbacks.
            **kwargs:

        Returns:
            The training history.

        """

    @abstractmethod
    def prepare_data(
        self, **kwargs
    ) -> Tuple[FedNdarray, FedNdarray, FedNdarray, FedNdarray]:
        """
        Prepare the train/test data. The subclass must provide this function
        and set the train_data, train_label, test_data, test_label attributes.
        This function has a defualt implementation.
        Returns:
            train_data(FedNdarray)
            train_label(FedNdarray)
            train_label(FedNdarray)
            test_label(FedNdarray)
        """
        pass

    @abstractmethod
    def create_base_model_alice(self, **kwargs) -> TorchModel:
        """
        Create the base model for alice.
        Args:
            **kwargs:

        Returns:
            Torch model.
        """
        pass

    @abstractmethod
    def create_base_model_bob(self, **kwargs) -> TorchModel:
        """
        Create the base model for bob.
        Args:
            **kwargs:

        Returns:
            Torch model.
        """
        pass

    @abstractmethod
    def create_fuse_model(self, **kwargs) -> TorchModel:
        """
        Create the fuse model.
        Args:
            **kwargs:

        Returns:
            Torch model.
        """
        pass

    @abstractmethod
    def classfication_type(self) -> ClassficationType:
        """
        The model classfication type
        Returns:
            The classification type.
        """
        pass

    @abstractmethod
    def base_input_mode(self) -> InputMode:
        """
        The base model input mode, expect 'single' or 'multi'.
        Only when the input is continuous and has only one tensor, return 'single'.
        Other return 'multi'.
        Returns:
            The input mode.
        """
        pass

    @abstractmethod
    def model_type(self) -> ModelType:
        """
        The application model type, like dnn, resnet18, etc.
        Returns:
            The model type.
        """
        pass

    @abstractmethod
    def dataset_type(self) -> DatasetType:
        """
        The dataset type like table, recommendation, image.
        Returns:
            dataset type
        """
        pass

    def create_dataset_builder_alice(self, *args, **kwargs) -> Optional[Callable]:
        """
        The dataset builder for alice.
        """
        raise NotImplementedError("create_dataset_builder_bob not implemented")

    def create_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        """
        The dataset builder for bob.
        """
        raise NotImplementedError("create_dataset_builder_bob not implemented")

    def create_predict_dataset_builder_alice(
        self, *args, **kwargs
    ) -> Optional[Callable]:
        return None

    def create_predict_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return None

    def resources_consumption(self) -> ResourcesPack:
        """
        Indicates the experience value of the resources that the application will consume.
        Please note that the unit of the memory is B.

        Returns:
            ResourcesPack: the resource consumptions with one trail.
        """
        pass


class ApplicationBase(ApplicationBaseAPI, ABC):
    alice: PYU
    bob: PYU
    device_y: PYU
    device_f: PYU
    num_classes: int
    epoch: int
    train_batch_size: int
    hidden_size: int
    dnn_base_units_size_alice: List[int]
    dnn_base_units_size_bob: List[int]
    dnn_fuse_units_size: List[int]
    dnn_embedding_dim: int
    deepfm_embedding_dim: int

    def __init__(
        self,
        alice: PYU | None = None,
        bob: PYU | None = None,
        device_y: PYU | None = None,
        has_custom_dataset: bool = False,
        total_fea_nums: int = -1,
        alice_fea_nums: int = -1,
        num_classes: int = 2,
        epoch=2,
        train_batch_size=128,
        hidden_size=64,
        dnn_base_units_size_alice: Optional[List[int]] = None,
        dnn_base_units_size_bob: Optional[List[int]] = None,
        dnn_fuse_units_size: Optional[List[int]] = None,
        dnn_embedding_dim: Optional[int] = None,
        deepfm_embedding_dim: Optional[int] = None,
    ):
        """
        Application Base Class. Some attributes are initialized here.
        Args:
            alice: Alice's PYU.
            bob:  Bob's PYU.
            device_y: Label device, must be alice or bob.
            num_classes: class nums for this algorithm.
            epoch: train epoch.
            train_batch_size: train batch size.
            hidden_size: hidden size
            dnn_base_units_size_alice: when the model contains dnn layer, need a dnn units size on base model.
            dnn_base_units_size_bob: bob side dnn base units size.
            dnn_fuse_units_size: dnn units size on fuse model.
            dnn_embedding_dim: embedding dim for dnn.
            deepfm_embedding_dim: embedding dim for deepfm.
        """
        super().__init__(alice, bob)
        self.alice = alice
        self.bob = bob
        self.device_y = device_y
        # device_f means the opposite side of device_y
        self.device_f = alice if device_y == bob else bob
        self.total_fea_nums = total_fea_nums
        self.alice_fea_nums = alice_fea_nums
        self.bob_fea_nums = total_fea_nums - self.alice_fea_nums
        self.num_classes = num_classes
        self.epoch = epoch
        self.train_batch_size = train_batch_size
        self.hidden_size = hidden_size
        self.dnn_base_units_size_alice = dnn_base_units_size_alice
        self.dnn_base_units_size_bob = dnn_base_units_size_bob
        self.dnn_fuse_units_size = dnn_fuse_units_size
        self.dnn_embedding_dim = dnn_embedding_dim
        self.deepfm_embedding_dim = deepfm_embedding_dim
        self._has_custom_dataset = has_custom_dataset
        self._is_data_prepared = False
        self._train_data = None
        self._train_label = None
        self._test_data = None
        self._test_label = None
        # for attack, we need some plain data.
        self._plain_train_data = None
        self._plain_test_data = None
        self._plain_train_label = None
        self._plain_test_label = None
        self._plain_train_alice_data = None
        self._plain_train_bob_data = None
        self._plain_test_alice_data = None
        self._plain_test_bob_data = None
        # simple set
        if is_simple_test():
            self.epoch = 1
        self._log_config()
        self.sl_model = None

    def __str__(self):
        return self.dataset_name() + self.model_type().value

    def set_config(self, config: Dict[str, str] | None):
        super().set_config(config)
        self.alice_fea_nums = self.config.get('alice_fea_nums', self.alice_fea_nums)
        self.bob_fea_nums = self.total_fea_nums - self.alice_fea_nums
        self.epoch = self.config.get('epoch', self.epoch)
        self.train_batch_size = self.config.get(
            'train_batch_size', self.train_batch_size
        )
        self.hidden_size = self.config.get('hidden_size', self.hidden_size)
        dnn_base_units_size_alice_ = self.config.get(
            'dnn_base_units_size_alice', self.dnn_base_units_size_alice
        )
        self.dnn_base_units_size_alice = self._handle_dnn_units_size(
            dnn_base_units_size_alice_
        )
        dnn_base_units_size_bob_ = self.config.get(
            'dnn_base_units_size_bob', self.dnn_base_units_size_bob
        )
        self.dnn_base_units_size_bob = self._handle_dnn_units_size(
            dnn_base_units_size_bob_
        )
        if (
            'dnn_base_units_size_bob' not in self.config
            and 'dnn_base_units_size_alice' in self.config
        ):
            # when find bob side's dnn units size is not exists in config but alice exists,
            # then consider bob side same with alice side.
            self.dnn_base_units_size_bob = self.dnn_base_units_size_alice
        self.dnn_fuse_units_size = self.config.get(
            'dnn_fuse_units_size', self.dnn_fuse_units_size
        )
        self.dnn_embedding_dim = self.config.get(
            'dnn_embedding_dim', self.dnn_embedding_dim
        )
        self.deepfm_embedding_dim = self.config.get(
            'deepfm_embedding_dim', self.deepfm_embedding_dim
        )
        self._log_config()

    def _log_config(self):
        """log the application base configuration"""
        logging.warning(
            f"After init, this trail of config is:\n"
            f"total_fea_nums:{self.total_fea_nums}, alice {self.alice_fea_nums}, bob {self.bob_fea_nums}\n"
            f"num_classes:{self.num_classes}, epoch:{self.epoch}, batch_size:{self.train_batch_size}\n"
            f"hidden_size:{self.hidden_size}\n"
            f"dnn_base_units_size alice: {self.dnn_base_units_size_alice}, bob: {self.dnn_base_units_size_bob}\n"
            f"dnn_fuse_units_size:{self.dnn_fuse_units_size}\n"
            f"dnn_embedding_dim:{self.dnn_embedding_dim}, deepfm_embedding_dim:{self.deepfm_embedding_dim}"
        )

    def _handle_dnn_units_size(self, units_size: Optional[List[int]]):
        """
        The user does not know the hidden size before tune dnn units size,
        so the units size is replaced by -1.
        When find negative value, this function will replace them into self.hidden_size.
        For example, -1 will be replacedd by hidden_size, -2 will be replaced by 2 * hidden_size.
        Args:
            units_size: List of int of dnn units size.

        Returns:
            List of int of dnn units size.
        """
        if units_size is None:
            return None
        return [int(-u * self.hidden_size) if u < 0 else u for u in units_size]

    def _prepare_data(self):
        if not self._is_data_prepared:
            self._is_data_prepared = True
            self._train_data, self._train_label, self._test_data, self._test_label = (
                self.prepare_data()
            )
            assert (
                self._train_data is not None
                and self._train_label is not None
                and self._test_data is not None
                and self._test_label is not None
            ), f'{type(self)} prepare_data must return not None type, please check.'

    def get_train_data(self):
        self._prepare_data()
        return self._train_data

    def get_train_label(self):
        self._prepare_data()
        return self._train_label

    def get_test_data(self):
        self._prepare_data()
        return self._test_data

    def get_test_label(self):
        self._prepare_data()
        return self._test_label

    def train(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        histories = self._train(callbacks, **kwargs)
        # Get the final train metrics from the trainning histories.
        train_metrics = {}
        for k, v in histories.items():
            if isinstance(v[-1], np.ndarray):
                assert len(v[-1].shape) == 0
                train_metrics[k] = v[-1].item()
            else:
                train_metrics[k] = v[-1]
        return train_metrics

    def _train(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        """
        The default train implementation.
        Args:
            callbacks: The attack or defense callback list.
            **kwargs: Option parameters.

        """
        base_model_dict = {
            self.alice: self.create_base_model_alice(),
            self.bob: self.create_base_model_bob(),
        }
        dataset_builder_dict = {
            self.alice: self.create_dataset_builder_alice(),
            self.bob: self.create_dataset_builder_bob(),
        }
        if dataset_builder_dict[self.alice] is None:
            dataset_builder_dict = None
        self.sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=self.device_y,
            model_fuse=self.create_fuse_model(),
            backend='torch',
            num_gpus=0.001 if global_config.is_use_gpu() else 0,
            random_seed=global_config.get_random_seed(),
        )
        shuffle = kwargs.get('shuffle', False)
        history = self.sl_model.fit(
            self.get_train_data(),
            self.get_train_label(),
            validation_data=(self.get_test_data(), self.get_test_label()),
            epochs=self.epoch,
            batch_size=self.train_batch_size,
            shuffle=shuffle,
            verbose=1,
            validation_freq=1,
            dataset_builder=dataset_builder_dict,
            callbacks=callbacks,
        )
        logging.warning(f"RESULT: {self} training history = {history}")
        return history

    def predict(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        # make sure the model is trained.
        assert self.sl_model is not None, f'predict must be called after train.'
        return self._predict(callbacks)

    def _predict(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        dataset_builder_dict = {
            self.alice: self.create_predict_dataset_builder_alice(),
            self.bob: self.create_predict_dataset_builder_bob(),
        }
        if dataset_builder_dict[self.alice] is None:
            dataset_builder_dict = None
        return self.sl_model.predict(
            self.get_test_data(),
            self.train_batch_size,
            dataset_builder=dataset_builder_dict,
            callbacks=callbacks,
        )

    def create_dataset_builder_alice(self, *args, **kwargs) -> Optional[Callable]:
        return None

    def create_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return None

    def create_predict_dataset_builder_alice(
        self, *args, **kwargs
    ) -> Optional[Callable]:
        return None

    def create_predict_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return None

    def get_device_y_fea_nums(self):
        assert (
            self.alice_fea_nums >= 0 and self.bob_fea_nums >= 0
        ), f"alice/bob's fea_nums need to init in ApplicationBase subcalss! got {self.alice_fea_nums}, {self.bob_fea_nums}"
        return self.alice_fea_nums if self.device_y == self.alice else self.bob_fea_nums

    def get_device_f_fea_nums(self):
        assert (
            self.alice_fea_nums >= 0 and self.bob_fea_nums >= 0
        ), f"alice/bob's fea_nums need to init in ApplicationBase subcalss! got {self.alice_fea_nums}, {self.bob_fea_nums}"
        return self.alice_fea_nums if self.device_f == self.alice else self.bob_fea_nums

    def get_total_fea_nums(self):
        return self.total_fea_nums

    def get_plain_train_data(self):
        if self._plain_train_data is not None:
            return self._plain_train_data
        self._plain_train_data = reveal_data(self.get_train_data())
        return self.get_plain_train_data()

    def get_plain_test_data(self):
        if self._plain_test_data is not None:
            return self._plain_test_data
        self._plain_test_data = reveal_data(self.get_test_data())
        return self.get_plain_test_data()

    def get_plain_train_alice_data(self):
        if self._plain_train_alice_data is not None:
            return self._plain_train_alice_data
        self._plain_train_alice_data = reveal_part_data(
            self.get_train_data(), self.alice
        )
        return self.get_plain_train_alice_data()

    def get_plain_train_bob_data(self):
        if self._plain_train_bob_data is not None:
            return self._plain_train_bob_data
        self._plain_train_bob_data = reveal_part_data(self.get_train_data(), self.bob)
        return self.get_plain_train_bob_data()

    def get_plain_train_device_y_data(self):
        return (
            self.get_plain_train_alice_data()
            if self.device_y == self.alice
            else self.get_plain_train_bob_data()
        )

    def get_plain_train_device_f_data(self):
        return (
            self.get_plain_train_alice_data()
            if self.device_f == self.alice
            else self.get_plain_train_bob_data()
        )

    def get_plain_test_alice_data(self):
        if self._plain_test_alice_data is not None:
            return self._plain_test_alice_data
        self._plain_test_alice_data = reveal_part_data(self.get_test_data(), self.alice)
        return self.get_plain_test_alice_data()

    def get_plain_test_bob_data(self):
        if self._plain_test_bob_data is not None:
            return self._plain_test_bob_data
        self._plain_test_bob_data = reveal_part_data(self.get_test_data(), self.bob)
        return self.get_plain_test_bob_data()

    def get_plain_test_device_y_data(self):
        return (
            self.get_plain_test_alice_data()
            if self.device_y == self.alice
            else self.get_plain_test_bob_data()
        )

    def get_plain_test_device_f_data(self):
        return (
            self.get_plain_test_alice_data()
            if self.device_f == self.alice
            else self.get_plain_test_bob_data()
        )

    def get_plain_train_label(self):
        if self._plain_train_label is not None:
            return self._plain_train_label
        self._plain_train_label = reveal_data(self.get_train_label())
        if len(self._plain_train_label.shape) > 0:
            if len(self._plain_train_label.shape) == 1:
                self._plain_train_label = self._plain_train_label[:, np.newaxis]
            assert (
                len(self._plain_train_label.shape) == 2
                and self._plain_train_label.shape[1] == 1
            )
            self._plain_train_label = self._plain_train_label.flatten().astype(np.int64)
        return self.get_plain_train_label()

    def get_plain_test_label(self):
        if self._plain_test_label is not None:
            return self._plain_test_label
        self._plain_test_label = reveal_data(self.get_test_label())
        if len(self._plain_test_label.shape) > 0:
            if len(self._plain_test_label.shape) == 1:
                self._plain_test_label = self._plain_test_label[:, np.newaxis]

            assert (
                len(self._plain_test_label.shape) == 2
                and self._plain_test_label.shape[1] == 1
            )
            self._plain_test_label = self._plain_test_label.flatten().astype(np.int64)
        return self.get_plain_test_label()

    def get_train_lable_neg_pos_counts(self) -> Tuple[int, int]:
        x = np.bincount(self.get_plain_train_label())
        if len(x) != 2:
            raise RuntimeError(f"neg_pos counts need 2 classes, got {len(x)} classes.")
        neg, pos = x
        return neg, pos

    def get_device_y_input_shape(self):
        return (
            list(self.get_plain_train_alice_data().shape)
            if self.device_y == self.alice
            else list(self.get_plain_train_bob_data().shape)
        )

    def get_device_f_input_shape(self):
        return (
            list(self.get_plain_train_alice_data().shape)
            if self.device_f == self.alice
            else list(self.get_plain_train_bob_data().shape)
        )

    def sample_device_y_train_data(self, sample_size: int = None, frac: float = None):
        data = (
            self.get_plain_train_alice_data()
            if self.device_y == self.alice
            else self.get_plain_train_bob_data()
        )
        data, _ = self._get_sample_data(data, None, sample_size, frac)
        return data.astype(np.float32)

    def sample_device_f_train_data(self, sample_size: int = None, frac: float = None):
        data = (
            self.get_plain_train_alice_data()
            if self.device_f == self.alice
            else self.get_plain_train_bob_data()
        )
        data, _ = self._get_sample_data(data, None, sample_size, frac)
        return data.astype(np.float32)

    def _check_custom_dataset(self):
        if self._has_custom_dataset:
            raise RuntimeError(
                "Application with custom dataset need to override the get_xxx_dataset methods."
            )

    @staticmethod
    def _get_sample_data(
        data: np.ndarray,
        label: np.ndarray | None = None,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if sample_size is not None or frac is not None or indexes is not None:
            data, indexes = sample_ndarray(
                data, sample_size=sample_size, frac=frac, indexes=indexes
            )
            if label is not None:
                label, _ = sample_ndarray(label, indexes=indexes)

        return data, label

    def _get_sample_dataset(
        self,
        data: np.ndarray,
        label: np.ndarray | None,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int | None = 0,
        **kwargs,
    ):
        data, label = self._get_sample_data(data, label, sample_size, frac, indexes)
        datasets = CustomTensorDataset(data, label, enable_label=enable_label, **kwargs)
        return datasets

    def get_device_y_train_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 0,
        **kwargs,
    ):
        """dataset after preprocess."""
        self._check_custom_dataset()
        data = self.get_plain_train_device_y_data()
        label = self.get_plain_train_label()
        return self._get_sample_dataset(
            data, label, sample_size, frac, indexes, enable_label=enable_label, **kwargs
        )

    def get_device_f_train_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 1,
        **kwargs,
    ):
        self._check_custom_dataset()
        data = self.get_plain_train_device_f_data()
        label = self.get_plain_train_label()
        return self._get_sample_dataset(
            data, label, sample_size, frac, indexes, enable_label=enable_label, **kwargs
        )

    def get_device_y_test_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 0,
        **kwargs,
    ):
        """dataset after preprocess."""
        self._check_custom_dataset()
        data = self.get_plain_test_device_y_data()
        label = self.get_plain_test_label()
        return self._get_sample_dataset(
            data, label, sample_size, frac, indexes, enable_label=enable_label, **kwargs
        )

    def get_device_f_test_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 1,
        **kwargs,
    ):
        self._check_custom_dataset()
        data = self.get_plain_test_device_f_data()
        label = self.get_plain_test_label()
        return self._get_sample_dataset(
            data, label, sample_size, frac, indexes, enable_label=enable_label, **kwargs
        )

    def search_space(self) -> Dict:
        tune_config: dict = read_tune_config(global_config.get_config_file_path())
        assert (
            'applications' in tune_config
        ), f"Missing 'application' after 'tune' in config file."
        application_config = tune_config['applications']
        assert (
            self.dataset_name() in application_config
        ), f"Missing {self.dataset_name()} in config file."
        dataset_config = application_config[self.dataset_name()]
        assert (
            self.model_type().value in dataset_config
        ), f"Missing {self.model_type().value} in config file."
        app_config = dataset_config[self.model_type().value]
        search_space = app_config if app_config is not None else {}
        if global_config.is_simple_test():
            search_space['train_batch_size'] = [32]
            search_space.pop('hidden_size', None)
            search_space.pop('dnn_base_units_size_bob', None)
            search_space.pop('dnn_base_units_size_alice', None)
            search_space.pop('dnn_fuse_units_size', None)
            search_space.pop('dnn_embedding_dim', None)
            search_space.pop('deepfm_embedding_dim', None)
        return search_space

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._train_data, self._train_label, self._test_data, self._test_label
        del self._plain_train_data, self._plain_test_data
        del self._plain_train_label, self._plain_test_label
        del self._plain_train_alice_data, self._plain_train_bob_data
        del self._plain_test_alice_data, self._plain_test_bob_data
        del self.sl_model
