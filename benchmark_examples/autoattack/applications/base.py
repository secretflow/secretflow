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
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.global_config import is_simple_test
from secretflow import PYU
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.utils import TorchModel


class ApplicationBaseAPI(object):
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
    def prepare_data(self, **kwargs):
        """
        Prepare the train/test data. The subclass must provide this function
        and set the train_data, train_label, test_data, test_label attributes.
        This function has a defualt implementation.
        Returns:
            None
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
    def alice_feature_nums_range(self) -> list:
        """
        To automate the splitting of models and datasets during the automated attack process,
        it is necessary to provide the range of feature counts for each dataset of Alice for adjustment purposes.
        Returns:
            An adjustable list range like [1,2,3].
        """
        pass

    def hidden_size_range(self) -> Optional[list]:
        """
        To automate the hidden size.
        Returns:
            An adjustable list range like [1,2,3].
        """
        return None

    def dnn_base_units_size_range_alice(self) -> Optional[List[List[int]]]:
        """
        Parameter range for tunning dnn units size used in base model (alice side).
        Note that some sizes may be associated with the hidden size, which is also a adjestable
        parameter that can be tuned, so you don't know the specific value.
        In this case, you can use a negative value to indicate its relathionship with hidden size.
        For example, use -1 to represent the hidden size, use -0.5 to represent half of the hidden size.
        Returns:
            An adjustable list of list range like [[64,32],[64,-1]].
        """
        return None

    def dnn_base_units_size_range_bob(self) -> Optional[List[List[int]]]:
        """
        Parameter range for tunning dnn units size used in base model (bob side).
        Refere to dnn_base_units_size_range_alice for details.

        Note that in most cases, we only need alice and bob to use the same dnn units size,
        in which case you only need to make this function return None, do not return same values
        with dnn_base_units_size_range_alice unless you really need to test the combination in which
        alice and bob has different dnn units size.
        Returns:
            An adjustable list of list range like [[64,32],[64,-1]].
        """
        return None

    def dnn_fuse_units_size_range(self) -> Optional[List[List[int]]]:
        """
        Parameter range for tunning dnn units size used in base model (bob side).
        Note that the last element is the output nums of the model.
        Returns:
            An adjustable list of list range like [[64,1],[64,1]].
        """
        return None

    def dnn_embedding_dim_range(self) -> Optional[List[int]]:
        """
        Parameter range for dnn embedding dim used in base model.
        Returns:
            An adjustable list of list range like [16,32].
        """
        return None

    def deepfm_embedding_dim_range(self) -> Optional[List[int]]:
        """
        Parameter range for deepfm embedding dim used in base model.
        Returns:
            An adjustable list of list range like [16,32].
        """
        return None

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

    @staticmethod
    def support_attacks() -> list:
        """
        Indicate which attacks the application supports.
        Returns:
            list of supported attacks names.
        """
        return []

    def lia_auxiliary_data_builder(self, batch_size=16, file_path=None):
        """
        The auxiliary dataset builder for lia attack.
        Args:
            batch_size: lia attack's bachsize.
            file_path: if need a file_path to read.

        Returns:
            A callable dataset builder.
        """
        raise NotImplementedError(
            f"need implement lia_auxiliary_data_builder on {type(self).__name__} "
        )

    def lia_auxiliary_model(self, ema=False):
        """
        Auxiliary modle for lia attack, which need a bottom model same as the application base.
        """
        raise NotImplementedError(
            f"need implement lia_auxiliary_model on {type(self).__name__} "
        )

    def fia_auxiliary_data_builder(self):
        """Fia auxiliary dataset builder"""
        raise NotImplementedError(
            f"need implement fia_auxiliary_data_builder on {type(self).__name__} "
        )

    def fia_victim_mean_attr(self):
        """Fia victim mean data."""
        raise NotImplementedError(
            f"need implement fia_mean_attr on {type(self).__name__}"
        )

    def fia_victim_model_dict(self, victim_model_save_path):
        raise NotImplementedError(
            f"need implement fia_victim_model_dict on {type(self).__name__}"
        )

    def fia_victim_input_shape(self):
        raise NotImplementedError(
            f"need implement fia_victim_input_shape on {type(self).__name__}"
        )

    def fia_attack_input_shape(self):
        raise NotImplementedError(
            f"need implement fia_attack_input_shape on {type(self).__name__}"
        )

    def replay_auxiliary_attack_configs(
        self, target_nums: int = 15
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        replace auxiliary data indexes
        Args:
            target_nums: the number of targets choice, 10 - 50.

        Returns:
            Tuple[int, np.ndarray, np.ndarray]: the target class, target_indexes, eval_indexes
        """
        raise NotImplementedError(
            f"need implement replay_auxiliary_data_indexes on {type(self).__name__}"
        )

    def replace_auxiliary_attack_configs(
        self, target_nums: int = 15
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The auxiliary attack configs for replace attack.
        Args:
            target_nums: how many poison targets to choose.

        Returns:
            Tuple of 5 emelents:
            target_class (int): the target class, if binary class, set 1.
            target_set (np.ndarray): target set based on the target class.
            train_poison_set (np.ndarray): choose some poison indexes.
            train_poison_np (np.ndarray): the poision data based on the train_poison_set.
            eval_poison_set (np.ndarray): choose some poison data on test datasets.
        """
        raise NotImplementedError(
            f"need implement replace_auxiliary_attack_configs on {type(self).__name__}"
        )

    def exploit_label_counts(self) -> Tuple[int, int]:
        raise NotImplementedError(
            f"need implement exploit_label_counts on {type(self).__name__}"
        )


class ApplicationBase(ApplicationBaseAPI, ABC):
    config: Dict
    alice: PYU
    bob: PYU
    device_y: PYU
    device_f: PYU
    num_classes: int
    epoch: int
    train_batch_size: int
    hiddenz_size: int
    dnn_base_units_size_alice: List[int]
    dnn_base_units_size_bob: List[int]
    dnn_fuse_units_size: List[int]
    dnn_embedding_dim: int
    deepfm_embedding_dim: int

    def __init__(
        self,
        config: Dict,
        alice: PYU,
        bob: PYU,
        device_y: PYU,
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
            config: A custom config dict, attributes will first use this dict, and the auto-attack will use this config.
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
        self.config = config
        self.alice = alice
        self.bob = bob
        self.device_y = device_y
        # device_f means the opposite side of device_y
        self.device_f = alice if device_y == bob else bob
        self.total_fea_nums = total_fea_nums
        self.alice_fea_nums = config.get('alice_fea_nums', alice_fea_nums)
        self.bob_fea_nums = total_fea_nums - self.alice_fea_nums
        self.num_classes = num_classes
        self.epoch = config.get('epoch', epoch)
        self.train_batch_size = config.get('train_batch_size', train_batch_size)
        self.hidden_size = config.get('hidden_size', hidden_size)
        dnn_base_units_size_alice_ = config.get(
            'dnn_base_units_size_alice', dnn_base_units_size_alice
        )
        self.dnn_base_units_size_alice = self._handle_dnn_units_size(
            dnn_base_units_size_alice_
        )
        dnn_base_units_size_bob_ = config.get(
            'dnn_base_units_size_bob', dnn_base_units_size_bob
        )
        self.dnn_base_units_size_bob = self._handle_dnn_units_size(
            dnn_base_units_size_bob_
        )
        if (
            self.dnn_base_units_size_bob is None
            and self.dnn_base_units_size_alice is not None
        ):
            # when find bob side is None in tunning, then consider bob side same with alice side.
            self.dnn_base_units_size_bob = self.dnn_base_units_size_alice
        self.dnn_fuse_units_size = config.get(
            'dnn_fuse_units_size', dnn_fuse_units_size
        )
        self.dnn_embedding_dim = config.get('dnn_embedding_dim', dnn_embedding_dim)
        self.deepfm_embedding_dim = config.get(
            'deepfm_embedding_dim', deepfm_embedding_dim
        )
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        # simple set
        if is_simple_test():
            self.epoch = 1
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
        If find negative value, this function will replace them into self.hidden_size.
        For example, -1 will be replacedd by hidden_size, -2 will be replaced by 2 * hidden_size.
        Args:
            units_size: List of int of dnn units size.

        Returns:
            List of int of dnn units size.
        """
        if units_size is None:
            return None
        return [int(-u * self.hidden_size) if u < 0 else u for u in units_size]

    def train(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        # make sure self data is prepared before the training.
        assert (
            self.train_data is not None
            and self.train_label is not None
            and self.test_data is not None
            and self.test_label is not None
        ), 'Data should be prepared before training.'
        return self._train(callbacks, **kwargs)

    def _train(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
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
        )
        shuffle = kwargs.get('shuffle', False)
        history = self.sl_model.fit(
            self.train_data,
            self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=self.epoch,
            batch_size=self.train_batch_size,
            shuffle=shuffle,
            verbose=1,
            validation_freq=1,
            dataset_builder=dataset_builder_dict,
            callbacks=callbacks,
        )
        logging.warning(
            f"RESULT: {type(self).__name__} {type(callbacks).__name__} training history = {history}"
        )
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
            self.test_data,
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

    def get_train_data(self):
        assert (
            self.train_data is not None
        ), f"data is None, try call prepare_data first."
        return self.train_data

    def get_train_label(self):
        assert (
            self.train_label is not None
        ), f"data is None, try call prepare_data first."
        return self.train_label

    def get_test_data(self):
        assert self.test_data is not None, f"data is None, try call prepare_data first."
        return self.test_data

    def get_test_label(self):
        assert (
            self.test_label is not None
        ), f"data is None, try call prepare_data first."
        return self.test_label

    def get_device_y_fea_nums(self):
        assert (
            self.alice_fea_nums >= 0 and self.bob_fea_nums >= 0
        ), f"alice/bob's fea_nums need to init in ApplicationBase subcalss! got {self.alice_fea_nums}, {self.bob_fea_nums}"
        return self.alice_fea_nums if self.device_y == self.alice else self.bob_fea_nums

    def get_device_f_fea_nums(self):
        assert (
            self.alice_fea_nums >= 0 and self.bob_fea_nums >= 0
        ), f"alice/bob's fea_nums need to init in ApplicationBase subcalss! got {self.abob_fea_numslice_fea_nums}, {self.bob_fea_nums}"
        return self.alice_fea_nums if self.device_f == self.alice else self.bob_fea_nums

    def get_total_fea_nums(self):
        return self.total_fea_nums
