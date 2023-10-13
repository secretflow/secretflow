#!/usr/bin/env python
# coding=utf-8
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tensorflow.python.keras import callbacks as callbacks_module
from torch import nn
from torch.nn.modules.loss import _Loss as BaseTorchLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import secretflow.device as ft
from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.metrics import AUC, Mean, Precision, Recall
from secretflow.ml.nn.sl.backend.tensorflow.utils import ForwardData
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.compressor import Compressor, SparseCompressor

# from secretflow.ml.nn.sl.backend.tensorflow.sl_base import SLBaseTFModel
from .sl_base import SLBaseTFModel
import pdb


@register_strategy(strategy_name="replay_split_nn", backend="tensorflow")
@proxy(PYUObject)
class Replay_PYUSLTFModel(SLBaseTFModel):
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        builder_fuse: Callable[[], tf.keras.Model],
        dp_strategy: DPStrategy,
        compressor: Compressor,
        random_seed: int = None,
        **kwargs,
    ):
        super().__init__(
            builder_base, builder_fuse, dp_strategy, compressor, random_seed, **kwargs
        )

        self.attack_args = kwargs.get("attack_args", None)
        self.train_target_indexes = self.attack_args["train_target_indexes"]
        self.valid_poisoning_indexes = self.attack_args["valid_poisoning_indexes"]
        # self.train_target_embeddings = {}
        self.train_target_embeddings = None
        self.target_len = len(self.attack_args["train_target_indexes"])
        self.record_counter = 0

    def forward_record(self, data_indexes, embeddings):
        # find out target samples in a batch
        target_set = np.intersect1d(data_indexes, self.train_target_indexes)
        target_offsets = np.where(data_indexes == target_set[:, None])[-1]

        tlen = len(target_offsets)

        # record target embeddings
        if tlen > 0:
            embeddings_np = embeddings.numpy()

            tshape = (tlen,) + embeddings_np.shape[1:]
            batch_embeddings = embeddings_np[target_offsets] + np.random.randn(*tshape)
            batch_embeddings = embeddings_np[target_offsets]
            embeddings_np[target_offsets] = batch_embeddings

            if self.train_target_embeddings is None:
                self.train_target_embeddings = np.zeros(
                    (self.target_len,) + embeddings.shape[1:]
                )

            self.train_target_embeddings[
                self.record_counter : self.record_counter + tlen
            ] = batch_embeddings
            self.record_counter += tlen
            if self.record_counter >= self.target_len:
                self.record_counter -= self.target_len

            # embeddings = tf.convert_to_tensor(embeddings_np)
        return embeddings

    def forward_replay(self, data_indexes, embeddings):
        # find out poison samples in a batch
        poison_set = np.intersect1d(data_indexes, self.valid_poisoning_indexes)
        poison_offsets = np.where(data_indexes == poison_set[:, None])[-1]
        plen = len(poison_offsets)

        # replay target embeddings
        if plen > 0 and len(self.train_target_embeddings) > 0:
            embeddings_np = embeddings.numpy()
            replay_keys = np.random.choice(
                np.arange(self.target_len), (plen,), replace=True
            )
            embeddings_np[poison_offsets] = self.train_target_embeddings[replay_keys]
            embeddings = tf.convert_to_tensor(embeddings_np)

        return embeddings

    def base_forward(self, stage="train", compress: bool = False) -> ForwardData:
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
            compress: Whether to compress cross device data.
        Returns: hidden embedding
        """

        assert (
            self.model_base is not None
        ), "Base model cannot be none, please give model define or load a trained model"

        data_x = None
        self.init_data()
        if stage == "train":
            train_data = next(self.train_set)
            if self.train_has_y:
                if self.train_has_s_w:
                    data_x = train_data[:-2]
                    train_y = train_data[-2]
                    self.train_sample_weight = train_data[-1]
                else:
                    data_x = train_data[:-1]
                    train_y = train_data[-1]
                # Label differential privacy
                if self.label_dp is not None:
                    dp_train_y = self.label_dp(train_y.numpy())
                    self.train_y = tf.convert_to_tensor(dp_train_y)
                else:
                    self.train_y = train_y
            else:
                data_x = train_data
        elif stage in ["eval_test", "eval"]:
            eval_data = next(self.eval_set)
            if self.eval_has_y:
                if self.eval_has_s_w:
                    data_x = eval_data[:-2]
                    eval_y = eval_data[-2]
                    self.eval_sample_weight = eval_data[-1]
                else:
                    data_x = eval_data[:-1]
                    eval_y = eval_data[-1]
                # Label differential privacy
                if self.label_dp is not None:
                    dp_eval_y = self.label_dp(eval_y.numpy())
                    self.eval_y = tf.convert_to_tensor(dp_eval_y)
                else:
                    self.eval_y = eval_y
            else:
                data_x = eval_data
        else:
            raise Exception("invalid stage")

        # Strip tuple of length one, e.g: (x,) -> x
        # modify: gradient replacement needs features and indexes
        assert len(data_x) >= 2
        data_indexes = data_x[-1]
        data_x = (
            data_x[0]
            if isinstance(data_x[:-1], Tuple) and len(data_x[:-1]) == 1
            else data_x[:-1]
        )

        self.tape = tf.GradientTape(persistent=True)
        with self.tape:
            self.h = self._base_forward_internal(data_x)

        self.data_x = data_x

        # modify:
        # replay attack
        if stage == "train":
            attack_h = self.forward_record(data_indexes, self.h)
        elif stage == "eval":
            attack_h = self.forward_replay(data_indexes, self.h)

        forward_data = ForwardData()
        if len(self.model_base.losses) > 0:
            forward_data.losses = tf.add_n(self.model_base.losses)
        # TODO: only vaild on no server mode, refactor when use agglayer or server mode.
        # no need to compress data on model_fuse side
        if compress and not self.model_fuse:
            if self.compressor:
                # modify:
                forward_data.hidden = self.compressor.compress(attack_h.numpy())
            else:
                raise Exception(
                    "can not find compressor when compress data in base_forward"
                )
        else:
            # modify:
            forward_data.hidden = attack_h
        return forward_data

    def base_backward(self, gradient, compress: bool = False):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
            compress: Whether to decompress gradient.
        """
        return_hiddens = []

        # TODO: only vaild on no server mode, refactor when use agglayer or server mode.
        # no need to decompress data on model_fuse side
        if compress and not self.model_fuse:
            if self.compressor:
                # bug: gradient might be a list?
                gradient = self.compressor.decompress(gradient)
            else:
                raise Exception(
                    "can not find compressor when decompress data in base_backward"
                )

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
