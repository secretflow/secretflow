#!/usr/bin/env python
# coding=utf-8
import secretflow as sf
import tensorflow as tf

# # # from secretflow.ml.nn import SLModel
from custom_base.custom_sl_model import CustomSLModel
from secretflow.data.ndarray import load
from secretflow.device import reveal

import torch
import numpy as np
from torch.nn import functional as F
import pandas as pd

import pdb
from tqdm import *

from test_model.tf.fuse_model import get_fuse_model
from test_model.tf_model_config import MODELS
from dataset.dataset_config import DATASETS
from attack.attack_config import BADNETS_ARGS, POISONING_ARGS
from config import METHODS, AGGREGATIONS, TIMES, PARTIES, PARTY_NUM
from tools.metric import asr
from tools.logger import config_logger, print_log

config_logger(fname="./logs/baseline_targets.log")

party_num = PARTY_NUM
for target_class in range(3, 10):
    sf.shutdown()

    # prepare parties
    parties = []
    for i in range(party_num):
        parties.append(PARTIES[i])

    party_devices = []
    for party in parties:
        party_devices.append(sf.PYU(party))

    sf.init(parties=parties, log_to_driver=False, address="local")
    for ds_name, ds_config in DATASETS.items():
        if ds_name not in ["cifar10", "mnist"]:
            continue

        for agg in AGGREGATIONS:
            for blurred in [True]:
                for method in METHODS:
                    model_config = MODELS[ds_name]

                    ds_path = ds_config["data_path"]
                    ds_args = ds_config["args"]
                    orig_dataset = ds_config["normal"](ds_name, ds_path, ds_args)
                    labels_set = orig_dataset.get_label_set()
                    num_classes = len(labels_set)

                    for i in range(TIMES):
                        # prepare data
                        BADNETS_ARGS["target_class"] = target_class
                        dst_dataset = ds_config["badnets"](
                            ds_name, ds_path, ds_args, BADNETS_ARGS
                        )
                        POISONING_ARGS[
                            "train_poisoning_indexes"
                        ] = dst_dataset.get_train_poisoning_indexes()
                        POISONING_ARGS[
                            "valid_poisoning_indexes"
                        ] = dst_dataset.get_valid_poisoning_indexes()
                        POISONING_ARGS[
                            "train_target_indexes"
                        ] = dst_dataset.get_train_target_indexes()
                        POISONING_ARGS[
                            "valid_target_indexes"
                        ] = dst_dataset.get_valid_target_indexes()

                        (
                            train_passive_datas,
                            train_active_data,
                        ) = dst_dataset.split_train(
                            party_num=party_num, channel_first=False
                        )
                        (
                            valid_passive_datas,
                            valid_active_data,
                        ) = dst_dataset.split_valid(
                            party_num=party_num, channel_first=False
                        )

                        party_feature_shapes = []
                        for passive_data in train_passive_datas:
                            feature_shape = passive_data.party_features.shape[1:]
                            party_feature_shapes.append(feature_shape)

                        data_names = {
                            "y_train": "data/y_train.npy",
                            "y_valid": "data/y_valid.npy",
                        }

                        data_files = {
                            "data/y_train.npy": F.one_hot(
                                train_active_data.party_labels, num_classes
                            )
                            .float()
                            .numpy(),
                            "data/y_valid.npy": F.one_hot(
                                valid_active_data.party_labels, num_classes
                            )
                            .float()
                            .numpy(),
                        }

                        for i, (party, party_device) in enumerate(
                            zip(parties, party_devices)
                        ):
                            train_file = "data/" + "x_{}_train.npz".format(party)
                            valid_file = "data/" + "x_{}_valid.npz".format(party)

                            data_names[party_device] = {
                                "train": train_file,
                                "valid": valid_file,
                            }

                            data_files[train_file] = {
                                "data": train_passive_datas[i].party_features.numpy(),
                                "indexes": train_passive_datas[i].party_indexes.numpy(),
                            }

                            data_files[valid_file] = {
                                "data": valid_passive_datas[i].party_features.numpy(),
                                "indexes": valid_passive_datas[i].party_indexes.numpy(),
                            }

                        for i, party_device in enumerate(party_devices[:-1]):
                            POISONING_ARGS[party_device] = {}

                        POISONING_ARGS[party_devices[-1]] = {
                            "train_poisoning_indexes": POISONING_ARGS[
                                "train_poisoning_indexes"
                            ],
                            "train_features": train_passive_datas[
                                -1
                            ].party_features.numpy(),
                            "valid_poisoning_indexes": POISONING_ARGS[
                                "valid_poisoning_indexes"
                            ],
                            "train_target_indexes": POISONING_ARGS[
                                "train_target_indexes"
                            ],
                            "valid_target_indexes": POISONING_ARGS[
                                "valid_target_indexes"
                            ],
                            "blurred": blurred,
                            "gamma": POISONING_ARGS["gamma"],
                        }

                        for fname, data in data_files.items():
                            if isinstance(data, dict):
                                np.savez(fname, **data)
                            else:
                                np.save(fname, data)

                        # prepare model
                        batch_size = model_config["batch_size"]
                        n_epochs = model_config["epochs"]

                        opt_args = {
                            "class_name": model_config["optimizer"],
                            "config": {
                                "learning_rate": model_config["lr"],
                            },
                        }

                        bottom_compile_args = {
                            "loss": None,
                            "metrics": None,
                        }

                        fuse_compile_args = {
                            "loss": model_config["loss"],
                            "metrics": model_config["metrics"],
                        }

                        bottom_models = {}
                        for feature_shape, party_device in zip(
                            party_feature_shapes, party_devices
                        ):
                            bottom_models[party_device] = model_config["model"](
                                feature_shape,
                                num_classes,
                                opt_args,
                                bottom_compile_args,
                            )
                        model_fuse = get_fuse_model(
                            [num_classes] * party_num,
                            num_classes,
                            agg,
                            opt_args,
                            fuse_compile_args,
                        )

                        train_dict, valid_dict, poison_dict = {}, {}, {}
                        for party_device in party_devices:
                            train_dict[party_device] = data_names[party_device]["train"]
                            valid_dict[party_device] = data_names[party_device]["valid"]

                        x_train = load(train_dict, allow_pickle=True)
                        x_valid = load(valid_dict, allow_pickle=True)
                        y_train = load(
                            {party_devices[0]: data_names["y_train"]}, allow_pickle=True
                        )
                        y_valid = load(
                            {party_devices[0]: data_names["y_valid"]}, allow_pickle=True
                        )

                        x_train = [x_train["data"], x_train["indexes"]]
                        x_valid = [x_valid["data"], x_valid["indexes"]]

                        strategy_dict = {}
                        for party_device in party_devices[:-1]:
                            strategy_dict[party_device] = "index_split_nn"

                        if method == "grad_replacement":
                            strategy_dict[party_devices[-1]] = "gradreplace_split_nn"
                        elif method == "replay":
                            strategy_dict[party_devices[-1]] = "replay_split_nn"
                        elif method == "normal":
                            strategy_dict[party_devices[-1]] = "index_split_nn"
                        else:
                            raise "Invalid method {}!!!".format(method)

                        # ready to train
                        sl_model = CustomSLModel(
                            base_model_dict=bottom_models,
                            device_y=party_devices[0],
                            model_fuse=model_fuse,
                            device_strategy_dict=strategy_dict,
                            attack_args=POISONING_ARGS,
                        )

                        history = sl_model.fit(
                            x=x_train,
                            y=y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=n_epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_freq=10,
                            verbose=1,
                        )
                        evaluation = sl_model.evaluate(x_valid, y_valid)
                        print("main task evaluation:", evaluation)

                        preds = sl_model.predict(x=x_valid, batch_size=batch_size)
                        preds_plain = []
                        for pred in preds:
                            preds_plain.append(tf.argmax(reveal(pred), axis=1))
                        preds_plain = tf.concat(preds_plain, axis=0)
                        y_labels = valid_active_data.party_labels
                        asr_result = asr(
                            preds_plain.numpy(),
                            y_labels,
                            target_class,
                            np.arange(len(y_labels)),
                            POISONING_ARGS["valid_poisoning_indexes"],
                        )

                        print_log(
                            [
                                '{"dataset_name":"' + ds_name + '",',
                                '"method":"' + method + '",',
                                '"party_num":{}'.format(party_num) + ",",
                                '"target_class":' + str(target_class) + ",",
                                '"aggregation":"' + agg + '",',
                                '"org_valid_evaluation":{:.4f}'.format(
                                    evaluation["accuracy"]
                                )
                                + ",",
                                '"asr":{:.4f}'.format(asr_result) + "}",
                            ],
                            oriention="logger",
                        )
print("end")
