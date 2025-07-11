#!/usr/bin/env python3
# *_* coding: utf-8 *_*
# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.callbacks.early_stopping import (
    EarlyStoppingBatch,
    EarlyStoppingEpoch,
)
from secretflow_fl.utils.simulation.datasets_fl import load_ml_1m

from .test_sl_deepfm import (
    create_base_model_alice,
    create_base_model_bob,
    create_dataset_builder_alice,
    create_dataset_builder_bob,
    create_fuse_model,
)


def test_callback_tf(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob

    # load data
    vdf = load_ml_1m(
        part={
            alice: [
                "UserID",
                "Gender",
                "Age",
                "Occupation",
                "Zip-code",
            ],
            bob: [
                "MovieID",
                "Rating",
                "Title",
                "Genres",
                "Timestamp",
            ],
        },
        num_sample=512,
    )
    label = vdf["Rating"]

    data = vdf.drop(columns=["Rating", "Timestamp", "Title", "Zip-code"])
    data["UserID"] = data["UserID"].astype("string")
    data["MovieID"] = data["MovieID"].astype("string")

    bs = 32

    data_builder_dict = {
        alice: create_dataset_builder_alice(
            batch_size=bs,
            repeat_count=1,
        ),
        bob: create_dataset_builder_bob(
            batch_size=bs,
            repeat_count=1,
        ),
    }
    # User-defined compiled keras model
    device_y = bob
    model_base_alice = create_base_model_alice()
    model_base_bob = create_base_model_bob()
    base_model_dict = {
        alice: model_base_alice,
        bob: model_base_bob,
    }
    model_fuse = create_fuse_model()

    # test EarlyStoppingBatch
    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
    )

    es_step = 1
    cb = EarlyStoppingBatch(es_step, monitor="auc_1", verbose=1, patience=1)
    history = sl_model.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=1,
        batch_size=bs,
        random_seed=1234,
        dataset_builder=data_builder_dict,
        early_stopping_batch_step=es_step,
        early_stopping_warmup_step=0,
        callbacks=[cb],
    )
    print("history: ", history)
    global_metric = sl_model.evaluate(
        data,
        label,
        batch_size=bs,
        random_seed=1234,
        dataset_builder=data_builder_dict,
    )
    print(global_metric)

    # test EarlyStoppingEpoch
    sl_model_2 = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
    )

    cb = EarlyStoppingEpoch(monitor="auc_1", verbose=1, patience=1)
    history = sl_model_2.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=20,
        batch_size=bs,
        random_seed=1234,
        dataset_builder=data_builder_dict,
        callbacks=[cb],
    )
    print("history: ", history)
    global_metric = sl_model_2.evaluate(
        data,
        label,
        batch_size=bs,
        random_seed=1234,
        dataset_builder=data_builder_dict,
    )
    print(global_metric)
