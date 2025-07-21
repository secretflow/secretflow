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

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torchmetrics import AUROC, Accuracy

from secretflow.data.split import train_test_split
from secretflow.preprocessing import StandardScaler
from secretflow.utils.simulation.data.dataframe import create_df
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.attacks.direction_based_scoring_torch import (
    DirectionBasedScoringAttack,
)
from secretflow_fl.utils.simulation.datasets_fl import load_criteo_unpartitioned

from .model_def import WideDeepBottomAlice, WideDeepBottomBob, WideDeepFuse


def test_direct_based_scoring_lia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob

    random_state = 1234
    num_samples = 200
    # the number for testing purposes. For more details about the dataset, please refer to the code in secretflow.utils.simulation.datasets

    sparse_feature = [
        "C" + str(i) for i in range(1, 27)
    ]  # C represents categorical features
    dense_feature = [
        "I" + str(i) for i in range(1, 14)
    ]  # I represents numerical features
    df = load_criteo_unpartitioned(num_samples)
    df[sparse_feature] = df[sparse_feature].fillna(
        "-1",
    )
    df[dense_feature] = df[dense_feature].fillna(
        "0",
    )

    feat_sizes = {}
    feat_sizes_dense = {feat: 1 for feat in dense_feature}
    feat_sizes_sparse = {feat: len(df[feat].unique()) for feat in sparse_feature}
    feat_sizes.update(feat_sizes_dense)
    feat_sizes.update(feat_sizes_sparse)
    for feat in sparse_feature:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    nms = MinMaxScaler(feature_range=(0, 1))
    df[dense_feature] = nms.fit_transform(df[dense_feature])

    fixlen_feature_columns = [(feat, "sparse") for feat in sparse_feature] + [
        (feat, "dense") for feat in dense_feature
    ]
    dnn_feature_columns = fixlen_feature_columns

    # Split the Dataset as the settings in the paper
    data = create_df(
        source=df,
        parts={
            alice: (1, 40),
            bob: (1, 40),
        },
        axis=1,
        shuffle=False,
        aggregator=None,
        comparator=None,
    )
    label = create_df(
        source=df,
        parts={bob: (0, 1)},
        axis=1,
        shuffle=False,
        aggregator=None,
        comparator=None,
    ).astype(np.float32)
    scaler = StandardScaler()
    data = scaler.fit_transform(data).astype("int64")
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.8, random_state=random_state
    )

    metrics = [
        metric_wrapper(Accuracy, task="binary"),
        metric_wrapper(AUROC, task="binary"),
    ]
    dnn_feature_columns = fixlen_feature_columns
    embedding_size = 12

    base_model_alice = TorchModel(
        model_fn=WideDeepBottomAlice,
        optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-3),
        feat_size=feat_sizes,
        embedding_size=embedding_size,
        dnn_feature_columns=dnn_feature_columns,
    )

    base_model_bob = TorchModel(
        model_fn=WideDeepBottomBob,
        optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-3),
        feat_size=feat_sizes,
    )

    fuse_model = TorchModel(
        model_fn=WideDeepFuse,
        loss_fn=nn.BCELoss,
        optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-3),
        metrics=metrics,
    )

    base_model_dict = {
        alice: base_model_alice,
        bob: base_model_bob,
    }
    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=bob,
        model_fuse=fuse_model,
        simulation=True,
        random_seed=1234,
        strategy="split_nn",
        backend="torch",
    )

    direction_lia = DirectionBasedScoringAttack(attack_party=alice, label_party=bob)
    history = sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=1,
        batch_size=128,
        shuffle=False,
        random_seed=1234,
        callbacks=[direction_lia],
    )
    attack_metrics = direction_lia.get_attack_metrics()
    assert "attack_acc" in attack_metrics
    print(attack_metrics)
    print(history)
