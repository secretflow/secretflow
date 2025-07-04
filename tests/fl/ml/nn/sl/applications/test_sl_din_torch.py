# Copyright 2024 Ant Group Co., Ltd.
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

import tempfile

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, Precision

from secretflow.data.vertical import read_csv
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.applications.sl_din_torch import DINBase, DINFuse
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper

from .test_sl_bst_torch import generate_data

tmp_dir = tempfile.TemporaryDirectory()
lia_path = tmp_dir.name
dataset_download_dir = lia_path + "/dataset_download/ml-1m"
gen_data_path = lia_path + "/data_sl_bst_torch"

fea_emb_input_size = {}


class AliceDataset(Dataset):
    def __init__(self, df, label_df):
        self.df = df
        self.label_df = label_df

    def __getitem__(self, index):
        user_id = torch.tensor([int(self.df["user_id"].iloc[index])])
        seq_ids = torch.tensor(
            [int(sid) for sid in self.df["sequence_movie_ids"].iloc[index].split(",")]
        )
        target_id = torch.tensor([int(self.df["target_movie_id"].iloc[index])])
        label = 1 if self.label_df["label"].iloc[index] > 3 else 0
        return (user_id, target_id, seq_ids), label

    def __len__(self):
        return len(self.label_df)


class BobDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        gender = torch.tensor([self.df["gender"].iloc[index]])
        age = torch.tensor([self.df["age_group"].iloc[index]])
        occupation = torch.tensor([self.df["occupation"].iloc[index]])
        return (gender, age, occupation)

    def __len__(self):
        return len(self.df)


def create_dataset_builder_alice(batch_size=32):
    def dataset_builder(x):
        data_set = AliceDataset(x[0], x[1])
        dataloader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


def create_dataset_builder_bob(batch_size=32):
    def dataset_builder(x):
        data_set = BobDataset(x[0])
        dataloader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


def create_base_model_alice():
    def create_model():
        fea_emb_size = {
            "user_id": [fea_emb_input_size["user_id"], 8],
            "target_id": [fea_emb_input_size["target_id"], 8],
        }
        fea_list = ["user_id", "target_id", "sequence_movie_ids"]
        model = DINBase(
            fea_list=fea_list,
            fea_emb_dim=fea_emb_size,
            sequence_fea=["sequence_movie_ids"],
            target_item_fea="target_id",
            seq_len={"sequence_movie_ids": 4},
            padding_idx=0,
        )
        return model

    return create_model


def create_base_model_bob():
    def create_model():
        # 定义特征列表
        fea_list = ["gender", "age_group", "occupation"]
        fea_emb_size = {}

        for key in fea_list:
            fea_emb_size[key] = [fea_emb_input_size[key], 8]

        model = DINBase(
            fea_list=fea_list,
            fea_emb_dim=fea_emb_size,
            sequence_fea=[],
            target_item_fea=None,
            seq_len=None,
        )
        return model

    return create_model


def create_fuse_model():
    def create_model():
        model = DINFuse(dnn_units_size=[48, 24])
        return model

    return create_model


def test_sl_din(sf_simulation_setup_devices):

    generate_data(gen_data_path, fea_emb_input_size)
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob

    batch_size = 64
    dataset_buidler_dict = {
        alice: create_dataset_builder_alice(batch_size=batch_size),
        bob: create_dataset_builder_bob(batch_size=batch_size),
    }

    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

    base_model_alice = TorchModel(
        model_fn=create_base_model_alice(),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(Accuracy, task="multiclass", num_classes=2, average="micro"),
            metric_wrapper(
                Precision, task="multiclass", num_classes=2, average="micro"
            ),
            metric_wrapper(AUROC, task="binary"),
        ],
    )

    base_model_bob = TorchModel(
        model_fn=create_base_model_bob(),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(Accuracy, task="multiclass", num_classes=2, average="micro"),
            metric_wrapper(
                Precision, task="multiclass", num_classes=2, average="micro"
            ),
            metric_wrapper(AUROC, task="binary"),
        ],
    )

    fuse_model = TorchModel(
        model_fn=create_fuse_model(),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(Accuracy, task="multiclass", num_classes=2, average="micro"),
            metric_wrapper(
                Precision, task="multiclass", num_classes=2, average="micro"
            ),
            metric_wrapper(AUROC, task="multiclass", num_classes=2),
        ],
    )

    base_model_dict = {
        alice: base_model_alice,
        bob: base_model_bob,
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=alice,
        model_fuse=fuse_model,
        random_seed=1234,
        backend="torch",
    )

    vdf = read_csv(
        {
            alice: gen_data_path + "/train_data_alice.csv",
            bob: gen_data_path + "/train_data_bob.csv",
        },
        delimiter="|",
    )
    label = vdf["label"]
    data = vdf.drop(columns=["label"])

    epoch = 1

    history = sl_model.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=epoch,
        batch_size=batch_size,
        shuffle=False,
        random_seed=1234,
        dataset_builder=dataset_buidler_dict,
    )
    print("history: ", history)
