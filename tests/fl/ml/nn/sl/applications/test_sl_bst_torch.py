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

import os
import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, Precision

from secretflow.data.vertical import read_csv
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.applications.sl_bst_torch import BSTBase, BSTBasePlus, BSTfuse
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper

tmp_dir = tempfile.TemporaryDirectory()
lia_path = tmp_dir.name
dataset_download_dir = lia_path + "/dataset_download/ml-1m"
gen_data_path = lia_path + "/data_sl_bst_torch"

fea_emb_input_size = {}


def generate_data(gen_data_path, fea_emb_input_size):
    import numpy as np
    import pandas as pd

    from secretflow.utils.simulation.datasets import _DATASETS, get_dataset, unzip

    global dataset_download_dir
    if not Path(dataset_download_dir).is_dir():
        filepath = get_dataset(_DATASETS["ml-1m"])
        unzip(filepath, dataset_download_dir)

    dataset_dir = dataset_download_dir + "/ml-1m"
    users = pd.read_csv(
        dataset_dir + "/users.dat",
        sep="::",
        names=["user_id", "gender", "age_group", "occupation", "zip_code"],
        engine="python",
    )

    movies = pd.read_csv(
        dataset_dir + "/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="ISO-8859-1",
    )

    ratings = pd.read_csv(
        dataset_dir + "/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        engine="python",
    )

    users["user_id"] = users["user_id"].apply(lambda x: f"{x}")
    users["age_group"] = users["age_group"].apply(lambda x: f"{x}")
    users["occupation"] = users["occupation"].apply(lambda x: f"{x}")

    movies["movie_id"] = movies["movie_id"].apply(lambda x: f"{x}")
    movies["genres"] = movies["genres"].apply(lambda x: ",".join(x.split("|")))

    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"{x}")
    ratings["user_id"] = ratings["user_id"].apply(lambda x: f"{x}")
    ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

    # encode movie_id
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(movies["movie_id"].unique())
    # reserve index 0 for padding
    movies["movie_id"] = le.transform(movies["movie_id"]) + 1
    movies["movie_id"] = movies["movie_id"].astype("string")
    ratings["movie_id"] = le.transform(ratings["movie_id"]) + 1
    ratings["movie_id"] = ratings["movie_id"].astype("string")
    fea_emb_input_size["target_id"] = len(movies["movie_id"].unique()) + 1

    ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

    ratings_data = pd.DataFrame(
        data={
            "user_id": list(ratings_group.groups.keys()),
            "movie_ids": list(ratings_group.movie_id.apply(list)),
            "ratings": list(ratings_group.rating.apply(list)),
            "timestamps": list(ratings_group.unix_timestamp.apply(list)),
        }
    )

    sequence_length = 5
    step_size = 4

    def create_sequences(values, window_size, step_size):
        sequences = []
        start_index = 0
        while True:
            end_index = start_index + window_size
            seq = values[start_index:end_index]
            if len(seq) < window_size:
                if len(seq) > 1:
                    seq.extend(["0"] * (window_size - len(seq)))
                    sequences.append(seq)
                break
            sequences.append(seq)
            start_index += step_size
        return sequences

    ratings_data.movie_ids = ratings_data.movie_ids.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)
    )

    ratings_data.ratings = ratings_data.ratings.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)
    )
    del ratings_data["timestamps"]

    ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode(
        "movie_ids", ignore_index=True
    )
    ratings_data_rating = ratings_data[["ratings"]].explode(
        "ratings", ignore_index=True
    )
    ratings_data_transformed = pd.concat(
        [ratings_data_movies, ratings_data_rating], axis=1
    )
    ratings_data_transformed = ratings_data_transformed.join(
        users.set_index("user_id"), on="user_id"
    )
    ratings_data_transformed["movie_id"] = ratings_data_transformed.movie_ids.apply(
        lambda x: x[-1] if "0" not in x else x[x.index("0") - 1]
    )

    ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
        lambda x: (
            ",".join(x[:-1])
            if "0" not in x
            else ",".join(x[: x.index("0") - 1] + x[x.index("0") :])
        )
    )

    ratings_data_transformed["label"] = ratings_data_transformed.ratings.apply(
        lambda x: x[-1] if "0" not in x else x[x.index("0") - 1]
    )

    ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
        lambda x: ",".join([str(v) for v in x[:-1]])
    )
    ratings_data_transformed = ratings_data_transformed.join(
        movies.set_index("movie_id"), on="movie_id"
    )

    del (
        ratings_data_transformed["zip_code"],
        ratings_data_transformed["title"],
        ratings_data_transformed["genres"],
        ratings_data_transformed["ratings"],
    )

    ratings_data_transformed.rename(
        columns={"movie_ids": "sequence_movie_ids", "movie_id": "target_movie_id"},
        inplace=True,
    )

    ratings_data_transformed = ratings_data_transformed.head(512)

    # encoder: str to id
    le = LabelEncoder()
    ratings_data_transformed[["user_id", "gender", "age_group", "occupation"]] = (
        ratings_data_transformed[
            ["user_id", "gender", "age_group", "occupation"]
        ].apply(le.fit_transform)
    )
    fea_emb_input_size["user_id"] = len(ratings_data_transformed["user_id"].unique())
    fea_emb_input_size["gender"] = len(ratings_data_transformed["gender"].unique())
    fea_emb_input_size["age_group"] = len(
        ratings_data_transformed["age_group"].unique()
    )
    fea_emb_input_size["occupation"] = len(
        ratings_data_transformed["occupation"].unique()
    )

    random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
    train_data = ratings_data_transformed[random_selection]
    test_data = ratings_data_transformed[~random_selection]

    if os.path.exists(gen_data_path):
        shutil.rmtree(gen_data_path)
    os.mkdir(gen_data_path)

    train_data.to_csv(
        gen_data_path + "/train_data.csv", index=False, sep="|", encoding="utf-8"
    )
    test_data.to_csv(
        gen_data_path + "/test_data.csv", index=False, sep="|", encoding="utf-8"
    )

    train_data_alice = train_data[
        ["user_id", "sequence_movie_ids", "target_movie_id", "label"]
    ]
    train_data_bob = train_data[
        [
            "gender",
            "age_group",
            "occupation",
        ]
    ]

    test_data_alice = test_data[
        ["user_id", "sequence_movie_ids", "target_movie_id", "label"]
    ]
    test_data_bob = test_data[
        [
            "gender",
            "age_group",
            "occupation",
        ]
    ]

    train_data_alice.to_csv(
        gen_data_path + "/train_data_alice.csv", index=False, sep="|", encoding="utf-8"
    )
    train_data_bob.to_csv(
        gen_data_path + "/train_data_bob.csv", index=False, sep="|", encoding="utf-8"
    )

    test_data_alice.to_csv(
        gen_data_path + "/test_data_alice.csv", index=False, sep="|", encoding="utf-8"
    )
    test_data_bob.to_csv(
        gen_data_path + "/test_data_bob.csv", index=False, sep="|", encoding="utf-8"
    )


class AliceDataset(Dataset):
    def __init__(self, df, label_df, vocab_dir):
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
    def __init__(self, df, vocab_dir):
        self.df = df

    def __getitem__(self, index):
        gender = torch.tensor([self.df["gender"].iloc[index]])
        age = torch.tensor([self.df["age_group"].iloc[index]])
        occupation = torch.tensor([self.df["occupation"].iloc[index]])

        return (
            gender,
            age,
            occupation,
        )

    def __len__(self):
        return len(self.df)


def create_dataset_builder_alice(batch_size=32):
    def dataset_builder(x):
        data_set = AliceDataset(x[0], x[1], gen_data_path)
        dataloader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


def create_dataset_builder_bob(batch_size=32):
    def dataset_builder(x):
        data_set = BobDataset(x[0], gen_data_path)
        dataloader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


def create_base_model_alice(mode="ori"):
    def create_model():
        fea_emb_size = {
            "user_id": [fea_emb_input_size["user_id"], 8],
            "target_id": [fea_emb_input_size["target_id"], 9],
        }

        fea_list = ["user_id", "target_id", "sequence_movie_ids"]

        if mode == "ori":
            model = BSTBase(
                fea_list=fea_list,
                fea_emb_dim=fea_emb_size,
                sequence_fea=["sequence_movie_ids"],
                target_item_fea="target_id",
                seq_len={"sequence_movie_ids": 4},
                padding_idx=0,
            )
        else:
            model = BSTBasePlus(
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
        fea_list = ["gender", "age_group", "occupation"]
        fea_emb_size = {}
        for key in fea_list:
            fea_emb_size[key] = [fea_emb_input_size[key], 8]

        model = BSTBase(
            fea_list=fea_list,
            fea_emb_dim=fea_emb_size,
            sequence_fea=[],
            target_item_fea=None,
            seq_len=None,
        )

        return model

    return create_model


def create_fuse_model(mode="ori"):
    def create_model():
        if mode == "ori":
            model = BSTfuse(dnn_units_size=[77, 64])
        else:
            model = BSTfuse(dnn_units_size=[50, 64])
        return model

    return create_model


def train_sl_bst_torch(sf_simulation_setup_devices, mode):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob

    vdf = read_csv(
        {
            alice: gen_data_path + "/train_data_alice.csv",
            bob: gen_data_path + "/train_data_bob.csv",
        },
        delimiter="|",
    )
    label = vdf["label"]
    data = vdf.drop(columns=["label"])
    bs = 128
    epoch = 1

    dataset_buidler_dict = {
        alice: create_dataset_builder_alice(batch_size=bs),
        bob: create_dataset_builder_bob(
            batch_size=bs,
        ),
    }
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

    base_model_alice = TorchModel(
        model_fn=create_base_model_alice(mode),
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
        model_fn=create_fuse_model(mode),
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

    history = sl_model.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=epoch,
        batch_size=bs,
        shuffle=False,
        random_seed=1234,
        dataset_builder=dataset_buidler_dict,
    )
    print("history: ", history)


def test_sl_bst(sf_simulation_setup_devices):
    generate_data(gen_data_path, fea_emb_input_size)
    train_sl_bst_torch(sf_simulation_setup_devices, mode="ori")
    train_sl_bst_torch(sf_simulation_setup_devices, mode="plus")
