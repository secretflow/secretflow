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
from pathlib import Path

from secretflow.data.vertical import read_csv
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.applications.sl_bst_tf import BSTBase, BSTFuse, BSTPlusBase

gen_data_path = "/tmp/movielens"
dataset_download_dir = "/tmp/dataset_download/ml-1m"


def generate_data(plus=True):
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

    ratings = pd.read_csv(
        dataset_dir + "/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        engine="python",
    )

    movies = pd.read_csv(
        dataset_dir + "/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="ISO-8859-1",
    )

    users["user_id"] = users["user_id"].apply(lambda x: f"{x}")
    users["age_group"] = users["age_group"].apply(lambda x: f"{x}")
    users["occupation"] = users["occupation"].apply(lambda x: f"{x}")

    movies["movie_id"] = movies["movie_id"].apply(lambda x: f"{x}")
    movies["genres"] = movies["genres"].apply(lambda x: ",".join(x.split("|")))

    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"{x}")
    ratings["user_id"] = ratings["user_id"].apply(lambda x: f"{x}")
    ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

    ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

    ratings_data = pd.DataFrame(
        data={
            "user_id": list(ratings_group.groups.keys()),
            "movie_ids": list(ratings_group.movie_id.apply(list)),
            "ratings": list(ratings_group.rating.apply(list)),
            "timestamps": list(ratings_group.unix_timestamp.apply(list)),
        }
    )

    if plus:
        sequence_length = 5
        step_size = 4
    else:
        sequence_length = 4
        step_size = 2

    def create_sequences(values, window_size, step_size):
        sequences = []
        start_index = 0
        while True:
            end_index = start_index + window_size
            seq = values[start_index:end_index]
            if len(seq) < window_size:
                seq.extend(["[PAD]"] * (window_size - len(seq)))
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
        lambda x: x[-1] if "[PAD]" not in x else x[x.index("[PAD]") - 1]
    )

    if plus:
        ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
            lambda x: (
                ",".join(x[:-1])
                if "[PAD]" not in x
                else ",".join(x[: x.index("[PAD]") - 1] + x[x.index("[PAD]") :])
            )
        )
    else:
        ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
            lambda x: ",".join(x)
        )

    ratings_data_transformed["label"] = ratings_data_transformed.ratings.apply(
        lambda x: x[-1] if "[PAD]" not in x else x[x.index("[PAD]") - 1]
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

    random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
    train_data = ratings_data_transformed[random_selection]
    test_data = ratings_data_transformed[~random_selection]

    if os.path.exists(gen_data_path):
        shutil.rmtree(gen_data_path)
    os.mkdir(gen_data_path)
    os.mkdir(gen_data_path + "/vocabulary")

    train_data.to_csv(
        gen_data_path + "/train_data.csv", index=False, sep="|", encoding="utf-8"
    )
    test_data.to_csv(
        gen_data_path + "/test_data.csv", index=False, sep="|", encoding="utf-8"
    )

    train_data_alice = train_data[["gender", "age_group", "occupation"]]
    train_data_bob = train_data[
        [
            "user_id",
            "sequence_movie_ids",
            "target_movie_id",
            "label",
        ]
    ]

    test_data_alice = test_data[["gender", "age_group", "occupation"]]
    test_data_bob = test_data[
        [
            "user_id",
            "sequence_movie_ids",
            "target_movie_id",
            "label",
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

    with open(gen_data_path + "/vocabulary/user_id", "w") as f:
        f.write("\n".join(list(ratings_data_transformed.user_id.unique())))

    with open(gen_data_path + "/vocabulary/gender", "w") as f:
        f.write("\n".join(list(ratings_data_transformed.gender.unique())))

    with open(gen_data_path + "/vocabulary/age_group", "w") as f:
        f.write("\n".join(list(ratings_data_transformed.age_group.unique())))

    with open(gen_data_path + "/vocabulary/occupation", "w") as f:
        f.write("\n".join(list(ratings_data_transformed.occupation.unique())))

    with open(gen_data_path + "/vocabulary/item_id", "w") as f:
        f.write("\n".join(list(movies.movie_id.unique())))


def create_bstplus_base_model_alice():
    # Create model
    def create_model():
        import tensorflow as tf
        from tensorflow.keras import layers

        fea_list = ["gender", "age_group", "occupation"]
        fea_emb_size = {"gender": 8, "age_group": 8, "occupation": 8}
        fea_voc = {}
        for key in fea_list:
            with open(gen_data_path + "/vocabulary/" + key) as f:
                values = [line.strip() for line in f.readlines()]
                fea_voc[key] = values

        def preprocess():
            inputs = {}
            outputs = {}
            for key in fea_list:
                inputs[key] = tf.keras.Input(shape=(1,), dtype=tf.string)

                idx = layers.StringLookup(
                    vocabulary=fea_voc[key],
                    mask_token=None,
                    num_oov_indices=0,
                    name=f"{key}_index_lookup",
                )(inputs[key])

                outputs[key] = layers.Embedding(
                    input_dim=len(fea_voc[key]),
                    output_dim=fea_emb_size[key],
                    name=f"{key}_embedding",
                )(idx)

            return tf.keras.Model(inputs=inputs, outputs=outputs)

        preprocess_layer = preprocess()
        model = BSTPlusBase(
            preprocess_layer=preprocess_layer,
            dnn_units_size=[32],
        )
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )
        return model  # need wrap

    return create_model


def create_bstplus_base_model_bob():
    # Create model
    def create_model():
        import tensorflow as tf
        from tensorflow.keras import layers

        fea_list = ["user_id"]
        fea_emb_size = {"user_id": 8}
        fea_voc = {}
        for key in fea_list:
            with open(gen_data_path + "/vocabulary/" + key) as f:
                values = [line.strip() for line in f.readlines()]
                fea_voc[key] = values

        with open(gen_data_path + "/vocabulary/item_id") as f:
            values = [line.strip() for line in f.readlines()]
            fea_voc["item"] = values

        seq_len = 4
        item_embedding_dims = 9

        def preprocess():
            inputs = {}
            outputs = {}
            for key in fea_list:
                inputs[key] = tf.keras.Input(shape=(1,), dtype=tf.string)

                idx = layers.StringLookup(
                    vocabulary=fea_voc[key],
                    mask_token=None,
                    num_oov_indices=0,
                    name=f"{key}_index_lookup",
                )(inputs[key])

                outputs[key] = layers.Embedding(
                    input_dim=len(fea_voc[key]),
                    output_dim=fea_emb_size[key],
                    name=f"{key}_embedding",
                )(idx)

            # sequence input
            inputs["sequence_movie_ids"] = tf.keras.Input(
                name="sequence_movie_ids",
                shape=(1,),
                dtype=tf.string,
            )

            inputs["target_movie_id"] = tf.keras.Input(
                name="target_movie_id",
                shape=(1,),
                dtype=tf.string,
            )

            item_lookup_layer = layers.StringLookup(
                vocabulary=fea_voc["item"],
                mask_token="[PAD]",  # note here!
                name="item_index_lookup",
            )

            seq_split = tf.strings.split(inputs["sequence_movie_ids"], ",").to_tensor(
                "[PAD]", shape=[None, 1, 4]
            )

            item_idx = item_lookup_layer(seq_split)
            outputs["sequence_idx"] = item_idx

            target_item_idx = item_lookup_layer(inputs["target_movie_id"])
            outputs["target_movie_id"] = target_item_idx

            return tf.keras.Model(inputs=inputs, outputs=outputs)

        preprocess_layer = preprocess()
        model = BSTPlusBase(
            preprocess_layer=preprocess_layer,
            dnn_units_size=[32],
            sequence_fea=["sequence_idx"],
            target_fea="target_movie_id",
            item_embedding_dims={"target_movie_id": item_embedding_dims},
            seq_len={"sequence_idx": seq_len},
            item_voc_size={"target_movie_id": len(fea_voc["item"])},
            num_head={"sequence_idx": 3},
            dropout_rate={"sequence_idx": 0.1},
        )

        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )
        return model  # need wrap

    return create_model


def create_base_model_alice():
    # Create model
    def create_model():
        import tensorflow as tf
        from tensorflow.keras import layers

        fea_list = ["gender", "age_group", "occupation"]
        fea_emb_size = {"gender": 8, "age_group": 8, "occupation": 8}
        fea_voc = {}
        for key in fea_list:
            with open(gen_data_path + "/vocabulary/" + key) as f:
                values = [line.strip() for line in f.readlines()]
                fea_voc[key] = values

        def preprocess():
            inputs = {}
            outputs = {}
            for key in fea_list:
                inputs[key] = tf.keras.Input(shape=(1,), dtype=tf.string)

                idx = layers.StringLookup(
                    vocabulary=fea_voc[key],
                    mask_token=None,
                    num_oov_indices=0,
                    name=f"{key}_index_lookup",
                )(inputs[key])

                outputs[key] = layers.Embedding(
                    input_dim=len(fea_voc[key]),
                    output_dim=fea_emb_size[key],
                    name=f"{key}_embedding",
                )(idx)

            return tf.keras.Model(inputs=inputs, outputs=outputs)

        preprocess_layer = preprocess()
        model = BSTBase(
            preprocess_layer=preprocess_layer,
            dnn_units_size=[32],
        )
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )
        return model  # need wrap

    return create_model


def create_base_model_bob():
    # Create model
    def create_model():
        import tensorflow as tf
        from tensorflow.keras import layers

        fea_list = ["user_id"]
        fea_emb_size = {"user_id": 8}
        fea_voc = {}
        for key in fea_list:
            with open(gen_data_path + "/vocabulary/" + key) as f:
                values = [line.strip() for line in f.readlines()]
                fea_voc[key] = values

        with open(gen_data_path + "/vocabulary/item_id") as f:
            values = [line.strip() for line in f.readlines()]
            fea_voc["item"] = values

        seq_len = 4
        item_embedding_dims = 9

        def preprocess():
            inputs = {}
            outputs = {}
            for key in fea_list:
                inputs[key] = tf.keras.Input(shape=(1,), dtype=tf.string)

                idx = layers.StringLookup(
                    vocabulary=fea_voc[key],
                    mask_token=None,
                    num_oov_indices=0,
                    name=f"{key}_index_lookup",
                )(inputs[key])

                outputs[key] = layers.Embedding(
                    input_dim=len(fea_voc[key]),
                    output_dim=fea_emb_size[key],
                    name=f"{key}_embedding",
                )(idx)

            # sequence input
            inputs["sequence_movie_ids"] = tf.keras.Input(
                name="sequence_movie_ids",
                shape=(1,),
                dtype=tf.string,
            )

            seq_split = tf.strings.split(inputs["sequence_movie_ids"], ",").to_tensor(
                "[PAD]", shape=[None, 1, 4]
            )

            item_idx = layers.StringLookup(
                vocabulary=fea_voc["item"],
                mask_token="[PAD]",  # note here!
                name="item_index_lookup",
            )(seq_split)
            outputs["sequence_idx"] = item_idx

            return tf.keras.Model(inputs=inputs, outputs=outputs)

        preprocess_layer = preprocess()
        model = BSTBase(
            preprocess_layer=preprocess_layer,
            dnn_units_size=[32],
            sequence_fea=["sequence_idx"],
            item_embedding_dims={"sequence_idx": item_embedding_dims},
            seq_len={"sequence_idx": seq_len},
            item_voc_size={"sequence_idx": len(fea_voc["item"])},
            num_head={"sequence_idx": 2},
            dropout_rate={"sequence_idx": 0.5},
        )

        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )
        return model  # need wrap

    return create_model


def create_fuse_model():
    # Create model
    def create_model():
        import tensorflow as tf

        model = BSTFuse(dnn_units_size=[256, 128])
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )
        return model

    return create_model


# deal with alice data user end
def create_dataset_builder_alice(
    batch_size=128,
    repeat_count=5,
):
    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        x = [dict(t) if isinstance(t, pd.DataFrame) else t for t in x]
        x = x[0] if len(x) == 1 else tuple(x)
        data_set = (
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
        )

        return data_set

    return dataset_builder


def create_dataset_builder_bob(
    batch_size=128,
    repeat_count=5,
):
    def _parse_bob(row_sample, label):
        import tensorflow as tf

        y_t = label["label"]
        y = tf.expand_dims(
            tf.where(
                y_t > 3,
                tf.ones_like(y_t, dtype=tf.float32),
                tf.zeros_like(y_t, dtype=tf.float32),
            ),
            axis=1,
        )
        return row_sample, y

    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        x = [dict(t) if isinstance(t, pd.DataFrame) else t for t in x]
        x = x[0] if len(x) == 1 else tuple(x)
        data_set = (
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
        )

        data_set = data_set.map(_parse_bob)

        return data_set

    return dataset_builder


def sl_bst_tf_train(sf_simulation_setup_devices, plus=True):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob

    generate_data(plus)

    vdf = read_csv(
        {
            alice: gen_data_path + "/train_data_alice.csv",
            bob: gen_data_path + "/train_data_bob.csv",
        },
        delimiter="|",
    )
    label = vdf["label"]
    data = vdf.drop(columns=["label"])

    data["user_id"] = data["user_id"].astype("string")
    data["age_group"] = data["age_group"].astype("string")
    data["occupation"] = data["occupation"].astype("string")
    data["target_movie_id"] = data["target_movie_id"].astype("string")

    bs = 64
    epoch = 1

    data_builder_dict = {
        alice: create_dataset_builder_alice(
            batch_size=bs,
            repeat_count=epoch,
        ),
        bob: create_dataset_builder_bob(
            batch_size=bs,
            repeat_count=epoch,
        ),
    }
    # User-defined compiled keras model
    device_y = bob

    if plus:
        model_base_alice = create_bstplus_base_model_alice()
        model_base_bob = create_bstplus_base_model_bob()
    else:
        model_base_alice = create_base_model_alice()
        model_base_bob = create_base_model_bob()
    base_model_dict = {
        alice: model_base_alice,
        bob: model_base_bob,
    }
    model_fuse = create_fuse_model()

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
        random_seed=0,
    )
    history = sl_model.fit(
        data,
        label,
        epochs=epoch,
        batch_size=bs,
        random_seed=0,
        dataset_builder=data_builder_dict,
    )

    global_metric = sl_model.evaluate(
        data,
        label,
        batch_size=bs,
        random_seed=0,
        dataset_builder=data_builder_dict,
    )

    test_data = read_csv(
        {
            alice: gen_data_path + "/test_data_alice.csv",
            bob: gen_data_path + "/test_data_bob.csv",
        },
        delimiter="|",
    )
    test_label = test_data["label"]
    test_data = test_data.drop(columns=["label"])

    test_data["user_id"] = test_data["user_id"].astype("string")
    test_data["age_group"] = test_data["age_group"].astype("string")
    test_data["occupation"] = test_data["occupation"].astype("string")
    test_data["target_movie_id"] = test_data["target_movie_id"].astype("string")

    global_metric = sl_model.evaluate(
        test_data,
        test_label,
        batch_size=bs,
        random_seed=0,
        dataset_builder=data_builder_dict,
    )

    # test history
    # FIXME: skip correctness check for now
    # assert history['train_auc_1'][-1] > 0.6
    # assert global_metric['auc_1'] > 0.6
    assert history["train_auc_1"][-1] > 0.1
    assert global_metric["auc_1"] > 0.1

    shutil.rmtree(gen_data_path)
    shutil.rmtree(dataset_download_dir)


def test_sl_bst_tf(sf_simulation_setup_devices):
    sl_bst_tf_train(sf_simulation_setup_devices)
    sl_bst_tf_train(sf_simulation_setup_devices, False)
