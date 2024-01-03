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

from typing import List, Optional, Tuple, Union

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.data.split import train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_deep_fm import DeepFMbase, DeepFMfuse
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.utils.simulation.datasets import load_ml_1m

NUM_USERS = 6040
NUM_MOVIES = 3952
GENDER_VOCAB = ["F", "M"]
AGE_VOCAB = [1, 18, 25, 35, 45, 50, 56]
OCCUPATION_VOCAB = [i for i in range(21)]
GENRES_VOCAB = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


class MovielensDeepfm(TrainBase):
    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        data_builder_dict = {
            self.alice: self.create_dataset_builder_alice(
                batch_size=128,
                repeat_count=5,
            ),
            self.bob: self.create_dataset_builder_bob(
                batch_size=128,
                repeat_count=5,
            ),
        }
        # User-defined compiled keras model
        device_y = self.bob
        model_base_alice = self.alice_base_model
        model_base_bob = self.bob_base_model
        base_model_dict = {
            self.alice: model_base_alice,
            self.bob: model_base_bob,
        }
        model_fuse = self.fuse_model

        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=device_y,
            model_fuse=model_fuse,
        )
        history = sl_model.fit(
            self.train_data,
            self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=5,
            batch_size=128,
            random_seed=1234,
            dataset_builder=data_builder_dict,
        )
        print(history)

    def _prepare_data(self) -> Tuple[VDataFrame, VDataFrame, VDataFrame, VDataFrame]:
        vdf = load_ml_1m(
            part={
                self.alice: [
                    "UserID",
                    "Gender",
                    "Age",
                    "Occupation",
                    "Zip-code",  # drop
                ],
                self.bob: [
                    "MovieID",
                    "Rating",  # label
                    "Title",  # drop
                    "Genres",
                    "Timestamp",  # drop
                ],
            },
            num_sample=10000,
        )
        label = vdf['Rating']
        data = vdf.drop(columns=["Rating", "Timestamp", "Title", "Zip-code"])
        data["UserID"] = data["UserID"].astype("string")
        data["MovieID"] = data["MovieID"].astype("string")
        random_state = 1234
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )
        return train_data, train_label, test_data, test_label

    def _create_base_model_alice(self):
        def create_model():
            import tensorflow as tf

            def preprocess():
                inputs = {
                    "UserID": tf.keras.Input(shape=(1,), dtype=tf.string),
                    "Gender": tf.keras.Input(shape=(1,), dtype=tf.string),
                    "Age": tf.keras.Input(shape=(1,), dtype=tf.int64),
                    "Occupation": tf.keras.Input(shape=(1,), dtype=tf.int64),
                }
                user_id_output = tf.keras.layers.Hashing(
                    num_bins=NUM_USERS, output_mode="one_hot"
                )
                user_gender_output = tf.keras.layers.StringLookup(
                    vocabulary=GENDER_VOCAB, output_mode="one_hot"
                )

                user_age_out = tf.keras.layers.IntegerLookup(
                    vocabulary=AGE_VOCAB, output_mode="one_hot"
                )
                user_occupation_out = tf.keras.layers.IntegerLookup(
                    vocabulary=OCCUPATION_VOCAB, output_mode="one_hot"
                )

                outputs = {
                    "UserID": user_id_output(inputs["UserID"]),
                    "Gender": user_gender_output(inputs["Gender"]),
                    "Age": user_age_out(inputs["Age"]),
                    "Occupation": user_occupation_out(inputs["Occupation"]),
                }
                return tf.keras.Model(inputs=inputs, outputs=outputs)

            preprocess_layer = preprocess()
            model = DeepFMbase(
                dnn_units_size=[256, 32],
                preprocess_layer=preprocess_layer,
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

    def _create_base_model_bob(self):
        def create_model():
            import tensorflow as tf

            # define preprocess layer
            def preprocess():
                inputs = {
                    "MovieID": tf.keras.Input(shape=(1,), dtype=tf.string),
                    "Genres": tf.keras.Input(shape=(1,), dtype=tf.string),
                }

                movie_id_out = tf.keras.layers.Hashing(
                    num_bins=NUM_MOVIES, output_mode="one_hot"
                )
                movie_genres_out = tf.keras.layers.TextVectorization(
                    output_mode='multi_hot', split="whitespace", vocabulary=GENRES_VOCAB
                )
                outputs = {
                    "MovieID": movie_id_out(inputs["MovieID"]),
                    "Genres": movie_genres_out(inputs["Genres"]),
                }
                return tf.keras.Model(inputs=inputs, outputs=outputs)

            preprocess_layer = preprocess()

            model = DeepFMbase(
                dnn_units_size=[256, 32],
                preprocess_layer=preprocess_layer,
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

    def _create_fuse_model(self):
        def create_model():
            import tensorflow as tf

            model = DeepFMfuse(dnn_units_size=[256, 256, 32])
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

    @staticmethod
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
                tf.data.Dataset.from_tensor_slices(x)
                .batch(batch_size)
                .repeat(repeat_count)
            )

            return data_set

        return dataset_builder

    @staticmethod
    def create_dataset_builder_bob(
        batch_size=128,
        repeat_count=5,
    ):
        def _parse_bob(row_sample, label):
            import tensorflow as tf

            y_t = label["Rating"]
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
                tf.data.Dataset.from_tensor_slices(x)
                .batch(batch_size)
                .repeat(repeat_count)
            )

            data_set = data_set.map(_parse_bob)

            return data_set

        return dataset_builder


def run(config, *, alice, bob, callbacks=None):
    ml_deepfm = MovielensDeepfm(config, alice, bob)
    train_data, train_label, test_data, test_label = ml_deepfm._prepare_data()
    data_builder_dict = {
        alice: ml_deepfm.create_dataset_builder_alice(
            batch_size=128,
            repeat_count=5,
        ),
        bob: ml_deepfm.create_dataset_builder_bob(
            batch_size=128,
            repeat_count=5,
        ),
    }
    # User-defined compiled keras model
    device_y = bob
    model_base_alice = ml_deepfm._create_base_model_alice()
    model_base_bob = ml_deepfm._create_base_model_bob()
    base_model_dict = {
        alice: model_base_alice,
        bob: model_base_bob,
    }
    model_fuse = ml_deepfm._create_fuse_model()

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
    )
    history = sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=5,
        batch_size=128,
        random_seed=1234,
        dataset_builder=data_builder_dict,
    )
    print(history)


def movie_lens_train_tf(alice, bob):
    ml_deepfm = MovielensDeepfm({}, alice, bob, bob)
    ml_deepfm.train()
