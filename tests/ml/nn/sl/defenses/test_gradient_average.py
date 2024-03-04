import numpy as np

from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.preprocessing import StandardScaler
from secretflow.utils.simulation.datasets import load_creditcard
from secretflow.ml.nn.sl.defenses.gradient_average import GradientAverage


def test_gradient_average(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    data = load_creditcard({alice: (0, 29)})
    label = load_creditcard({bob: (29, 30)}).astype(np.float32)
    scaler = StandardScaler()
    data = scaler.fit_transform(data).astype(np.float32)
    random_state = 1234
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.8, random_state=random_state
    )
    hidden_dim_1 = 16
    hidden_dim_2 = 4

    train_data = test_data
    train_label = test_label

    def create_base_net(input_dim, hidden_dim, name="first_net"):
        # Create model
        def create_model():
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            model = keras.Sequential(
                [
                    keras.Input(shape=input_dim),
                    layers.Dense(hidden_dim // 2, activation="relu"),
                    layers.Dense(hidden_dim, activation="relu"),
                ],
                name=name,
            )
            # Compile model
            model.summary()
            optimizer = tf.keras.optimizers.Adam()
            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy", tf.keras.metrics.AUC()],
            )
            return model

        return create_model

    def create_fuse_model(input_dim_1, output_dim, party_nums, name="fuse_model"):
        def create_model():
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            # input
            input_layers = keras.Input(
                input_dim_1,
            )
            output = layers.Dense(output_dim, activation="sigmoid")(input_layers)

            model = keras.Model(
                inputs=input_layers,
                outputs=output,
                name=name,
            )
            model.summary()
            optimizer = tf.keras.optimizers.Adam()

            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy", tf.keras.metrics.AUC()],
            )
            return model

        return create_model

    base_model_dict = {
        alice: create_base_net(input_dim=29, hidden_dim=hidden_dim_1),
    }
    fuse_model = create_fuse_model(input_dim_1=hidden_dim_1, party_nums=2, output_dim=1)

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=bob,
        model_fuse=fuse_model,
        simulation=True,
        random_seed=1234,
        strategy="split_nn",
        callback=[GradientAverage],
    )

    history = sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=1,
        batch_size=256,
        shuffle=False,
        random_seed=1234,
    )

    assert history["val_accuracy"][-1] > 0.8
