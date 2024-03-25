
import matplotlib.pyplot as plt
import secretflow as sf
from tensorflow import keras
from tensorflow.keras import layers
from secretflow.security.aggregation import SPUAggregator, SecureAggregator
from secretflow.ml.nn import FLModel
from secretflow.data.ndarray import load
from secretflow.utils.simulation.datasets import load_mnist


# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob', 'charlie'], address='local')
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

(x_train, y_train), (x_test, y_test) = load_mnist(
    parts=[alice, bob], normalized_x=True, categorical_y=True
)

import numpy as np
from secretflow.utils.simulation.datasets import dataset

mnist = np.load(dataset('mnist'), allow_pickle=True)
image = mnist['x_train']
label = mnist['y_train']


def create_conv_model(input_shape, num_classes, name='model'):
    def create_model():

        # Create model
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        # Compile model
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model

    return create_model



num_classes = 10
input_shape = (28, 28, 1)
model = create_conv_model(input_shape, num_classes)

device_list = [alice, bob]
secure_aggregator = SecureAggregator(charlie,[alice, bob])
spu_aggregator = SPUAggregator(spu)
fed_model = FLModel(
    server=charlie,
    device_list=device_list,
    model=model,
    aggregator=secure_aggregator,
    strategy="fed_svd_agg",
    backend="tensorflow",
)

history = fed_model.fit(
    x_train[0:100],
    y_train[0:100],
    validation_data=(x_test, y_test),
    epochs=1,
    sampler_method="batch",
    batch_size=32,
    aggregate_freq=1,
)
