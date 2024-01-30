MODELS_CODE = """
# pre imported:
# import tensorflow as tf
# from tensorflow import Module, keras
# from tensorflow.keras import Model, layers
# from tensorflow.keras.layers import Layer
# from secretflow.ml.nn import applications as apps


def create_base_model(input_dim, output_dim):
    # Create model
    model = keras.Sequential(
        [
            keras.Input(shape=input_dim),
            layers.Dense(100, activation="relu"),
            layers.Dense(output_dim, activation="relu"),
        ]
    )
    return model

def create_fuse_model(input_dim):
    input_layers = [
        keras.Input(input_dim) for _ in range(2)
    ]
    merged_layer = layers.concatenate(input_layers)
    fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
    output = layers.Dense(2, activation='softmax')(fuse_layer)
    output = output[:, 1:]

    model = keras.Model(inputs=input_layers, outputs=output)
    return model

hidden_size = 64

fit(
    client_base=create_base_model(12, hidden_size),
    server_base=create_base_model(4, hidden_size),
    server_fuse=create_fuse_model(hidden_size),
)
"""

SUBCLASS_MULTIINPUT_MODEL_CODE = """
class SubClassBaseModel(Model):
    def __init__(self, output_dim):
        super().__init__()
        self.dense_internal = keras.Sequential(
            [
                layers.Dense(100, activation="relu"),
                layers.Dense(output_dim, activation="relu"),
            ]
        )

    def call(self, x, training=None):
        inputs = [tf.expand_dims(f, axis=1) for _, f in x.items()]
        merged_inputs = layers.concatenate(inputs)
        return self.dense_internal(merged_inputs, training=training)


def create_base_model(mock_input, output_dim):
    # Create model
    model = SubClassBaseModel(output_dim)
    # Create forward pass
    model(mock_input)
    return model


def create_fuse_model(input_dim):
    input_layers = [keras.Input(input_dim) for _ in range(2)]
    merged_layer = layers.concatenate(input_layers)
    fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
    output = layers.Dense(2, activation='softmax')(fuse_layer)
    output = output[:, 1:]

    model = Model(inputs=input_layers, outputs=output)
    return model


hidden_size = 64


def create_mock_input(input_names):
    return {name: tf.constant([0.0]) for name in input_names}


client_features = [
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


server_features = ["age", "job", "marital", "education"]

fit(
    client_base=create_base_model(
        create_mock_input(client_features),
        hidden_size,
    ),
    server_base=create_base_model(
        create_mock_input(server_features),
        hidden_size,
    ),
    server_fuse=create_fuse_model(hidden_size),
)
"""


LOSS_FUNC_CODE = """
def loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

compile_loss(loss)
"""

LOSS_CLASS_CODE = """
class CustomLoss(Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ]
    )
    def __call__(self, y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)


compile_loss(CustomLoss())
"""
