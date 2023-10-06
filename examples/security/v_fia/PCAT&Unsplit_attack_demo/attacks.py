import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".secretflow/datasets")
attack_model_path = "./model/alice/"
client_model_path = "./model/bob/"
decoder_model_path = "./model/"
data_path = f"{_CACHE_DIR}/" + "mnist.npz"
batch_size = 64
iterations = 10000
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
getImagesDS = lambda X, n: np.concatenate([x[0].numpy()[None,] for x in X.take(n)])
SIZE = 32


class Clone_model(tf.keras.Model):
    def __init__(self, n_channels=1):
        super(Clone_model, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            padding="same",
            input_shape=(28, 28, n_channels),
        )

        self.ReLU1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.conv1(x)
        x = self.ReLU1(x)
        # x = self.pool1(x)
        return x


def parse(x):
    x = x[:, :, :, None]
    x = tf.tile(x, (1, 1, 1, 3))
    x = tf.image.resize(x, (SIZE, SIZE))
    x = x / (255 / 2) - 1
    x = tf.clip_by_value(x, -1.0, 1.0)
    return x


def TV(x):
    batch_size = tf.cast(tf.shape(x)[0], tf.float32)
    h_x = tf.shape(x)[2]
    w_x = tf.shape(x)[3]
    count_h = tf.cast(tf.reduce_prod(tf.shape(x[:, :, 1:, :])), tf.float32)
    count_w = tf.cast(tf.reduce_prod(tf.shape(x[:, :, :, 1:])), tf.float32)
    h_tv = tf.reduce_sum(tf.square(x[:, :, 1:, :] - x[:, :, : h_x - 1, :]))
    w_tv = tf.reduce_sum(tf.square(x[:, :, :, 1:] - x[:, :, :, : w_x - 1]))
    # 处理除以零的情况
    count_h = tf.maximum(count_h, 1e-12)
    count_w = tf.maximum(count_w, 1e-12)
    # 处理无效值的情况
    h_tv = tf.where(tf.math.is_finite(h_tv), h_tv, tf.zeros_like(h_tv))
    w_tv = tf.where(tf.math.is_finite(w_tv), w_tv, tf.zeros_like(w_tv))

    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return tf.reduce_sum(x**2)


def get_examples_by_class(images, labels, class_id, count=1):
    indices = np.where(labels == class_id)[0][:count]
    return images[indices]


def distance_data_loss(a, b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)


def plot(X, save_path=None):
    n = len(X)
    X = (X + 1) / 2
    fig, ax = plt.subplots(1, n, figsize=(n * 3, 3))
    plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=-0.05)
    for i in range(n):
        ax[i].imshow((X[i]), cmap="inferno")
        ax[i].set(xticks=[], yticks=[])
        ax[i].set_aspect("equal")
    plt.savefig(save_path)
    return fig


def attack(attack_model, decoder, data, base_model_name):
    clent_model = tf.keras.models.load_model(client_model_path + base_model_name)

    smashed_data = attack_model(data, training=False)
    smashed_data_client = clent_model(data, training=False)

    recover_data = decoder(smashed_data, training=False)
    recover_data_client = decoder(smashed_data_client, training=False)

    return recover_data.numpy(), recover_data_client.numpy()


def model_inversion_stealing(
    clone_model,
    target,
    origin,
    input_size,
    lambda_tv=0.1,
    lambda_l2=1,
    main_iters=1000,
    input_iters=100,
    model_iters=100,
):
    x_pred = tf.Variable(tf.fill(input_size, 0.5), trainable=True)
    # x_pred = np.expand_dims(x_pred, axis=3)
    input_opt = tf.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    model_opt = tf.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    mse = tf.losses.MeanSquaredError()

    for main_iter in range(main_iters):
        for input_iter in range(input_iters):
            with tf.GradientTape() as input_tape:
                pred = clone_model(x_pred)
                loss = (
                    mse(pred, target)
                    + lambda_tv * TV(x_pred)
                    + lambda_l2 * l2loss(x_pred)
                )
                # if input_iter % 100 == 0:
                #     tf.print("loss ", loss)
            input_grads = input_tape.gradient(loss, x_pred)
            input_opt.apply_gradients([(input_grads, x_pred)])

        for model_iter in range(model_iters):
            with tf.GradientTape() as model_tape:
                pred = clone_model(x_pred)
                loss = mse(pred, target)
                # if model_iter % 100 == 0:
                #     tf.print("loss2 ", loss)
            model_grads = model_tape.gradient(loss, clone_model.trainable_variables)
            model_opt.apply_gradients(zip(model_grads, clone_model.trainable_variables))

        if main_iter % 100 == 0:
            image = x_pred.numpy()
            plt.imshow(image[0], cmap="gray")
            plt.show()

    return x_pred


def decoder(input_shape, channels):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    model.add(
        tf.keras.layers.Conv2DTranspose(256, 3, 2, padding="same", activation="relu")
    )
    model.add(
        tf.keras.layers.Conv2DTranspose(128, 3, 2, padding="same", activation="relu")
    )
    model.add(tf.keras.layers.Conv2D(channels, 3, 1, padding="same", activation="tanh"))
    model.summary()
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def get_dataset():
    with np.load(data_path) as f:
        x_test, y_test = f["x_test"], f["y_test"]
    x_test = x_test.astype(np.float32)

    x_test = np.array(parse(x_test))

    encoder = OneHotEncoder(sparse=False)
    y_test = encoder.fit_transform(y_test.reshape(-1, 1))

    x = tf.data.Dataset.from_tensor_slices(x_test)
    y = tf.data.Dataset.from_tensor_slices(y_test)

    xy = tf.data.Dataset.zip((x, y))
    final_show_data = getImagesDS(xy, 20)

    data_train = xy.batch(batch_size, drop_remainder=True).repeat(-1)
    input_shape = data_train.element_spec[0].shape
    iterator = data_train.take(iterations)

    return input_shape, final_show_data, iterator


def attack_WithTrain(base_model_name, decoder_model_name):
    shape, final_show_data, data = get_dataset()

    client_model = tf.keras.models.load_model(client_model_path + base_model_name)
    attack_model = tf.keras.models.load_model(attack_model_path + base_model_name)
    decoder_model = tf.keras.models.load_model(decoder_model_path + decoder_model_name)

    smashed_data_attack = attack_model(final_show_data, training=False)
    smashed_data_client = client_model(final_show_data, training=False)

    recover_data_attack = decoder_model(smashed_data_attack, training=False)
    recover_data_server = decoder_model(smashed_data_client, training=False)

    fig1 = plot(final_show_data, save_path="result/aw_original.png")
    fig2 = plot(
        recover_data_attack.numpy(), save_path="result/aw_recover_data_attack.png"
    )
    fig3 = plot(
        recover_data_server.numpy(), save_path="result/aw_recover_data_server.png"
    )

    fig1.show()
    fig2.show()
    fig3.show()


def attack_AfterTrain(base_model_name):
    shape, final_show_data, iterator = get_dataset()
    attack_model = tf.keras.models.load_model(attack_model_path + base_model_name)

    z_shape = attack_model.output.shape.as_list()[1:]
    decoder_model = decoder(z_shape, 3)

    for data, label in iterator:
        with tf.GradientTape(persistent=True) as tape:
            smash_data = attack_model(data, training=True)

            rec_smash_data = decoder_model(smash_data, training=True)

            decoder_loss = distance_data_loss(data, rec_smash_data)

        var = decoder_model.trainable_variables
        gradients = tape.gradient(decoder_loss, var)
        optimizer.apply_gradients(zip(gradients, var))

    decoder_model.save("decoder_pcat_" + base_model_name + ".h5")

    final_recover_data, final_client_recover_data = attack(
        attack_model, decoder_model, final_show_data, base_model_name
    )

    fig1 = plot(final_show_data, save_path="result/at_original.png")
    fig2 = plot(final_recover_data, save_path="result/at_recover_data_attack.png")
    fig3 = plot(
        final_client_recover_data, save_path="result/at_recover_data_server.png"
    )

    fig1.show()
    fig2.show()
    fig3.show()


def unsplit_attack(base_model_name):
    client = tf.keras.models.load_model(client_model_path + base_model_name)
    clone = Clone_model()

    with np.load(data_path) as f:
        x_test, y_test = f["x_test"], f["y_test"]

    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # inversion_targets = [get_examples_by_class(x_test, y_test, c, count=1) for c in range(10)]
    inversion_targets = [
        get_examples_by_class(x_test, y_test, c, count=1) for c in range(1, 6)
    ]
    shapes = [target.shape for target in inversion_targets]
    print(shapes)
    dataset = tf.data.Dataset.from_tensor_slices(inversion_targets)
    # perform inversion and stealing attack
    results, losses = [], []
    for idx, target in enumerate(inversion_targets):
        client_out = client(target)
        result = model_inversion_stealing(
            clone,
            client_out,
            target,
            target.shape,
            main_iters=1000,
            input_iters=100,
            model_iters=100,
        )
        loss = tf.reduce_mean(tf.keras.losses.MSE(result, target))
        losses.append(loss)

        print(f"\tImage {idx} loss: {loss}")
        image = result.numpy()
        plt.imshow(image[0][0], cmap="gray")
        plt.show()

    print(f"Average MSE: {sum(losses) / len(losses)}")
