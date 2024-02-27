#!/usr/bin/env python
# coding=utf-8
import sys

sys.path.append("..")

import pdb

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as F
import torch
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

eps = 1.0e-35

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_visible_devices(gpus[1], "GPU")
        tf.config.experimental.set_virtual_device_configuration(
            gpus[1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)],
        )
    except RuntimeError as e:
        print(e)


def check_nan(values):
    check_results = tf.math.is_nan(values)
    return len(tf.where(check_results == True)) > 0


def sharpen_np(probs, T):
    probs_np = probs.numpy()
    if len(probs_np.shape) == 1:
        temp = np.power(probs_np, 1.0 / T)
        temp = temp / (np.power(1.0 - temp, 1.0 / T) + temp)
    else:
        temp = np.power(probs_np, 1.0 / T)
        temp_sum = np.sum(temp, axis=1, keepdims=True)
        temp = temp / temp_sum
    return tf.convert_to_tensor(temp)


def sharpen(probs, T):
    if len(probs.shape) == 1:
        temp = tf.pow(probs, 1.0 / T)
        temp = temp / (tf.pow(1.0 - temp, 1.0 / T) + temp)
    else:
        temp = tf.pow(probs, 1.0 / T)
        temp_sum = tf.reduce_sum(temp, axis=1, keepdims=True)
        temp = temp / temp_sum
    return temp


class CoAE_Loss:
    def __init__(self, ew=0.1, pw=10.0, nw=1.0):
        self.ew = ew
        self.pw = pw
        self.nw = nw

    def log2(self, x):
        return F.log(x) / F.log(2.0)

    def entropy(self, y_latent):
        loss = -y_latent * F.log(y_latent + eps)
        return F.mean(loss)

    def cross_entropy(self, y_true, y_hat):
        loss = F.sum(-y_true * F.log(y_hat + eps), axis=1)
        return F.mean(loss)

    def cross_entropy_for_onehot(self, y_true, y_hat):
        loss = F.sum(-y_true * F.log(F.softmax(y_hat, axis=1)), axis=1)
        return F.mean(loss)

    def __call__(self, y_true, y_latent, y_hat):
        eloss = self.entropy(y_latent)
        ploss = self.cross_entropy_for_onehot(y_true, y_hat)
        nloss = self.cross_entropy_for_onehot(y_true, y_latent)

        loss = self.pw * ploss - self.ew * eloss - self.nw * nloss
        return loss


class AutoEncoder(keras.Model):
    def __init__(self, input_dim, latent_dim, T=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.T = T

        self.encoder = Sequential(
            [
                layers.Dense(latent_dim**2, activation="relu"),
                layers.Dense(input_dim, activation="softmax"),
            ]
        )

        self.decoder = Sequential(
            [
                layers.Dense(latent_dim**2, activation="relu"),
                layers.Dense(input_dim, activation="softmax"),
            ]
        )

    def call(self, y_true):
        y_latent = self.encoder(y_true)
        y_hat = self.decoder(y_latent)
        return y_latent, y_hat


if __name__ == "__main__":
    # num_classes = 10
    for num_classes in [5, 10]:
        for ew in [0.01, 0.03, 0.05, 0.07, 0.10]:
            loss_fn = CoAE_Loss(pw=1.0, ew=ew, nw=0.01)
            model = AutoEncoder(num_classes, 6 * num_classes + 2)

            optimizer = tf.keras.optimizers.get(
                {"class_name": "adam", "config": {"learning_rate": 5.0e-4}}
            )

            epochs = 50
            batch_size = 256
            train_size, test_size = 60000, 10000
            T = 0.025

            y_rand = tf.random.uniform([train_size, num_classes])
            y_softmax = F.softmax(y_rand, axis=1)
            y_train = sharpen_np(y_softmax, T=T)

            y_rand = tf.random.uniform([test_size, num_classes])
            y_softmax = F.softmax(y_rand, axis=1)
            y_test = sharpen_np(y_softmax, T=T)
            y_labels = tf.argmax(y_test, axis=1)

            y_onehot = F.one_hot(
                indices=np.random.randint(num_classes, size=(test_size,)),
                num_classes=num_classes,
            )
            y_labels_onehot = tf.argmax(y_onehot, axis=1)

            def train_on_batch(y_true):
                with tf.GradientTape() as tape:
                    y_latent, y_hat = model(y_true)
                    loss = loss_fn(y_true, y_latent, y_hat)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss, y_latent, y_hat

            dataset = (
                tf.data.Dataset.from_tensor_slices(y_train)
                .shuffle(buffer_size=train_size)
                .batch(batch_size)
            )
            for i in range(epochs):
                total_loss, batch_count = 0, 0
                for y_true in dataset:
                    loss, y_latent, y_hat = train_on_batch(y_true)
                    total_loss = total_loss + loss
                    batch_count += 1

                if (i + 1) % 10 == 0:
                    y_latent, y_hat = model(y_test)
                    y_latent_onehot, y_hat_onehot = model(y_onehot)
                    acc_p = len(
                        tf.where((F.argmax(y_hat, axis=1) == y_labels) == True)
                    ) / len(y_labels)
                    acc_n = len(
                        tf.where((F.argmax(y_latent, axis=1) != y_labels) == True)
                    ) / len(y_labels)
                    acc_p_onehot = len(
                        tf.where(
                            (F.argmax(y_hat_onehot, axis=1) == y_labels_onehot) == True
                        )
                    ) / len(y_labels_onehot)
                    acc_n_onehot = len(
                        tf.where(
                            (F.argmax(y_latent_onehot, axis=1) != y_labels_onehot)
                            == True
                        )
                    ) / len(y_labels_onehot)
                    cur_loss = total_loss / batch_count
                    print(
                        "epoch",
                        i + 1,
                        "acc_p:",
                        acc_p,
                        "acc_n:",
                        acc_n,
                        "acc_p_onehot:",
                        acc_p_onehot,
                        "acc_n_onehot:",
                        acc_n_onehot,
                        "loss:",
                        cur_loss.numpy(),
                    )

            model_path = "../../trained_model/{}-{:.3f}.ckpt".format(num_classes, ew)
            model.save_weights(model_path)
        print("end")
