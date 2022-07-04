"""local training for bank marketing dataset and save the split sub-datasets

The following codes are demos only. 
It's **NOT for production** due to system security concerns,
please **DO NOT** use it directly in production.
"""

import os
from random import seed

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from .funcs import (
    create_bank_model,
    create_partitioned_dataset,
    get_split_dimension,
    preprocess,
)
from secretflow.data.io.util import open

flags.DEFINE_float("lr", 0.0005, "learning rate")
flags.DEFINE_integer("batch_size", 128, "batch size of training")
flags.DEFINE_integer("num_clients", 5, "number of clients to partition")
flags.DEFINE_string("filename", "./data/bank/bank_{}_{}_of_{}.csv", "filename template")
flags.DEFINE_bool("save_file", True, "save the partitioned datasets")
flags.DEFINE_integer('seed_number', 1, 'seed number for random')
flags.DEFINE_enum(
    'save_type', 'csv', ['np', 'csv'], 'save type of partitioned datasets'
)
FLAGS = flags.FLAGS


def get_filename(tag, i):
    return FLAGS.filename.format(tag, i, FLAGS.num_clients)


def main(_):
    os.environ['PYTHONHASHSEED'] = str(1)
    seed(FLAGS.seed_number)
    tf.random.set_seed(FLAGS.seed_number)
    np.random.seed(FLAGS.seed_number)

    # datasaet
    num_classes = 1

    file_path = (
        '../../../../tests/datasets/bank/bank.csv'
    )
    optimizer = Adam(learning_rate=FLAGS.lr)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', verbose=1, patience=30, restore_best_weights=True
    )

    dataset = pd.read_csv(open(file_path), delimiter=',')
    encoder = preprocessing.LabelEncoder()
    columns = [
        'job',
        'marital',
        'education',
        'default',
        'housing',
        'loan',
        'contact',
        'poutcome',
        'month',
        'deposit',
    ]
    dataset = preprocess(dataset, encoder, columns)
    dataset_array = dataset.to_numpy()

    features = dataset_array[:, :-1]
    label = dataset_array[:, -1]

    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)

    new_data = np.concatenate(
        (features, np.reshape(label, (label.shape[0], 1))), axis=1
    )
    dataset[:] = new_data

    train_X, test_X, train_y, test_y = train_test_split(
        features, label, test_size=0.2, random_state=FLAGS.seed_number
    )

    if FLAGS.num_clients == 0:
        train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        train_ds = (
            train_ds.shuffle(1024)
            .batch(FLAGS.batch_size)
            .repeat(1)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        valid_ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        valid_ds = (
            valid_ds.batch(FLAGS.batch_size)
            .repeat(1)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        test_ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        test_batch = test_y.shape[0]
        test_ds = (
            test_ds.batch(test_batch).repeat(1).prefetch(tf.data.experimental.AUTOTUNE)
        )

        model = create_bank_model(train_X.shape[1], num_classes, optimizer)

        model.fit(
            train_ds, validation_data=valid_ds, epochs=20, callbacks=[early_stopping]
        )
        model.evaluate(test_ds)

    else:
        X_train_partitions = create_partitioned_dataset(train_X, FLAGS.num_clients)
        X_test_partitions = create_partitioned_dataset(test_X, FLAGS.num_clients)

        local_metrics = []
        for i in range(FLAGS.num_clients):
            train_ds = tf.data.Dataset.from_tensor_slices(
                (X_train_partitions[i], train_y)
            )
            train_ds = (
                train_ds.shuffle(1024)
                .batch(FLAGS.batch_size)
                .repeat(1)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

            test_ds = tf.data.Dataset.from_tensor_slices((X_test_partitions[i], test_y))
            test_ds = (
                test_ds.batch(FLAGS.batch_size)
                .repeat(1)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

            model = create_bank_model(
                X_train_partitions[i].shape[1], num_classes, optimizer
            )

            model.fit(
                train_ds, validation_data=test_ds, epochs=20, callbacks=[early_stopping]
            )
            test_local_metric = model.evaluate(test_ds, return_dict=True)

            local_metrics.append(test_local_metric['auc'])

            tf.print(
                "local test auc of client {} is {}".format(i, test_local_metric['auc'])
            )

        average_local_metrics = np.average(local_metrics)

        tf.print("local average test auc is ", average_local_metrics)

        if FLAGS.save_file:
            if FLAGS.save_type == 'np':
                for i, feature in enumerate(X_train_partitions):
                    filename = get_filename('train_x', i)
                    np.save(filename, feature)
                filename = get_filename('train_y', 0)
                np.save(filename, train_y)

                for i, feature in enumerate(X_test_partitions):
                    filename = get_filename('test_x', i)
                    np.save(filename, feature)
                filename = get_filename('test_y', 0)
                np.save(filename, test_y)
            else:
                dim = get_split_dimension(dataset.shape[1], FLAGS.num_clients)
                assert len(dim) == FLAGS.num_clients + 1
                for i in range(FLAGS.num_clients):
                    start = dim[i]
                    end = dim[i + 1]
                    part_dataset = dataset.iloc[:, start:end]
                    filename = get_filename('data', i)
                    part_dataset.to_csv(filename, index=False)


if __name__ == "__main__":
    app.run(main)
