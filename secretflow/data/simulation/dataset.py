import pandas as pd
import numpy as np
from typing import Dict
from keras.datasets import mnist
from keras.utils import np_utils
import tempfile
import os
import math
import requests
import io
import zipfile

iris_uri = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dermatology_uri = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
bank_marketing_uri = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
)
_temp_dir = tempfile.mkdtemp()


def load_mnist_data(party_ratio: Dict = None):
    if party_ratio is None:
        raise Exception("party cannot be none")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = x_train / 255, x_test / 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    fed_train_set = {}
    train_length = len(x_train)
    start_idx = 0
    for party, ratio in party_ratio.items():
        end_idx = math.ceil(train_length * ratio)
        p_train = x_train[start_idx:end_idx]
        p_label = y_train[start_idx:end_idx]
        file_name = "_".join(["mnist", party.party])
        data_path = os.path.join(_temp_dir, f"{file_name}.npz")
        np.savez(data_path, image=p_train, label=p_label)
        fed_train_set[party] = data_path
        start_idx = end_idx

    fed_test_set = {}
    test_length = len(x_test)
    for party, ratio in party_ratio.items():
        end_idx = math.ceil(test_length * ratio)
        p_test = x_test[start_idx:end_idx]
        p_label = y_test[start_idx:end_idx]
        file_name = "_".join(["mnist", party.party])
        data_path = os.path.join(_temp_dir, f"{file_name}_test.npz")
        np.savez(data_path, image=p_test, label=p_label)
        fed_test_set[party] = data_path
        start_idx = end_idx

    return fed_train_set, fed_test_set


def load_iris_data(party_ratio: Dict = None):
    response = requests.get(iris_uri)
    response.raise_for_status()
    iris_df = pd.read_csv(
        io.BytesIO(response.content),
        names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
    )
    fed_train_set = {}
    train_length = len(iris_df)
    start_idx = 0
    for party, ratio in party_ratio.items():
        end_idx = math.ceil(train_length * ratio)
        p_train = iris_df[start_idx:end_idx]
        file_name = "_".join(["iris", party.party])
        data_path = os.path.join(_temp_dir, f"{file_name}.csv")
        p_train.to_csv(data_path, index=False)
        fed_train_set[party] = data_path
        start_idx = end_idx
    return fed_train_set


def load_dermatology_data(party_ratio: Dict = None):
    response = requests.get(dermatology_uri)
    response.raise_for_status()
    columns = [f"V{no}" for no in range(1, 35)]
    columns.append("CLASS")
    dermatology_df = pd.read_csv(
        io.BytesIO(response.content),
        names=columns,
    )
    dermatology_df.replace("?", 0, inplace=True)
    dermatology_df["CLASS"] = dermatology_df["CLASS"] - 1
    fed_train_set = {}
    train_length = len(dermatology_uri)
    start_idx = 0
    for party, ratio in party_ratio.items():
        end_idx = math.ceil(train_length * ratio)
        p_train = dermatology_df[start_idx:end_idx]
        file_name = "_".join(["dermatology", party.party])
        data_path = os.path.join(_temp_dir, f"{file_name}.csv")
        p_train.to_csv(data_path, index=False)
        fed_train_set[party] = data_path
        start_idx = end_idx
    return fed_train_set


def load_bank_marketing_data(party_ratio: Dict = None):
    response = requests.get(bank_marketing_uri)
    response.raise_for_status()
    directory_to_extract_to = os.path.join(_temp_dir, "bank_marketing")
    with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    file_path = os.path.join(directory_to_extract_to, "bank.csv")
    bank_marketing_df = pd.read_csv(file_path, sep=";")
    fed_train_set = {}

    for party, column in party_ratio.items():
        p_train = bank_marketing_df[column]
        file_name = "_".join(["bank_marketing", party.party])
        data_path = os.path.join(_temp_dir, f"{file_name}.csv")
        p_train.to_csv(data_path, index=True, index_label="id")
        fed_train_set[party] = data_path
    return fed_train_set
