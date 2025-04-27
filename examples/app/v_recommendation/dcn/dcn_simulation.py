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

import json

import pandas as pd
import torch
from data.fed_data_gen import generate_alice_bob_criteo_1m_data, load_criteo_partitioned
from matplotlib import pyplot as plt
from model.sl_dcn_torch import DCNBase, DCNFuse
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, Precision

import secretflow as sf
from secretflow.data.vertical import read_csv
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.utils import TorchModel

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob'], address="local", log_to_driver=False)
alice, bob = sf.PYU('alice'), sf.PYU('bob')


alice_base_out_dim = 0
bob_base_out_dim = 0
batch_size = 2048


class AliceDataset(Dataset):
    def __init__(
        self,
        df,
        label_df,
        cat_features=None,
        num_features=None,
        categories=None,
    ):
        self.df = df
        self.label_df = label_df
        self.x_cat = (
            torch.tensor(self.df[cat_features].values, dtype=torch.long)
            if cat_features
            else None
        )
        self.x_num = (
            torch.tensor(self.df[num_features].values, dtype=torch.float32)
            if num_features
            else None
        )

        self.label = torch.tensor(self.label_df.values, dtype=torch.float)
        self.categories = categories

    def __getitem__(self, index):
        return ((self.x_num[index], self.x_cat[index]), self.label[index])

    def __len__(self):
        return len(self.label_df)

    def get_categories(self):
        return self.categories


class BobDataset(Dataset):
    def __init__(self, df, cat_features=None, num_features=None, categories=None):
        self.df = df
        self.x_cat = (
            torch.tensor(self.df[cat_features].values, dtype=torch.long)
            if cat_features
            else None
        )
        self.x_num = (
            torch.tensor(self.df[num_features].values, dtype=torch.float32)
            if num_features
            else None
        )
        self.categories = categories

    def __getitem__(self, index):
        return (self.x_num[index], self.x_cat[index])

    def __len__(self):
        return len(self.df)

    def get_categories(self):
        return self.categories


def gen_alice_bob_cat_num_categories():
    alice_criteo_train_1m, bob_criteo_train_1m = generate_alice_bob_criteo_1m_data()
    alice_cat_features = [x for x in alice_criteo_train_1m.columns if x.startswith('C')]
    alice_num_features = [x for x in alice_criteo_train_1m.columns if x.startswith('I')]
    alice_categories = [alice_criteo_train_1m[x].max() + 1 for x in alice_cat_features]
    bob_cat_features = [x for x in bob_criteo_train_1m.columns if x.startswith('C')]
    bob_num_features = [x for x in bob_criteo_train_1m.columns if x.startswith('I')]
    bob_categories = [bob_criteo_train_1m[x].max() + 1 for x in bob_cat_features]

    return (
        alice_cat_features,
        alice_num_features,
        alice_categories,
        bob_cat_features,
        bob_num_features,
        bob_categories,
    )


(
    alice_cat_features,
    alice_num_features,
    alice_categories,
    bob_cat_features,
    bob_num_features,
    bob_categories,
) = gen_alice_bob_cat_num_categories()


def create_dataset_builder_alice(batch_size=32):

    def dataset_builder(x):
        data_set = AliceDataset(
            x[0],
            x[1],
            alice_cat_features,
            alice_num_features,
            alice_categories,
        )
        dataloader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


def create_dataset_builder_bob(batch_size=32):

    def dataset_builder(x):
        data_set = BobDataset(x[0], bob_cat_features, bob_num_features, bob_categories)
        dataloader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


dataset_buidler_dict = {
    alice: create_dataset_builder_alice(batch_size=batch_size),
    bob: create_dataset_builder_bob(
        batch_size=batch_size,
    ),
}


# 构建模型
def create_base_model_alice():
    d_numerical = len(alice_num_features)
    d_embed_max = 8
    categories = alice_categories
    d_cat_sum = sum([min(max(int(x**0.5), 2), d_embed_max) for x in categories])
    # logger.debug(f"alice d_cat_sum: {d_cat_sum}")
    mlp_layers = [128, 64, 32]
    global alice_base_out_dim
    alice_base_out_dim = d_numerical + d_cat_sum + mlp_layers[-1]

    def create_model():
        model = DCNBase(
            d_numerical=d_numerical,
            categories=categories,
            d_cat_sum=d_cat_sum,
            d_embed_max=d_embed_max,
            n_cross=2,
            mlp_layers=mlp_layers,
            mlp_dropout=0.25,
        )

        return model

    return create_model


def create_base_model_bob():
    d_numerical = len(bob_num_features)
    d_embed_max = 8
    categories = bob_categories
    d_cat_sum = sum([min(max(int(x**0.5), 2), d_embed_max) for x in categories])
    mlp_layers = [128, 64, 32]
    global bob_base_out_dim
    bob_base_out_dim = d_numerical + d_cat_sum + mlp_layers[-1]

    def create_model():
        model = DCNBase(
            d_numerical=d_numerical,
            categories=categories,
            d_cat_sum=d_cat_sum,
            d_embed_max=d_embed_max,
            n_cross=2,
            mlp_layers=mlp_layers,
            mlp_dropout=0.25,
        )

        return model

    return create_model


def create_fuse_model():
    total_fuse_dim = alice_base_out_dim + bob_base_out_dim

    def create_model():
        model = DCNFuse(n_classes=1, total_fuse_dim=total_fuse_dim)

        return model

    return create_model


def run():
    epoch = 100
    loss_fn = nn.BCEWithLogitsLoss
    optim_fn = optim_wrapper(optim.Adam, lr=0.002, weight_decay=0.001)

    base_model_alice = TorchModel(
        model_fn=create_base_model_alice(),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
            metric_wrapper(Precision, task="binary"),
            metric_wrapper(AUROC, task="binary"),
        ],
    )

    base_model_bob = TorchModel(
        model_fn=create_base_model_bob(),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
            metric_wrapper(Precision, task="binary"),
            metric_wrapper(AUROC, task="binary"),
        ],
    )

    fuse_model = TorchModel(
        model_fn=create_fuse_model(),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
            metric_wrapper(Precision, task="binary"),
            metric_wrapper(AUROC, task="binary"),
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
        backend='torch',
    )
    # 加载数据
    vdf = load_criteo_partitioned([alice, bob], train=True)
    label = vdf["label"]
    data = vdf.drop(columns=["label"])

    val_vdf = load_criteo_partitioned([alice, bob], train=False)
    val_label = val_vdf["label"]
    val_data = val_vdf.drop(columns=["label"])

    history = sl_model.fit(
        data,
        label,
        validation_data=(val_data, val_label),
        epochs=epoch,
        batch_size=batch_size,
        shuffle=False,
        random_seed=1234,
        dataset_builder=dataset_buidler_dict,
    )
    # 保存训练历史
    print('history: ', history)
    history_converted = {
        key: [v.tolist() for v in value] for key, value in history.items()
    }
    with open('history.json', 'w', encoding='utf-8') as f:
        json.dump(history_converted, f, ensure_ascii=False, indent=4)


def plot_metric(dfhistory, metric, epoch):
    plt.figure()
    train_metrics = dfhistory["train_" + metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, epoch + 1)
    plt.plot(epochs, train_metrics[:epoch], 'bo--')
    plt.plot(epochs, val_metrics[:epoch], 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    file_path = '{}.png'.format(metric)  # 你可以指定完整的文件路径
    plt.savefig(file_path)
    plt.show()


if __name__ == '__main__':
    run()
    # 读取训练历史并绘制曲线
    with open('history.json', 'r', encoding='utf-8') as file:
        # data = json.load(file)
        train_val_his = json.load(file)
    plot_metric(train_val_his, 'loss', epoch=30)
    plot_metric(train_val_his, 'BinaryAccuracy', epoch=30)
    plot_metric(train_val_his, 'BinaryPrecision', epoch=30)
    plot_metric(train_val_his, 'BinaryAUROC', epoch=30)
