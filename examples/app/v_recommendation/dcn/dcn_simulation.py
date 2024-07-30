import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import secretflow as sf


class AliceDataset(Dataset):
    def __init__(self, df, label_df, vocab_dir=""):
        self.df = df
        self.label_df = label_df
        cat_features = [x for x in self.df.columns if x.startswith('C')]
        num_features = [x for x in self.df.columns if x.startswith('I')]
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

        categories = [self.df[col].max() + 1 for col in cat_features]

        self.categories = categories

    def __getitem__(self, index):

        return (self.x_num[index], self.x_cat[index]), self.label[index]

    def __len__(self):
        return len(self.label_df)

    def get_categories(self):
        return self.categories


class BobDataset(Dataset):
    def __init__(self, df, vocab_dir=""):
        self.df = df
        cat_features = [x for x in self.df.columns if x.startswith('C')]
        num_features = [x for x in self.df.columns if x.startswith('I')]
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

        categories = [self.df[col].max() + 1 for col in cat_features]
        self.categories = categories

    def __getitem__(self, index):

        return (self.x_num[index], self.x_cat[index])

    def __len__(self):
        return len(self.df)

    def get_categories(self):
        return self.categories


gen_data_path = (
    r'/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/'
)
train_alice = pd.read_csv(gen_data_path + 'train_alice.csv', sep='|')
train_alice_label = train_alice['label']
train_alice_data = train_alice.drop(columns=["label"])
alice_dataset = AliceDataset(train_alice_data, train_alice_label)

train_bob = pd.read_csv(gen_data_path + 'train_bob.csv', sep='|')
bob_dataset = BobDataset(train_bob)


def create_dataset_builder_alice(batch_size=32):
    def dataset_builder(x):
        data_set = AliceDataset(x[0], x[1], gen_data_path)
        dataloader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


def create_dataset_builder_bob(batch_size=32):
    def dataset_builder(x):
        data_set = BobDataset(x[0], gen_data_path)
        dataloader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
        )
        return dataloader

    return dataset_builder


# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob'], address="local", log_to_driver=False)
alice, bob = sf.PYU('alice'), sf.PYU('bob')


batch_size = 32
dataset_buidler_dict = {
    alice: create_dataset_builder_alice(batch_size=batch_size),
    bob: create_dataset_builder_bob(
        batch_size=batch_size,
    ),
}

import pandas as pd
from model.sl_dcn_torch import DCNBase, DCNFuse

alice_base_out_dim = 0
bob_base_out_dim = 0


# 构建模型
def create_base_model_alice():
    d_numerical = 6
    d_embed_max = 8
    categories = alice_dataset.get_categories()
    d_cat_sum = sum([min(max(int(x**0.5), 2), d_embed_max) for x in categories])
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
    d_numerical = 7
    d_embed_max = 8
    categories = bob_dataset.get_categories()
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


from torch import nn, optim
from torchmetrics import AUROC, Accuracy, Precision

from secretflow.ml.nn import SLModel
from secretflow.ml.nn.core.torch import metric_wrapper, optim_wrapper
from secretflow.ml.nn.utils import TorchModel

# loss_fn = nn.CrossEntropyLoss
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


from secretflow.data.vertical import read_csv

vdf = read_csv(
    {
        alice: '/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/train_alice.csv',
        bob: '/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/train_bob.csv',
    },
    delimiter='|',
)
label = vdf["label"]
data = vdf.drop(columns=["label"])

val_vdf = read_csv(
    {
        alice: '/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/val_alice.csv',
        bob: '/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/val_bob.csv',
    },
    delimiter='|',
)

val_label = val_vdf["label"]
val_data = val_vdf.drop(columns=["label"])

epoch = 10
history = sl_model.fit(
    data,
    label,
    validation_data=(data, label),
    epochs=epoch,
    batch_size=batch_size,
    shuffle=False,
    random_seed=1234,
    dataset_builder=dataset_buidler_dict,
)
print('history: ', history)
