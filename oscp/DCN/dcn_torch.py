import logging

import torch
from dataset import CriteoDataset
from torch.utils.data import DataLoader, Dataset

import secretflow as sf

fm = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s] [%(funcName)s:%(lineno)d]"
# 设置日志级别 打印日志
logging.basicConfig(
    level=logging.DEBUG,
    format=fm,
    filename="/root/develop/Open_Source/ant-sf/secretflow/oscp/testlog/log01.log",
)
# 基本用法


class AliceDataset(Dataset):
    def __init__(self, df, label_df, vocab_dir):
        self.df = df
        self.label_df = label_df

    def __getitem__(self, index):

        cat_features = [x for x in self.df.columns if x.startswith('C')]
        num_features = [x for x in self.df.columns if x.startswith('I')]

        x_cat = (
            torch.tensor(self.df[cat_features].values, dtype=torch.long)
            if cat_features
            else None
        )
        x_num = (
            torch.tensor(self.df[num_features].values, dtype=torch.float32)
            if num_features
            else None
        )

        label = torch.tensor(self.label_df.values, dtype=torch.float)
        logging.debug(
            "xnum特征: {}, xcat特征: {}, y标签: {}".format(
                sf.reveal(x_num), sf.reveal(x_cat), label
            )
        )
        return (x_num[index], x_cat[index]), label[index]

    def __len__(self):
        return len(self.label_df)


class BobDataset(Dataset):
    def __init__(self, df, vocab_dir):
        self.df = df

    def __getitem__(self, index):

        cat_features = [x for x in self.df.columns if x.startswith('C')]
        num_features = [x for x in self.df.columns if x.startswith('I')]

        x_cat = (
            torch.tensor(self.df[cat_features].values, dtype=torch.long)
            if cat_features
            else None
        )
        x_num = (
            torch.tensor(self.df[num_features].values, dtype=torch.float32)
            if num_features
            else None
        )
        # x_cat = torch.tensor([int(self.df['C'])])

        logging.debug("xnum特征: {}, xcat特征: {}".format(x_num, x_cat))
        return (x_num[index], x_cat[index])

    def __len__(self):
        return len(self.df)


gen_data_path = r'/root/develop/ant-sf/secretflow/OSCP/DCN/dcn_split'


def create_dataset_builder_alice(batch_size=32):
    def dataset_builder(x):
        print(x)
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
from sl_dcn_torch import DCNBase, DCNFuse


def compute_base_fuse_model_kwargs(d_embed_max):
    train_alice = pd.read_csv(
        "/root/develop/Open_Source/ant-sf/secretflow/oscp/DCN/dcn_split/train_alice.csv",
        sep="|",
    )
    train_bob = pd.read_csv(
        "/root/develop/Open_Source/ant-sf/secretflow/oscp/DCN/dcn_split/train_bob.csv",
        sep="|",
    )
    cat_features_alice = [x for x in train_alice.columns if x.startswith('C')]
    num_features_alice = [x for x in train_alice.columns if x.startswith('I')]
    cat_features_bob = [x for x in train_bob.columns if x.startswith('C')]
    num_features_bob = [x for x in train_bob.columns if x.startswith('I')]

    d_numerical_alice = train_alice[num_features_alice].values.shape[1]
    # embedding空间的大小
    categorical_alice = [train_alice[col].max() + 1 for col in cat_features_alice]

    d_numerical_bob = train_bob[num_features_bob].values.shape[1]
    categorical_bob = [train_bob[col].max() + 1 for col in cat_features_bob]
    # embedding维度

    alice_d_embed_list = [
        min(max(int(x**0.5), 2), d_embed_max) for x in categorical_alice
    ]
    bob_d_embed_list = [min(max(int(x**0.5), 2), d_embed_max) for x in categorical_bob]
    alice_d_cat_sum = sum(alice_d_embed_list)
    bob_d_cat_sum = sum(bob_d_embed_list)
    alice_d_in = d_numerical_alice + alice_d_cat_sum
    bob_d_in = d_numerical_bob + bob_d_cat_sum
    alice_mlp_layers = [128, 64, 32]
    bob_mlp_layers = [128, 64, 32]
    alice_embed_dict = {
        "categorical": categorical_alice,
        "d_embed_list": alice_d_embed_list,
    }
    bob_embed_dict = {
        "categorical": categorical_bob,
        "d_embed_list": bob_d_embed_list,
    }
    return (
        d_numerical_alice,
        d_numerical_bob,
        alice_embed_dict,
        bob_embed_dict,
        alice_d_in,
        bob_d_in,
        alice_mlp_layers,
        bob_mlp_layers,
    )


(
    d_numerical_alice,
    d_numerical_bob,
    alice_embed_dict,
    bob_embed_dict,
    alice_d_in,
    bob_d_in,
    alice_mlp_layers,
    bob_mlp_layers,
) = compute_base_fuse_model_kwargs(8)


# 构建模型
def create_base_model_alice():

    def create_model():
        model = DCNBase(
            d_numerical=d_numerical_alice,
            d_embed_dict=alice_embed_dict,
            d_embed_max=8,
            d_in=alice_d_in,
            n_cross=2,
            mlp_layers=alice_mlp_layers,
            mlp_dropout=0.25,
        )

        return model

    return create_model


def create_base_model_bob():
    def create_model():
        model = DCNBase(
            d_numerical=d_numerical_bob,
            d_embed_dict=bob_embed_dict,
            d_embed_max=8,
            d_in=bob_d_in,
            n_cross=2,
            mlp_layers=bob_mlp_layers,
            mlp_dropout=0.25,
        )

        return model

    return create_model


def create_fuse_model():
    total_fuse_dim = alice_d_in + bob_d_in + alice_mlp_layers[-1] + bob_mlp_layers[-1]

    def create_model():
        model = DCNFuse(n_classes=1, total_fuse_dim=total_fuse_dim)

        return model

    return create_model


from torch import nn, optim
from torchmetrics import AUROC, Accuracy, Precision

from secretflow.ml.nn import SLModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn.utils import TorchModel

# loss_fn = nn.CrossEntropyLoss
loss_fn = nn.BCEWithLogitsLoss
optim_fn = optim_wrapper(optim.Adam, lr=0.002, weight_decay=0.001)

base_model_alice = TorchModel(
    model_fn=create_base_model_alice(),
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, task="multiclass", num_classes=2, average='micro'),
        metric_wrapper(AUROC, task="binary"),
        metric_wrapper(Precision, task="multiclass", num_classes=2, average='micro'),
    ],
)

base_model_bob = TorchModel(
    model_fn=create_base_model_bob(),
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, task="multiclass", num_classes=2, average='micro'),
        metric_wrapper(AUROC, task="multiclass"),
        metric_wrapper(Precision, task="multiclass", num_classes=2, average='micro'),
    ],
)

# fuse_model = TorchModel(
#     model_fn=create_fuse_model(),
#     loss_fn=loss_fn,
#     optim_fn=optim_fn,
#     metrics=[
#         metric_wrapper(Accuracy, task="multiclass", num_classes=2, average='micro'),
#         metric_wrapper(Precision, task="multiclass", num_classes=2, average='micro'),
#         metric_wrapper(AUROC, task="multiclass", num_classes=2),
#     ],
# )

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
        alice: '/root/develop/Open_Source/ant-sf/secretflow/oscp/DCN/dcn_split/train_alice.csv',
        bob: '/root/develop/Open_Source/ant-sf/secretflow/oscp/DCN/dcn_split/train_bob.csv',
    },
    delimiter='|',
)
label = vdf["label"]
data = vdf.drop(columns=["label"])

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
