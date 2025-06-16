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
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# 预处理数据
def DataPreprocess():
    # import data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    print(train_df.shape, test_df.shape)
    # 先将label这一列保存起来，再从train中删除
    label = train_df['Label']
    del train_df['Label']

    # 进行数据合并，为了同时对train和test数据进行预处理
    data_df = pd.concat((train_df, test_df))

    del data_df['Id']

    print(data_df.columns)

    # 特征分开类别
    sparse_feas = [col for col in data_df.columns if col[0] == 'C']
    dense_feas = [col for col in data_df.columns if col[0] == 'I']

    # 填充缺失值
    data_df[sparse_feas] = data_df[sparse_feas].fillna('-1')
    data_df[dense_feas] = data_df[dense_feas].fillna(0)

    # 进行编码  类别特征编码
    for feat in sparse_feas:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # 数值特征归一化
    mms = MinMaxScaler()
    data_df[dense_feas] = mms.fit_transform(data_df[dense_feas])

    # 创建一个列表来存储每个稀疏特征的最大类别数
    sparse_feature_info = []

    # 对每个稀疏特征列，计算其最大值，并将其添加到列表中
    for feature in sparse_feas:
        max_value = data_df[feature].max()
        sparse_feature_info.append(
            {'feat_num': int(max_value) + 1}
        )  # +1 因为类别从 0 开始

    # 将列表转换为 NumPy 数组并保存
    np.save('fea_col.npy', sparse_feature_info)

    # 分开测试集和训练集
    train = data_df[: train_df.shape[0]]
    test = data_df[train_df.shape[0] :]

    train['Label'] = label

    train_set, val_set = train_test_split(train, test_size=0.2, random_state=2020)

    print(train_set['Label'].value_counts())
    print(val_set['Label'].value_counts())

    # 保存文件
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)

    train_set.to_csv('data/train_set1.csv', index=0)
    val_set.to_csv('data/val_set1.csv', index=0)
    test.to_csv('data/test_set1.csv', index=0)


# 存储每个稀疏特征的最大类别数
def generate_fea_col(filename, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(filename)

    # 找到所有稀疏特征列（以 'C' 开头）
    sparse_features = [col for col in df.columns if col.startswith('C')]

    # 创建一个列表来存储每个稀疏特征的最大类别数

    max_value_dict = {feature: df[feature].max() for feature in sparse_features}
    # 保存字典为 .npy 文件
    np.save(output_file, max_value_dict)

    print("max_value_dict 已保存为 max_value_dict.npy")


# 得到训练所使用的数据形式
def getTrainData(filename, feafile, columns_for_alice, columns_for_bob):
    df = pd.read_csv(filename)
    print(df.columns)

    # C开头的列代表稀疏特征，I开头的列代表的是稠密特征
    dense_features_cols_alice = [col for col in columns_for_alice if col[0] == 'I']
    dense_features_cols_bob = [col for col in columns_for_bob if col[0] == 'I']

    # 这个文件里面存储了稀疏特征的最大范围，用于设置Embedding的输入维度

    max_fea_col = np.load(feafile, allow_pickle=True).item()

    # fea_col = pickle.load(feafile)
    sparse_features_cols_alice = []
    for col in columns_for_alice:
        if col[0] == 'C':
            sparse_features_cols_alice.append(max_fea_col[col])
    sparse_features_cols_bob = []
    for col in columns_for_bob:
        if col[0] == 'C':
            sparse_features_cols_bob.append(max_fea_col[col])

    # 将处理后的数据用于模型训练
    df_data_alice = df[columns_for_alice].values
    df_data = df[columns_for_bob]
    df_data_bob, labels = df_data.drop(columns='Label').values, df_data['Label'].values

    return (
        df_data_alice,
        df_data_bob,
        labels,
        dense_features_cols_alice,
        sparse_features_cols_alice,
        dense_features_cols_bob,
        sparse_features_cols_bob,
    )
