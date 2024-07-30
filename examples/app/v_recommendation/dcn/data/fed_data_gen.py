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

import pandas as pd

origin_train_data_path = r"/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/criteo_train_small.csv"
origin_val_data_path = r"/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/criteo_val_small.csv"

origin_train_df = pd.read_csv(origin_train_data_path)
origin_val_df = pd.read_csv(origin_val_data_path)

alice_col = [
    'label',
    'I1',
    'I2',
    'I3',
    'I4',
    'I5',
    'I6',
    'C14',
    'C15',
    'C16',
    'C17',
    'C18',
    'C19',
    'C20',
    'C21',
    'C22',
    'C23',
    'C24',
    'C25',
    'C26',
]
bob_col = [
    'I7',
    'I8',
    'I9',
    'I10',
    'I11',
    'I12',
    'I13',
    'C27',
    'C28',
    'C29',
    'C30',
    'C31',
    'C32',
    'C33',
    'C34',
    'C35',
    'C36',
    'C37',
    'C38',
    'C39',
]
sample_train_data = origin_train_df.sample(frac=0.1)
sample_val_data = origin_val_df.sample(frac=0.1)

train_alice = sample_train_data.loc[:, alice_col]
train_bob = sample_train_data.loc[:, bob_col]

val_alice = sample_val_data.loc[:, alice_col]
val_bob = sample_val_data.loc[:, bob_col]

train_alice.to_csv(
    "/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/train_alice.csv",
    index=False,
    sep="|",
    encoding='utf-8',
)
train_bob.to_csv(
    "/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/train_bob.csv",
    index=False,
    sep="|",
    encoding='utf-8',
)

val_alice.to_csv(
    "/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/val_alice.csv",
    index=False,
    sep="|",
    encoding='utf-8',
)
val_bob.to_csv(
    "/root/develop/ant-sf/secretflow/examples/app/v_recommendation/dcn/data/val_bob.csv",
    index=False,
    sep="|",
    encoding='utf-8',
)
