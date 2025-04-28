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

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from secretflow.data.vertical import VDataFrame, read_csv
from secretflow.device.device.pyu import PYU
from secretflow_fl.utils.simulation.datasets_fl import load_criteo_unpartitioned

# split origin 1m data to alice and bob, each has different column, alice_criteo_train_1m.csv and bob_criteo_train_1m.csv are generated
# to static cat_num_categories for alice and bob, which will be used in dcn-model's input parameters

_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.secretflow/datasets')


def generate_alice_bob_criteo_1m_data(num_samples: int = 1000000):
    dfdata = load_criteo_unpartitioned(num_samples=num_samples)

    dfdata.columns = (
        ["label"]
        + ["I" + str(x) for x in range(1, 14)]
        + ["C" + str(x) for x in range(14, 40)]
    )

    cat_cols = [x for x in dfdata.columns if x.startswith('C')]
    num_cols = [x for x in dfdata.columns if x.startswith('I')]
    num_pipe = Pipeline(
        steps=[('impute', SimpleImputer()), ('quantile', QuantileTransformer())]
    )

    for col in cat_cols:
        dfdata[col] = LabelEncoder().fit_transform(dfdata[col])

    dfdata[num_cols] = num_pipe.fit_transform(dfdata[num_cols])

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

    alice_criteo_train_1m = dfdata.loc[:, alice_col]
    bob_criteo_train_1m = dfdata.loc[:, bob_col]

    return alice_criteo_train_1m, bob_criteo_train_1m


# generate and save train, val and test data for alice and bob
def load_criteo_partitioned(
    part: List[PYU],
    num_samples: int = 1000000,
    split_ratio: float = 0.2,
    data_dir: str = None,
    train: bool = True,
):
    alice_criteo_train_1m, bob_criteo_train_1m = generate_alice_bob_criteo_1m_data(
        num_samples=num_samples
    )
    data_dir = (
        os.path.join(data_dir, 'criteo_partitioned')
        if data_dir
        else os.path.join(_CACHE_DIR, 'criteo_partitioned')
    )

    if not Path(data_dir).is_dir():
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    def _generate_train_val_test(
        alice_criteo_train_1m: pd.DataFrame,
        bob_criteo_train_1m: pd.DataFrame,
        split_ratio: float = 0.2,
    ):
        alice_train_val, alice_test = train_test_split(
            alice_criteo_train_1m, test_size=split_ratio
        )
        alice_train, alice_val = train_test_split(
            alice_train_val, test_size=split_ratio
        )
        bob_train_val, bob_test = train_test_split(
            bob_criteo_train_1m, test_size=split_ratio
        )
        bob_train, bob_val = train_test_split(bob_train_val, test_size=split_ratio)
        alice_train.to_csv(
            os.path.join(data_dir, "train_alice.csv"),
            index=False,
            sep="|",
            encoding='utf-8',
        )

        bob_train.to_csv(
            os.path.join(data_dir, "train_bob.csv"),
            index=False,
            sep="|",
            encoding='utf-8',
        )

        alice_val.to_csv(
            os.path.join(data_dir, "val_alice.csv"),
            index=False,
            sep="|",
            encoding='utf-8',
        )

        bob_val.to_csv(
            os.path.join(data_dir, "val_bob.csv"),
            index=False,
            sep="|",
            encoding='utf-8',
        )

    _generate_train_val_test(alice_criteo_train_1m, bob_criteo_train_1m, split_ratio)

    fed_train_csv = {}
    fed_val_csv = {}
    for p in part:
        fed_train_csv[p] = os.path.join(data_dir, "train_{}.csv".format(p))
        fed_val_csv[p] = os.path.join(data_dir, "val_{}.csv".format(p))

    if train:
        return read_csv(
            fed_train_csv,
            delimiter="|",
        )
    else:
        return read_csv(
            fed_val_csv,
            delimiter="|",
        )
