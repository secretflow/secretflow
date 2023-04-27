# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from secretflow.component.component import Component, IoType
from secretflow.data.split import train_test_split as train_test_split_fn
from secretflow.data.vertical import read_csv
from secretflow.device.device.pyu import PYU
from secretflow.device.driver import wait
from secretflow.protos.component.comp_def_pb2 import TableType

train_test_split_comp = Component(
    "train_test_split",
    domain="preprocessing",
    version="0.0.1",
    desc="""Split arrays or matrices into random train and test subsets.
    Check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """,
)


train_test_split_comp.float_param(
    name="train_size",
    desc="proportion of the dataset to include in the train split.",
    is_list=False,
    is_optional=True,
    default_value=0.75,
    allowed_values=None,
    lower_bound=0.0,
    upper_bound=1.0,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)
train_test_split_comp.float_param(
    name="test_size",
    desc="proportion of the dataset to include in the test split.",
    is_list=False,
    is_optional=True,
    default_value=0.25,
    allowed_values=None,
    lower_bound=0.0,
    upper_bound=1.0,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)
train_test_split_comp.int_param(
    name="random_state",
    desc="Controls the shuffling applied to the data before applying the split.",
    is_list=False,
    is_optional=True,
    default_value=1234,
)
train_test_split_comp.bool_param(
    name="shuffle",
    desc="Whether or not to shuffle the data before splitting.",
    is_list=False,
    is_optional=True,
    default_value=True,
)
train_test_split_comp.table_io(
    io_type=IoType.INPUT,
    name="input",
    desc="input",
    types=[TableType.VERTICAL_PARTITIONING_TABLE],
    col_params=None,
)
train_test_split_comp.table_io(
    io_type=IoType.OUTPUT,
    name="train",
    desc="train",
    types=[TableType.VERTICAL_PARTITIONING_TABLE],
    col_params=None,
)
train_test_split_comp.table_io(
    io_type=IoType.OUTPUT,
    name="test",
    desc="test",
    types=[TableType.VERTICAL_PARTITIONING_TABLE],
    col_params=None,
)


@train_test_split_comp.eval_fn
def train_test_split_eval_fn(
    *, ctx, train_size, test_size, random_state, shuffle, input, train, test
):
    input_parties = input.table_metadata.vertical_partitioning.parties
    input_paths = input.table_metadata.vertical_partitioning.paths

    train_parties = train.table_metadata.vertical_partitioning.parties
    train_paths = train.table_metadata.vertical_partitioning.paths

    test_parties = test.table_metadata.vertical_partitioning.parties
    test_paths = test.table_metadata.vertical_partitioning.paths

    pyus = {k: PYU(k) for k in ctx['pyu']}
    assert len(pyus) == 2

    input_dict = {pyus[k]: v for k, v in zip(input_parties, input_paths)}
    train_dict = {pyus[k]: v for k, v in zip(train_parties, train_paths)}
    test_dict = {pyus[k]: v for k, v in zip(test_parties, test_paths)}

    input_df = read_csv(input_dict)

    train_df, test_df = train_test_split_fn(
        input_df,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    wait(train_df.to_csv(train_dict, index=False))
    wait(test_df.to_csv(test_dict, index=False))
