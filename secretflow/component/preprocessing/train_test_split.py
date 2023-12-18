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
from secretflow.component.data_utils import (
    DistDataType,
    VerticalTableWrapper,
    dump_vertical_table,
    load_table,
)
from secretflow.data.split import train_test_split as train_test_split_fn

train_test_split_comp = Component(
    "train_test_split",
    domain="preprocessing",
    version="0.0.1",
    desc="""Split datasets into random train and test subsets.

    - Please check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """,
)


train_test_split_comp.float_attr(
    name="train_size",
    desc="Proportion of the dataset to include in the train subset. The sum of test_size and train_size should be in the (0, 1] range.",
    is_list=False,
    is_optional=True,
    default_value=0.75,
    allowed_values=None,
    lower_bound=0.0,
    upper_bound=1.0,
    lower_bound_inclusive=False,
    upper_bound_inclusive=False,
)
train_test_split_comp.float_attr(
    name="test_size",
    desc="Proportion of the dataset to include in the test subset. The sum of test_size and train_size should be in the (0, 1] range.",
    is_list=False,
    is_optional=True,
    default_value=0.25,
    allowed_values=None,
    lower_bound=0.0,
    upper_bound=1.0,
    lower_bound_inclusive=False,
    upper_bound_inclusive=False,
)
train_test_split_comp.int_attr(
    name="random_state",
    desc="Specify the random seed of the shuffling.",
    is_list=False,
    is_optional=True,
    default_value=1024,
    lower_bound=0,
    lower_bound_inclusive=False,
)
train_test_split_comp.bool_attr(
    name="shuffle",
    desc="Whether to shuffle the data before splitting.",
    is_list=False,
    is_optional=True,
    default_value=True,
)
train_test_split_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)
train_test_split_comp.io(
    io_type=IoType.OUTPUT,
    name="train",
    desc="Output train dataset.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)
train_test_split_comp.io(
    io_type=IoType.OUTPUT,
    name="test",
    desc="Output test dataset.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)


@train_test_split_comp.eval_fn
def train_test_split_eval_fn(
    *, ctx, train_size, test_size, random_state, shuffle, input_data, train, test
):
    input_df = load_table(
        ctx, input_data, load_features=True, load_ids=True, load_labels=True
    )

    pyus = list(input_df.partitions.keys())
    assert len(pyus) == 2

    with ctx.tracer.trace_running():
        train_df, test_df = train_test_split_fn(
            input_df,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    train_db = dump_vertical_table(
        ctx,
        train_df,
        train,
        VerticalTableWrapper.from_dist_data(input_data, train_df.shape[0]),
        input_data.system_info,
    )

    test_db = dump_vertical_table(
        ctx,
        test_df,
        test,
        VerticalTableWrapper.from_dist_data(input_data, test_df.shape[0]),
        input_data.system_info,
    )

    return {"train": train_db, "test": test_db}
