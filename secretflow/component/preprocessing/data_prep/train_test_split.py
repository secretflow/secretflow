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


import math
from typing import Tuple

import numpy as np
import pyarrow as pa

from secretflow.component.core import (
    Component,
    CompVDataFrame,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    register,
)


@register(domain="data_prep", version="1.0.0")
class TrainTestSplit(Component):
    '''
    Split datasets into random train and test subsets.

    - Please check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    '''

    train_size: float = Field.attr(
        desc="Proportion of the dataset to include in the train subset. The sum of test_size and train_size should be in the (0, 1] range.",
        default=0.75,
        bound_limit=Interval.open(0.0, 1.0),
    )
    test_size: float = Field.attr(
        desc="Proportion of the dataset to include in the test subset. The sum of test_size and train_size should be in the (0, 1] range.",
        default=0.25,
        bound_limit=Interval.open(0.0, 1.0),
    )
    random_state: int = Field.attr(
        desc="Specify the random seed of the shuffling.",
        default=1024,
        bound_limit=Interval.open(0.0, None),
    )
    shuffle: bool = Field.attr(
        desc="Whether to shuffle the data before splitting.",
        default=True,
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    train_ds: Output = Field.output(
        desc="Output train dataset.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    test_ds: Output = Field.output(
        desc="Output test dataset.",
        types=[DistDataType.VERTICAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        reader = CompVDataFrameReader(ctx.storage, ctx.tracer, self.input_ds)
        train_writer = CompVDataFrameWriter(ctx.storage, ctx.tracer, self.train_ds.uri)
        test_writer = CompVDataFrameWriter(ctx.storage, ctx.tracer, self.test_ds.uri)

        def split_fn(
            in_table: pa.Table,
            train_size: float,
            test_size: float,
            seed: int,
            shuffle: bool,
        ) -> Tuple[pa.Table, pa.Table]:
            total = in_table.shape[0]
            train = math.ceil(total * train_size)
            test = math.ceil(total * test_size)
            if shuffle:
                rand = np.random.RandomState(seed)
                indices = rand.permutation(total)
            else:
                indices = np.arange(total)

            return in_table.take(indices[:train]), in_table.take(
                indices[total - test :]
            )

        with train_writer, test_writer:
            for batch in reader:
                train_df = CompVDataFrame({}, batch.system_info)
                test_df = CompVDataFrame({}, batch.system_info)
                for pyu, table in batch.partitions.items():
                    train_data, test_data = pyu(split_fn)(
                        table.data,
                        self.train_size,
                        self.test_size,
                        self.random_state,
                        self.shuffle,
                    )
                    train_df.set_data(train_data)
                    test_df.set_data(test_data)
                train_writer.write(train_df)
                test_writer.write(test_df)

        train_writer.dump_to(self.train_ds)
        test_writer.dump_to(self.test_ds)
