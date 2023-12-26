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
from typing import List, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from spu.ops.groupby import (
    groupby,
    groupby_agg_postprocess,
    groupby_count_cleartext,
    groupby_max_via_shuffle,
    groupby_mean_via_shuffle,
    groupby_min_via_shuffle,
    groupby_sum_via_shuffle,
    groupby_var_via_shuffle,
    shuffle_cols,
    view_key_postprocessing,
)

from secretflow.device import PYU, SPU, SPUCompilerNumReturnsPolicy, SPUObject, reveal


def get_agg_fun(agg):
    if agg == 'sum':
        return groupby_sum_via_shuffle
    elif agg == 'max':
        return groupby_max_via_shuffle
    elif agg == 'min':
        return groupby_min_via_shuffle
    elif agg == 'mean':
        return groupby_mean_via_shuffle
    elif agg == 'var':
        return groupby_var_via_shuffle
    else:
        raise ValueError(f'Unknown agg {agg}')


# TODO(zoupeicheng.zpc): add version that return SPUObjects and append with special shuffle-reveal-postprocess
# TODO(zoupeicheng.zpc): allow reveal to a specific party.
# 1. use groupby ops instead of groupby with shuffle ops
# 2. return key and value as SPUObjects
# 3. add method shuffle_reveal_postprocess
class DataFrameGroupBy:
    def __init__(
        self,
        spu: SPU,
        parties: List[PYU],
        key_cols: List[SPUObject],
        target_cols: List[SPUObject],
        key_col_names: List[str],
        target_col_names: List[str],
        n_samples: int,
    ):
        """DataFrameGroupBy supports grouby operations (experimental feature).
        The shape of output is not protected, so the number of groups is revealed.

        Caution: when applying the aggregation function, the aggregation results will be revealed.

        The group number limit function not implemented here.


        Args:
            spu (SPU): SPU device for group by operation
            parties (List[PYU]): parties to generate random order
            key_cols (List[SPUObject]): by what to group by
            target_cols (List[SPUObject]): value columns to aggregate
            target_col_names (List[str]): name of value columns to aggregate
            n_samples (int): number of samples
            max_group_size (int): max number of groups for safety consideration, default to be 10000.
        """
        self.spu = spu
        assert len(key_cols) > 0, "number of key cols must be >0"
        assert len(target_cols) > 0, "number of target cols must be >0"
        assert len(key_col_names) == len(
            key_cols
        ), f"key cols number must match, get key name number{len(key_col_names)}, key number  {len(key_cols)}"
        key_columns_sorted, target_columns_sorted, segment_ids, seg_end_marks = spu(
            groupby, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_COMPILER
        )(key_cols, target_cols)
        self.parties = parties
        self.key_columns_sorted = key_columns_sorted
        self.target_columns_sorted = target_columns_sorted
        self.segment_ids = segment_ids
        self.seg_end_marks = seg_end_marks
        self.target_columns_names = target_col_names
        self.key_col_names = key_col_names
        self.num_key_cols = len(key_cols)
        self.n_samples = n_samples
        self.num_groups = int(reveal(spu(lambda x: x[-1] + 1)(segment_ids)))

    def set_max_group_size(self, max_group_size: int):
        self.max_group_size = max_group_size

    def gen_secret_random_order(self):
        """
        generate random order for each party
        """
        shape = (self.n_samples,)
        spu = self.spu
        random_orders = [
            party(lambda: np.random.random(shape))().to(spu) for party in self.parties
        ]

        def element_wise_sum(arrays):
            return jnp.sum(jnp.vstack(arrays), axis=0)

        return spu(element_wise_sum)(random_orders)

    def __getitem__(self, *target_col_names: str) -> 'DataFrameGroupBy':
        if isinstance(target_col_names, tuple):
            target_col_names = target_col_names[0]
        if isinstance(target_col_names, str):
            target_col_names = [target_col_names]

        assert sum(
            [col in self.target_columns_names for col in target_col_names]
        ) == len(
            target_col_names
        ), f"target col names must be in {self.target_columns_names}, got {target_col_names}, {len(target_col_names)}, {type(target_col_names)}"
        cols = [
            self.target_columns_sorted[self.target_columns_names.index(i)]
            for i in target_col_names
        ]
        self.target_columns_names = target_col_names
        self.target_columns_sorted = cols
        return self

    def _reveal_and_postprocess_value(self, agg_result: SPUObject) -> np.ndarray:
        agg_result = reveal(agg_result)
        agg_result = groupby_agg_postprocess(
            agg_result[0],
            agg_result[1],
            agg_result[2],
            self.num_groups,
        )
        # align with pandas behavior
        if agg_result.shape[1] == 1:
            return agg_result.reshape(
                -1,
            )
        return agg_result

    def _view_key(self, to_list=True):
        key_cols = self.key_columns_sorted
        secret_order = self.gen_secret_random_order()

        segment_end_marks = self.seg_end_marks

        keys = reveal(self.spu(shuffle_cols)(key_cols, segment_end_marks, secret_order))
        keys = view_key_postprocessing(keys, self.num_groups)
        assert keys.shape[1] == len(self.key_col_names), f"{keys}"
        if to_list:
            return matrix_to_cols(keys)
        else:
            return keys

    def _agg(
        self, fn_name: str, target_col_names: List[str] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """Apply aggregation function to the groupby segregation intermediate results.
        Note that the current implementation will directly return the aggregation result.
        """
        col_names = (
            self.target_columns_names if target_col_names is None else target_col_names
        )

        for col_name in col_names:
            assert (
                col_name in self.target_columns_names
            ), f"{col_name} not in {self.target_columns_names}"
        cols = [
            self.target_columns_sorted[self.target_columns_names.index(col_name)]
            for col_name in col_names
        ]
        secret_order = self.gen_secret_random_order()

        segment_end_marks = self.seg_end_marks
        segment_ids = self.segment_ids
        agg_fun = get_agg_fun(fn_name)
        agg_result = self.spu(
            agg_fun,
        )(cols, segment_end_marks, segment_ids, secret_order)

        values = self._reveal_and_postprocess_value(agg_result)
        return self.to_df(values, col_names)

    # NOTE: this is infact not ok. Since SPU cannot differentiate NA from 0.
    # The result is that all columns have the group sample number as counts
    # DOES NOT TRULY SUPPORT THIS CASE FOR NOW.
    def count(
        self, target_col_names: List[str] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        segment_ids = self.segment_ids
        target_col_names = (
            self.target_columns_names if target_col_names is None else target_col_names
        )
        target_col_num = len(target_col_names)
        if (target_col_num == 1) and (self.num_key_cols == 1):
            values = groupby_count_cleartext(reveal(segment_ids))

        else:
            reshaped_column = groupby_count_cleartext(reveal(segment_ids))[
                :, np.newaxis
            ]
            values = np.repeat(reshaped_column, target_col_num, axis=1)
        return self.to_df(values, target_col_names, cast_type=np.int32)

    def to_df(self, values, target_col_names, cast_type=None):
        if isinstance(target_col_names, tuple):
            target_col_names = list(target_col_names)
        target_col_num = len(target_col_names)
        by_keys = self._view_key(to_list=False)
        assert by_keys.shape[1] == len(
            self.key_col_names
        ), f"by key data {by_keys}, key names {self.key_col_names}"

        if (target_col_num == 1) and (self.num_key_cols == 1):
            if cast_type is not None:
                values = values.astype(cast_type)
            return pd.Series(
                data=values,
                index=by_keys.reshape((-1)),
                name=target_col_names[0],
            )
        else:
            if len(by_keys.shape) < 2:
                by_keys = by_keys.reshape((-1, 1))
            if len(values.shape) < 2:
                values = values.reshape((-1, 1))
            all_values = np.concatenate([by_keys, values], axis=1)
            df = pd.DataFrame(
                data=all_values, columns=self.key_col_names + target_col_names
            )
            df.set_index(self.key_col_names, inplace=True)
            if cast_type is not None:
                df[target_col_names] = df[target_col_names].astype(cast_type)
            return df

    def sum(self, target_col_names: List[str] = None):
        return self._agg('sum', target_col_names)

    def mean(self, target_col_names: List[str] = None):
        return self._agg('mean', target_col_names)

    def var(self, target_col_names: List[str] = None):
        return self._agg('var', target_col_names)

    def max(self, target_col_names: List[str] = None):
        return self._agg('max', target_col_names)

    def min(self, target_col_names: List[str] = None):
        return self._agg('min', target_col_names)


def matrix_to_cols(matrix):
    return [matrix[:, i] for i in range(matrix.shape[1])]
