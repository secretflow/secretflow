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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.∏
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import pandas as pd

import secretflow.compute as sc
from secretflow.component.data_utils import (
    DistDataType,
    dump_vertical_table,
    load_table_select_and_exclude_pair,
    model_dumps,
    VerticalTableWrapper,
)
from secretflow.component.preprocessing.core.meta_utils import (
    apply_meta_change,
    dict_to_str,
    produce_meta_change,
)
from secretflow.component.preprocessing.core.version import (
    PREPROCESSING_RULE_MAX_MAJOR_VERSION,
    PREPROCESSING_RULE_MAX_MINOR_VERSION,
)
from secretflow.data.core import partition
from secretflow.device.driver import reveal


def float_almost_equal(
    a: Union[sc.Array, float], b: Union[sc.Array, float], epsilon: float = 1e-07
) -> sc.Array:
    return sc.less(sc.abs(sc.subtract(a, b)), epsilon)


def v_preprocessing_transform(
    ctx,
    in_ds,
    trans_features,
    transform_func,
    out_ds,
    out_rules,
    rules_name,
    load_features=True,
    load_labels=True,
    load_ids=True,
    assert_one_party=True,
):
    trans, remains = load_table_select_and_exclude_pair(
        ctx,
        in_ds,
        load_features=load_features,
        load_labels=load_labels,
        load_ids=load_ids,
        col_selects=trans_features,
    )

    if assert_one_party:
        assert len(trans.partitions) == 1

    drop_cols = {}
    add_features = {}
    add_labels = {}
    additional_info_objects = []
    runner_objs = []

    def _fit_transform(trans_data, remain_data):
        """
        wrap the fit transform funtion and return the transformed data,
        and info for new meta and trace_runner which can dump preprocessing additional_info for easy replay.

        fit_transform_f should takes on the trans_data (pd.DataFrame) as input
            and return transformed data in sc.table and (optional can be empty) additional_info for report.
        """
        assert trans_data is not None
        trans_data, add_labels, additional_info = transform_func(trans_data)
        runner = trans_data.dump_runner()
        drop_columns, add_columns = trans_data.column_changes()
        assert set(add_labels).issubset(set(add_columns))
        add_features = [c for c in add_columns if c not in set(add_labels)]

        trans_data = trans_data.to_pandas()
        if remain_data is not None:
            new_data = pd.concat([remain_data, trans_data], axis=1)
        else:
            new_data = trans_data

        return (
            new_data,
            drop_columns,
            add_features,
            add_labels,
            additional_info,
            runner,
        )

    with ctx.tracer.trace_running():
        new_datas = {}
        for pyu in trans.partitions.keys():
            trans_data = trans.partitions[pyu].data
            if pyu in remains.partitions.keys():
                remain_data = remains.partitions.pop(pyu).data
            else:
                remain_data = None

            (
                trans_data,
                drop_columns,
                add_fs,
                add_ls,
                additional_info,
                runner,
            ) = pyu(
                _fit_transform, num_returns=6
            )(trans_data, remain_data)

            new_datas[pyu] = trans_data
            drop_cols[pyu.party] = drop_columns
            add_features[pyu.party] = add_fs
            add_labels[pyu.party] = add_ls
            additional_info_objects.append(additional_info)
            runner_objs.append(runner)

        for pyu in new_datas:
            remains.partitions[pyu] = partition(new_datas[pyu])
        # meta info is not protected
        drop_cols = reveal(drop_cols)
        add_features = reveal(add_features)
        add_labels = reveal(add_labels)

    meta = VerticalTableWrapper.from_dist_data(in_ds, trans.shape[0])

    meta_change_dict = produce_meta_change(
        meta,
        drop_cols=drop_cols,
        new_labels=add_labels,
        new_features=add_features,
        ref_dtypes=remains.dtypes,
    )
    meta = apply_meta_change(meta, meta_change_dict)

    output_dd = dump_vertical_table(
        ctx,
        remains,
        out_ds,
        meta,
        in_ds.system_info,
    )

    # build rules for onehot_substitution
    model_dd = model_dumps(
        rules_name,
        DistDataType.PREPROCESSING_RULE,
        PREPROCESSING_RULE_MAX_MAJOR_VERSION,
        PREPROCESSING_RULE_MAX_MINOR_VERSION,
        runner_objs,
        dict_to_str(meta_change_dict),
        ctx.local_fs_wd,
        out_rules,
        in_ds.system_info,
    )

    return output_dd, model_dd, additional_info_objects
