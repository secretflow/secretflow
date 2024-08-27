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

import copy
import json
import logging
from typing import Union

import pandas as pd

import secretflow.compute as sc
from secretflow.component.data_utils import DistDataType, model_dumps
from secretflow.component.dataframe import (
    CompDataFrame,
    load_table_select_and_exclude_pair,
)
from secretflow.component.preprocessing.core.version import (
    PREPROCESSING_RULE_MAX_MAJOR_VERSION,
    PREPROCESSING_RULE_MAX_MINOR_VERSION,
)
from secretflow.data.core import partition
from secretflow.data.vertical import VDataFrame
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
        assert len(trans.partitions) == 1, (
            f"preprocessing {transform_func.__name__} can only handle features from one party, "
            f"but got trans_features {trans_features} from parties {trans.partitions.keys()}"
        )

    drop_cols = {}
    add_features = {}
    add_labels = {}
    additional_info_objects = []
    runner_objs = []

    def _fit_transform(trans_data, remain_data):
        """
        wrap the fit transform funtion and return the transformed data.

        fit_transform_f should takes on the trans_data (pa.Table) as input
            and return transformed data in sc.table and (optional can be empty) additional_info for report.
        """
        assert trans_data is not None
        try:
            trans_data, add_labels, additional_info = transform_func(trans_data)
        except Exception as e:
            logging.exception(f"transform_func error.")
            return None, None, None, None, None, None, e
        runner = trans_data.dump_runner()
        drop_columns, add_columns, _ = trans_data.column_changes()
        add_labels = [] if add_labels is None else add_labels
        assert set(add_labels).issubset(set(add_columns))
        add_features = [c for c in add_columns if c not in set(add_labels)]

        trans_data = trans_data.to_table()
        if remain_data is not None:
            for i in range(trans_data.shape[1]):
                remain_data = remain_data.append_column(
                    trans_data.field(i), trans_data.column(i)
                )
            new_data = remain_data
        else:
            new_data = trans_data

        return (
            new_data,
            drop_columns,
            add_features,
            add_labels,
            additional_info,
            runner,
            None,
        )

    with ctx.tracer.trace_running():
        new_datas = {}
        errors = []
        for pyu in trans.partitions.keys():
            trans_data = trans.data(pyu)
            if remains is not None and remains.has_data(pyu):
                remain_data = remains.data(pyu)
            else:
                remain_data = None

            (
                trans_data,
                drop_columns,
                add_fs,
                add_ls,
                additional_info,
                runner,
                err_obj,
            ) = pyu(_fit_transform, num_returns=7)(trans_data, remain_data)

            errors.append(err_obj)

            new_datas[pyu] = trans_data
            drop_cols[pyu.party] = drop_columns
            add_features[pyu.party] = add_fs
            add_labels[pyu.party] = add_ls
            additional_info_objects.append(additional_info)
            runner_objs.append(runner)

        errors = [e for e in reveal(errors) if e is not None]
        if errors:
            raise errors[0]
        drop_cols = reveal(drop_cols)
        add_features = reveal(add_features)
        add_labels = reveal(add_labels)

    new_partitions = copy.deepcopy(trans.partitions)
    for pyu in trans.partitions:
        trans_table = trans.partitions[pyu]
        if remains is not None and pyu in remains.partitions:
            remain_table = remains.partitions.pop(pyu)
        else:
            remain_table = None
        drop_col = set(drop_cols[pyu.party])
        add_feature = add_features[pyu.party]
        add_label = add_labels[pyu.party]
        new_partitions[pyu].data = new_datas[pyu]
        new_partitions[pyu].id_cols = remain_table.id_cols if remain_table else []
        new_partitions[pyu].id_cols.extend(trans_table.id_cols)
        new_partitions[pyu].feature_cols = (
            remain_table.feature_cols if remain_table else []
        )
        new_partitions[pyu].feature_cols.extend(
            [c for c in trans_table.feature_cols if c not in drop_col]
        )
        new_partitions[pyu].feature_cols.extend(add_feature)
        new_partitions[pyu].label_cols = remain_table.label_cols if remain_table else []
        new_partitions[pyu].label_cols.extend(
            [c for c in trans_table.label_cols if c not in drop_col]
        )
        new_partitions[pyu].label_cols.extend(add_label)

    if remains:
        for pyu in remains.partitions:
            new_partitions[pyu] = remains.partitions[pyu]

    new_ds = CompDataFrame(new_partitions, in_ds.system_info)

    # build rules for substitution component
    model_dd = model_dumps(
        ctx,
        rules_name,
        DistDataType.PREPROCESSING_RULE,
        PREPROCESSING_RULE_MAX_MAJOR_VERSION,
        PREPROCESSING_RULE_MAX_MINOR_VERSION,
        runner_objs,
        json.dumps(add_labels),
        out_rules,
        in_ds.system_info,
    )

    return new_ds.to_distdata(ctx, out_ds), model_dd, additional_info_objects
