# Copyright 2024 Ant Group Co., Ltd.
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

import bisect
import random
import sys

import pandas as pd

from secretflow import reveal
from secretflow.component.component import CompEvalError, Component, IoType
from secretflow.component.data_utils import (
    DistDataType,
    VerticalTableWrapper,
    dump_table,
    load_table,
)
from secretflow.data import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab

sample_comp = Component(
    "sample",
    domain="data_prep",
    version="0.0.1",
    desc="Sample data set.",
)
sample_comp.int_attr(
    name="random_state",
    desc="Specify the random seed of the shuffling.",
    is_list=False,
    is_optional=True,
    default_value=1024,
    lower_bound=0,
    lower_bound_inclusive=False,
)
sample_comp.union_attr_group(
    name="sample_num_option",
    desc="sample number algorithm and parameters",
    group=[
        sample_comp.int_attr(
            name="number",
            desc="Specify the total number of sampling.",
            is_list=False,
            is_optional=True,
            default_value=1024,
            lower_bound=0,
            lower_bound_inclusive=False,
        ),
        sample_comp.float_attr(
            name="frac",
            desc="Proportion of the dataset to sample in the set. The fraction should be larger than 0.",
            is_list=False,
            is_optional=True,
            default_value=0.75,
            allowed_values=None,
            lower_bound=0.0,
            lower_bound_inclusive=False,
        ),
    ],
)
sample_comp.union_attr_group(
    name="sample_algorithm",
    desc="sample algorithm and parameters",
    group=[
        sample_comp.struct_attr_group(
            name="random",
            desc="Random sample.",
            group=[
                sample_comp.bool_attr(
                    name="replacement",
                    desc="If true, sampling with replacement. If false, sampling without replacement.",
                    is_list=False,
                    is_optional=True,
                    default_value=False,
                ),
            ],
        ),
        sample_comp.union_selection_attr(
            name="system",
            desc="system sample.",
        ),
        sample_comp.struct_attr_group(
            name="stratify",
            desc="stratify sample.",
            group=[
                sample_comp.str_attr(
                    name="observe_feature",
                    desc="stratify sample observe feature.",
                    is_list=False,
                    is_optional=False,
                    default_value="",
                ),
                sample_comp.bool_attr(
                    name="replacements",
                    desc="If true, sampling with replacement. If false, sampling without replacement.",
                    is_list=True,
                    is_optional=False,
                    default_value=[],
                ),
                sample_comp.float_attr(
                    name="quantiles",
                    desc="stratify sample quantiles",
                    is_list=True,
                    is_optional=False,
                    default_value=[],
                ),
                sample_comp.float_attr(
                    name="weights",
                    desc="stratify sample weights",
                    is_list=True,
                    is_optional=False,
                    default_value=[],
                    lower_bound=0.0,
                    upper_bound=1.0,
                    lower_bound_inclusive=False,
                    upper_bound_inclusive=False,
                ),
            ],
        ),
    ],
)
sample_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)
sample_comp.io(
    io_type=IoType.OUTPUT,
    name="sample_output",
    desc="Output sampled dataset.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)
sample_comp.io(
    io_type=IoType.OUTPUT,
    name="reports",
    desc="Output sample reports",
    types=[DistDataType.REPORT],
)


# 随机采样
# if replacement==true 执行放回采样
# if replacement==false执行不放回采样
RANDOM_SAMPLE = "random"
# 系统等距抽样
SYSTEM_SAMPLE = "system"
# 分层抽样
STRATIFY_SAMPLE = "stratify"


@sample_comp.eval_fn
def sample_fn(
    *,
    ctx,
    random_state,
    sample_num_option,
    sample_num_option_number,
    sample_num_option_frac,
    sample_algorithm,
    sample_algorithm_random_replacement,
    sample_algorithm_stratify_observe_feature,
    sample_algorithm_stratify_replacements,
    sample_algorithm_stratify_quantiles,
    sample_algorithm_stratify_weights,
    input_data,
    sample_output,
    reports,
):
    input_df = load_table(
        ctx, input_data, load_features=True, load_ids=True, load_labels=True
    )

    pyus = list(input_df.partitions.keys())
    part0 = input_df.partitions[pyus[0]]
    assert isinstance(input_df, VDataFrame), "input_df must be VDataFrame"

    # get row number
    total_num = part0.shape[0]
    sample_algorithm_obj = SampleAlgorithmFactory().create_sample_algorithm(
        input_df,
        total_num,
        sample_num_option,
        sample_num_option_number,
        sample_num_option_frac,
        sample_algorithm,
        sample_algorithm_random_replacement,
        sample_algorithm_stratify_replacements,
        sample_algorithm_stratify_quantiles,
        sample_algorithm_stratify_weights,
    )
    sample_df, report_results = sample_algorithm_obj.perform_sample(
        random_state,
        sample_algorithm_stratify_observe_feature,
    )

    if input_data.type == DistDataType.VERTICAL_TABLE:
        meta = VerticalTableWrapper.from_dist_data(input_data, sample_df.shape[0])
    else:
        meta = IndividualTable()
        input_data.meta.Unpack(meta)

    sample_db = dump_table(
        ctx,
        sample_df,
        sample_output,
        meta,
        input_data.system_info,
    )

    report_dd = transform_report(
        reports, sample_algorithm, report_results, input_data.system_info
    )
    return {"sample_output": sample_db, "reports": report_dd}


def build_sample_desc(result):
    desc = Descriptions(
        items=[
            Descriptions.Item(
                name="num_before_sample", type="int", value=Attribute(i64=result[0])
            ),
            Descriptions.Item(
                name="num_after_sample", type="int", value=Attribute(i64=result[1])
            ),
            Descriptions.Item(
                name="sample_rate", type="float", value=Attribute(f=result[2])
            ),
        ]
    )
    return desc


def sample_div_report(report_result):
    return [
        Div(
            children=[
                Div.Child(
                    type="descriptions",
                    descriptions=build_sample_desc(report_result),
                )
            ],
        )
    ]


def transform_report(reports, sample_algorithm, report_results, system_info):
    meta = Report()
    tabs = []
    if sample_algorithm == STRATIFY_SAMPLE:
        for i in range(len(report_results)):
            if i == 0:
                tabs.append(
                    Tab(
                        name="total_sample_info",
                        desc="total sample info",
                        divs=sample_div_report(report_results[i]),
                    )
                )
            else:
                tabs.append(
                    Tab(
                        name=f'bucket_{i - 1}',
                        desc='bucket sample info',
                        divs=sample_div_report(report_results[i]),
                    )
                )

        meta = Report(
            name="reports",
            desc="stratify sample report",
            tabs=tabs,
        )

    report_dd = DistData(
        name=reports,
        type=str(DistDataType.REPORT),
        system_info=system_info,
    )
    report_dd.meta.Pack(meta)

    return report_dd


def calculate_sample_number(
    option, number: int, frac: float, replacement: bool, total_num: int
):
    assert option in [
        "number",
        "frac",
    ], f"option must be one of ['number', 'frac'], but got {option}"
    sample_num = 0
    if option == "number":
        sample_num = number
    else:
        sample_num = round(frac * total_num)

    if sample_num > total_num and not replacement:
        raise CompEvalError(
            f"Replacement has to be set to True when sample number {sample_num} is larger than dataset size {total_num}."
        )

    return sample_num


class SampleAlgorithmFactory:
    def create_sample_algorithm(
        self,
        input_df: VDataFrame,
        total_num: int,
        sample_num_option,
        sample_num_option_number,
        sample_num_option_frac,
        sample_algorithm,
        random_replacement: bool,
        stratify_replacements: list[bool],
        quantiles: list[float],
        weights: list[float],
    ):
        if sample_algorithm == RANDOM_SAMPLE:
            return RandomSampleAlgorithm(
                input_df,
                total_num,
                sample_num_option,
                sample_num_option_number,
                sample_num_option_frac,
                random_replacement,
            )
        elif sample_algorithm == SYSTEM_SAMPLE:
            return SystemSampleAlgorithm(
                input_df,
                total_num,
                sample_num_option,
                sample_num_option_number,
                sample_num_option_frac,
            )
        elif sample_algorithm == STRATIFY_SAMPLE:
            return StratifySampleAlgorithm(
                input_df,
                total_num,
                sample_num_option,
                sample_num_option_number,
                sample_num_option_frac,
                stratify_replacements,
                quantiles,
                weights,
            )
        else:
            raise AssertionError(
                f'sample_algorithm must be one of [{RANDOM_SAMPLE}, {SYSTEM_SAMPLE}, {STRATIFY_SAMPLE}], but got {sample_algorithm}'
            )


class SampleAlgorithm:
    def __init__(
        self,
        in_df: VDataFrame,
        total_num: int,
    ):
        self.in_df = in_df
        self.total_num = total_num

    def perform_sample(
        self,
        random_state: int,
        observe_feature: str,
    ):
        random_ids, report_results = self._algorithm(random_state, observe_feature)
        return self._filter_data(random_ids), report_results

    def _algorithm(
        self,
        random_state: int,
        observe_feature: str,
    ):
        pass

    def _filter_data(self, random_ids: list[int]):
        in_tables = {}

        def _filter(df: pd.DataFrame, random_ids: list[int]):
            rows = [(df.loc[idx]).tolist() for idx in random_ids]
            return pd.DataFrame(rows, columns=df.columns)

        for pyu, table in self.in_df.partitions.items():
            new_data = pyu(_filter)(table.data, random_ids)
            in_tables[pyu] = partition(new_data)

        return VDataFrame(in_tables)


class RandomSampleAlgorithm(SampleAlgorithm):
    def __init__(
        self,
        in_df: VDataFrame,
        total_num: int,
        sample_num_option,
        sample_num_option_number,
        sample_num_option_frac,
        replacement: bool,
    ):
        super().__init__(
            in_df,
            total_num,
        )
        self.replacement = replacement
        self.sample_num = calculate_sample_number(
            sample_num_option,
            sample_num_option_number,
            sample_num_option_frac,
            self.replacement,
            total_num,
        )

    def _algorithm(
        self,
        random_state: int,
        _observe_feature: str,
    ):
        device0 = next(iter(self.in_df.partitions))
        random_ids = device0(RandomSampleAlgorithm._random_algorithm)(
            random_state, self.replacement, self.total_num, self.sample_num
        )
        return reveal(random_ids), []

    @staticmethod
    def _random_algorithm(
        random_state: int,
        replacement: bool,
        total_num: int,
        sample_num: int,
    ):
        random_ids = []
        if not replacement:
            random.seed(random_state)
            random_ids = random.sample(range(0, total_num - 1), sample_num)
        else:
            rand_num = random.Random(random_state)
            for _ in range(sample_num):
                randnum = rand_num.randint(0, total_num - 1)
                random_ids.append(randnum)
        random_ids.sort()
        return random_ids


class SystemSampleAlgorithm(SampleAlgorithm):
    def __init__(
        self,
        in_df: VDataFrame,
        total_num: int,
        sample_num_option,
        sample_num_option_number,
        sample_num_option_frac,
    ):
        super().__init__(
            in_df,
            total_num,
        )
        self.sample_num = calculate_sample_number(
            sample_num_option,
            sample_num_option_number,
            sample_num_option_frac,
            # System sample forbid replacement
            False,
            total_num,
        )

    def _algorithm(
        self,
        _random_state: int,
        _observe_feature: str,
    ):
        device0 = next(iter(self.in_df.partitions))
        system_ids = device0(SystemSampleAlgorithm._system_algorithm)(
            self.total_num, self.sample_num
        )
        return reveal(system_ids), []

    @staticmethod
    def _system_algorithm(total_num: int, sample_num: int):
        system_ids = []
        interval = round(total_num / sample_num)
        for idx in range(0, total_num, interval):
            system_ids.append(idx)
        system_ids.sort()
        return system_ids


class StratifySampleAlgorithm(SampleAlgorithm):
    def __init__(
        self,
        in_df: VDataFrame,
        total_num: int,
        sample_num_option,
        sample_num_option_number,
        sample_num_option_frac,
        stratify_replacements: list[bool],
        quantiles: list[float],
        weights: list[float],
    ):
        assert (
            len(quantiles) > 0
        ), "quantiles is necessary for Stratify sample, but get null"
        assert len(quantiles) + 1 == len(
            stratify_replacements
        ), f"len(quantiles) + 1 must equal len(replacements), but got len(quantile):{len(quantiles)}, len(replacements):{len(stratify_replacements)}"

        if len(weights) > 0:
            assert len(weights) == len(
                stratify_replacements
            ), f"len(weights) must equal len(replacements), but got len(weights):{len(weights)}, len(replacements):{len(stratify_replacements)}"

        super().__init__(
            in_df,
            total_num,
        )
        self.sample_num = calculate_sample_number(
            sample_num_option,
            sample_num_option_number,
            sample_num_option_frac,
            True,
            total_num,
        )
        self.replacements = stratify_replacements
        self.quantiles = quantiles
        self.weights = weights

    def _algorithm(
        self,
        random_state: int,
        observe_feature: str,
    ):
        feature_owner = None
        for device, cols in self.in_df.partition_columns.items():
            if observe_feature in cols:
                feature_owner = device
                break
        assert (
            feature_owner is not None
        ), f"Feature owner is not found, {observe_feature} not in {self.in_df.columns}"

        result = feature_owner(StratifySampleAlgorithm._stratify_algorithm)(
            random_state,
            self.replacements,
            observe_feature,
            self.in_df.partitions[feature_owner].data,
            self.quantiles,
            self.total_num,
            self.sample_num,
            self.weights,
        )
        return reveal(result)

    @staticmethod
    def _split_buckets(df: pd.DataFrame, observe_feature: str, quantiles: list[float]):
        feature_series = df[observe_feature]
        bucket_idxs = [[] for _ in range(len(quantiles) + 1)]
        for i, val in feature_series.items():
            idx = bisect.bisect_left(quantiles, float(val))
            bucket_idxs[idx].append(i)
        return bucket_idxs

    @staticmethod
    def _stratify_algorithm(
        random_state: int,
        replacements: list[bool],
        observe_feature: str,
        df: pd.DataFrame,
        quantiles: list[float],
        total_num: int,
        sample_num: int,
        weights: list[float],
    ):
        bucket_idxs = StratifySampleAlgorithm._split_buckets(
            df, observe_feature, quantiles
        )
        random_ids = []
        summary_report = (total_num, sample_num, sample_num / total_num)
        reports = [summary_report]
        rand_num = random.Random(random_state)
        for i in range(len(bucket_idxs)):
            bucket_size = len(bucket_idxs[i])
            if len(weights) > 0:
                target_size = round(sample_num * weights[i])
            else:
                target_size = round(bucket_size * sample_num / total_num)
            assert target_size > 0, f"bucket {i} target_size is 0"
            reports.append((bucket_size, target_size, target_size / bucket_size))

            if replacements[i]:
                for _ in range(target_size):
                    randnum = rand_num.randint(0, sys.maxsize)
                    random_ids.append(bucket_idxs[i][randnum % bucket_size])
            else:
                assert (
                    target_size <= bucket_size
                ), f"The data in bucket {i} is not enough for sample, expect {target_size} ,but bucket have {bucket_size}, please reset replacement or bucket weights"
                random.seed(random_state)
                permute_vec = bucket_idxs[i]
                random.shuffle(permute_vec)
                random_ids.extend(permute_vec[:target_size])

        random_ids.sort()
        return (random_ids, reports)
