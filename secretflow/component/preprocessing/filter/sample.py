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
from dataclasses import dataclass

import pandas as pd

from secretflow.component.core import (
    Component,
    CompVDataFrame,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    Reporter,
    UnionGroup,
    VTable,
    register,
)
from secretflow.data import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import reveal

# 随机采样
# if replacement==true 执行放回采样
# if replacement==false执行不放回采样
RANDOM_SAMPLE = "random"
# 系统等距抽样
SYSTEM_SAMPLE = "system"
# 分层抽样
STRATIFY_SAMPLE = "stratify"


@dataclass
class Random:
    frac: float = Field.attr(
        desc="Proportion of the dataset to sample in the set. The fraction should be larger than 0.",
        default=0.8,
        bound_limit=Interval.open(0.0, 10000.0),
    )
    random_state: int = Field.attr(
        desc="Specify the random seed of the shuffling.",
        default=1024,
        bound_limit=Interval.open(0, None),
    )
    replacement: bool = Field.attr(
        desc="If true, sampling with replacement. If false, sampling without replacement.",
        default=False,
    )


@dataclass
class System:
    frac: float = Field.attr(
        desc="Proportion of the dataset to sample in the set. The fraction should be larger than 0 and less than or equal to 0.5.",
        default=0.2,
        bound_limit=Interval.open_closed(0.0, 0.5),
    )


@dataclass
class Stratify:
    frac: float = Field.attr(
        desc="Proportion of the dataset to sample in the set. The fraction should be larger than 0.",
        default=0.8,
        bound_limit=Interval.open(0.0, 10000.0),
    )
    random_state: int = Field.attr(
        desc="Specify the random seed of the shuffling.",
        default=1024,
        bound_limit=Interval.open(0, None),
    )
    observe_feature: str = Field.attr(desc="stratify sample observe feature.")
    replacements: list[bool] = Field.attr(
        desc="If true, sampling with replacement. If false, sampling without replacement."
    )
    quantiles: list[float] = Field.attr(
        desc="stratify sample quantiles",
        list_limit=Interval.closed(1, 1000),
    )
    weights: list[float] = Field.attr(
        desc="stratify sample weights",
        default=[],
        bound_limit=Interval.open(0.0, 1.0),
    )


@dataclass
class Algorithm(UnionGroup):
    random: Random = Field.struct_attr(desc="Random sample.")
    system: System = Field.struct_attr(desc="system sample.")
    stratify: Stratify = Field.struct_attr(desc="stratify sample.")


@register(domain='data_filter', version='1.0.0')
class Sample(Component):
    '''
    Sample data set.
    '''

    sample_algorithm: Algorithm = Field.union_attr(
        desc="sample algorithm and parameters"
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output sampled dataset.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output sample report",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        input = VTable.from_distdata(self.input_ds)

        # TODO: streaming, avoid load all data into mem.
        # FIXME: avoid to_pandas, use pa.Table
        input_df = ctx.load_table(input).to_pandas(check_null=False)

        pyus = list(input_df.partitions.keys())
        part0 = input_df.partitions[pyus[0]]
        assert isinstance(input_df, VDataFrame), "input_df must be VDataFrame"

        # get row number
        total_num = part0.shape[0]
        assert total_num > 0, "total_num must greater than 0"
        alg = self.sample_algorithm
        alg_name = alg.get_selected()
        sample_algorithm_obj = SampleAlgorithmFactory.create(
            input_df,
            total_num,
            alg_name,
            alg.random,
            alg.system,
            alg.stratify,
        )
        sample_df, report_results = sample_algorithm_obj.perform_sample()
        out_df = CompVDataFrame.from_pandas(sample_df, input.schemas)
        ctx.dump_to(out_df, self.output_ds)

        r = Reporter(name="reports", desc="stratify sample report")
        self.build_report(r, alg_name, report_results)
        r.dump_to(self.report, self.input_ds.system_info)

    @staticmethod
    def build_report(r: Reporter, algorithm: str, results):
        if algorithm != STRATIFY_SAMPLE:
            return
        for i, result in enumerate(results):
            if i == 0:
                (name, desc) = ("采样结果表", "total sample info")
            else:
                (name, desc) = (f'bucket_{i - 1}', 'bucket sample info')
            items = {
                "num_before_sample": int(result[0]),
                "num_after_sample": int(result[1]),
                "sample_rate": float(result[2]),
            }
            r.add_tab(items, name=name, desc=desc)


def calculate_sample_number(frac: float, replacement: bool, total_num: int):
    sample_num = round(frac * total_num)

    if sample_num > total_num and not replacement:
        raise ValueError(
            f"Replacement has to be set to True when sample number {sample_num} is larger than dataset size {total_num}."
        )

    return sample_num


class SampleAlgorithmFactory:
    @staticmethod
    def create(
        input_df: VDataFrame,
        total_num: int,
        algorithm: str,
        random: Random,
        system: System,
        stratify: Stratify,
    ):
        if algorithm == RANDOM_SAMPLE:
            return RandomSampleAlgorithm(
                input_df,
                total_num,
                random.frac,
                random.random_state,
                random.replacement,
            )
        elif algorithm == SYSTEM_SAMPLE:
            return SystemSampleAlgorithm(
                input_df,
                total_num,
                system.frac,
            )
        elif algorithm == STRATIFY_SAMPLE:
            return StratifySampleAlgorithm(
                input_df,
                total_num,
                stratify.frac,
                stratify.random_state,
                stratify.observe_feature,
                stratify.replacements,
                stratify.quantiles,
                stratify.weights,
            )
        else:
            raise AssertionError(
                f'sample_algorithm must be one of [{RANDOM_SAMPLE}, {SYSTEM_SAMPLE}, {STRATIFY_SAMPLE}], but got {algorithm}'
            )


class SampleAlgorithm:
    def __init__(
        self,
        in_df: VDataFrame,
        total_num: int,
    ):
        self.in_df = in_df
        self.total_num = total_num

    def perform_sample(self):
        random_ids, report_results = self._algorithm()
        return self._filter_data(random_ids), report_results

    def _algorithm(self):
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
        frac: float,
        random_state: int,
        replacement: bool,
    ):
        super().__init__(
            in_df,
            total_num,
        )
        self.replacement = replacement
        self.random_state = random_state
        self.sample_num = calculate_sample_number(
            frac,
            self.replacement,
            total_num,
        )

    def _algorithm(
        self,
    ):
        device0 = next(iter(self.in_df.partitions))
        random_ids = device0(RandomSampleAlgorithm._random_algorithm)(
            self.random_state, self.replacement, self.total_num, self.sample_num
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
        frac: float,
    ):
        super().__init__(
            in_df,
            total_num,
        )
        self.sample_num = calculate_sample_number(
            frac,
            # System sample forbid replacement
            False,
            total_num,
        )

    def _algorithm(self):
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
        frac: float,
        random_state: int,
        observe_feature,
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
            epsilon = 1e-5
            assert (
                abs(sum(weights) - 1.0) < epsilon
            ), f"sum of weights must be 1.0, but got {sum(weights)}, weights len: {len(weights)}"

        super().__init__(
            in_df,
            total_num,
        )
        self.sample_num = calculate_sample_number(
            frac,
            True,
            total_num,
        )
        self.random_state = random_state
        self.observe_feature = observe_feature
        self.replacements = stratify_replacements
        self.quantiles = quantiles
        self.weights = weights

    def _algorithm(self):
        feature_owner = None
        for device, cols in self.in_df.partition_columns.items():
            if self.observe_feature in cols:
                feature_owner = device
                break
        assert (
            feature_owner is not None
        ), f"Feature owner is not found, {self.observe_feature} not in {self.in_df.columns}"

        result = feature_owner(StratifySampleAlgorithm._stratify_algorithm)(
            self.random_state,
            self.replacements,
            self.observe_feature,
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
            assert bucket_size > 0, f"bucket {i} bucket_size is 0"
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
