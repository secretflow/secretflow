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

import logging
import re

import numpy as np
import pandas as pd
import pytest
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow_spec.v1.report_pb2 import Report

from secretflow.component.core import (
    DistDataType,
    VTable,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.preprocessing.filter.sample import (
    RANDOM_SAMPLE,
    STRATIFY_SAMPLE,
    SYSTEM_SAMPLE,
    Random,
    RandomSampleAlgorithm,
    SampleAlgorithmFactory,
    Stratify,
    StratifySampleAlgorithm,
    System,
    SystemSampleAlgorithm,
    calculate_sample_number,
)

RANDOM_STATE = 1234


def test_calculate_sample_num():
    total_num = 100

    frac = 0.5
    sample_num = calculate_sample_number(frac, False, total_num)
    assert sample_num == round(total_num * frac)

    frac = 2
    sample_num = frac * total_num
    with pytest.raises(
        Exception,
        match=f"Replacement has to be set to True when sample number {sample_num} is larger than dataset size {total_num}.",
    ):
        calculate_sample_number(frac, False, total_num)


def test_illegal_sample():
    df_alice = pd.DataFrame({'a': [i for i in range(100)]})
    total_num = 100
    illegal_sample_method = "Illegal"

    expected_msg = f"sample_algorithm must be one of [random, system, stratify], but got {illegal_sample_method}"
    with pytest.raises(AssertionError, match=re.escape(expected_msg)):
        SampleAlgorithmFactory.create(
            df_alice, total_num, illegal_sample_method, None, None, None
        )


def test_random_sample():
    print("test_random_sample")
    total_num = 100
    sample_num = 10

    df_alice = pd.DataFrame({'a': [i for i in range(100)]})
    alg = SampleAlgorithmFactory.create(
        df_alice,
        total_num,
        RANDOM_SAMPLE,
        Random(frac=0.8, random_state=RANDOM_STATE, replacement=True),
        None,
        None,
    )
    assert isinstance(alg, RandomSampleAlgorithm)

    # replacement
    replacement_random_ids = RandomSampleAlgorithm._random_algorithm(
        RANDOM_STATE, True, total_num, sample_num
    )
    reference_replacement_random_ids = [0, 4, 10, 11, 14, 56, 74, 85, 88, 99]
    assert replacement_random_ids == reference_replacement_random_ids
    print(replacement_random_ids)
    logging.info(f'random_ids:{replacement_random_ids}')

    # no replacement
    noreplacement_random_ids = RandomSampleAlgorithm._random_algorithm(
        RANDOM_STATE, False, total_num, sample_num
    )
    noreplacement_reference_random_ids = [0, 4, 10, 11, 12, 14, 56, 74, 85, 88]
    assert noreplacement_random_ids == noreplacement_reference_random_ids

    # non equal
    fail_random_ids = [19, 58, 98, 22, 90, 50, 93, 44, 55, 64]
    assert noreplacement_random_ids != fail_random_ids


def test_system_sample_rounding_error():
    df_alice = pd.DataFrame({'a': [i for i in range(99)]})
    total_num = 99

    alg = SampleAlgorithmFactory.create(
        df_alice, total_num, SYSTEM_SAMPLE, None, System(frac=0.5), None
    )
    assert isinstance(alg, SystemSampleAlgorithm)

    random_ids = SystemSampleAlgorithm._system_algorithm(total_num, alg.sample_num)
    reference_random_ids = []
    for i in range(0, total_num, 2):
        reference_random_ids.append(i)

    assert len(random_ids) == len(reference_random_ids)


def test_system_sample():
    df_alice = pd.DataFrame({'a': [i for i in range(100)]})
    total_num = 100

    alg = SystemSampleAlgorithm(df_alice, total_num, 0.1)

    random_ids = SystemSampleAlgorithm._system_algorithm(total_num, alg.sample_num)
    reference_random_ids = []
    for i in range(0, total_num, 10):
        reference_random_ids.append(i)

    assert len(random_ids) == len(reference_random_ids)


def test_stratify_sample():
    df_alice = pd.DataFrame(
        {
            "id": [
                "K1",
                "K9",
                "k10",
                "K11",
                "K5",
                "K14",
                "k4",
                "K17",
                "K18",
                "k15",
                "K16",
                "K12",
                "K2",
                "k3",
                "K19",
                "K6",
                "K7",
                "k8",
                "K13",
            ],
            "amount": [
                1,
                9,
                10,
                11,
                5,
                14,
                4,
                17,
                18,
                15,
                16,
                12,
                2,
                3,
                19,
                6,
                7,
                8,
                13,
            ],
        }
    )
    total_num = 19
    sample_num = 13

    quantiles = [5.3, 14.7]
    weights = [0.3, 0.3, 0.4]
    alg = SampleAlgorithmFactory.create(
        df_alice,
        total_num,
        STRATIFY_SAMPLE,
        None,
        None,
        Stratify(
            frac=0.8,
            random_state=RANDOM_STATE,
            observe_feature='',
            replacements=[False, False, False],
            quantiles=quantiles,
            weights=weights,
        ),
    )
    assert isinstance(alg, StratifySampleAlgorithm)

    bucket_idxes = StratifySampleAlgorithm._split_buckets(df_alice, 'amount', quantiles)
    assert bucket_idxes == [
        [0, 4, 6, 12, 13],
        [1, 2, 3, 5, 11, 15, 16, 17, 18],
        [7, 8, 9, 10, 14],
    ]

    replacements = [False, False, False]
    weights = []
    random_ids, report = alg._stratify_algorithm(
        RANDOM_STATE,
        replacements,
        'amount',
        df_alice,
        quantiles,
        total_num,
        sample_num,
        weights,
    )
    assert random_ids == [3, 4, 5, 6, 8, 9, 11, 13, 14, 15, 16, 18]
    assert report == [
        (19, 13, 0.6842105263157895),
        (5, 3, 0.6),
        (9, 6, 0.6666666666666666),
        (5, 3, 0.6),
    ]

    replacements = [True, True, True]
    random_ids, _ = alg._stratify_algorithm(
        RANDOM_STATE,
        replacements,
        'amount',
        df_alice,
        quantiles,
        total_num,
        sample_num,
        weights,
    )
    assert random_ids == [0, 1, 5, 5, 6, 7, 7, 10, 11, 13, 15, 16]

    replacements = [False, False, False]
    weights = [0.3, 0.3, 0.4]
    random_ids, _ = alg._stratify_algorithm(
        RANDOM_STATE,
        replacements,
        'amount',
        df_alice,
        quantiles,
        total_num,
        sample_num,
        weights,
    )
    assert random_ids == [0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 18]

    replacements = [True, True, True]
    random_ids, _ = alg._stratify_algorithm(
        RANDOM_STATE,
        replacements,
        'amount',
        df_alice,
        quantiles,
        total_num,
        sample_num,
        weights,
    )
    assert random_ids == [0, 4, 5, 6, 7, 7, 8, 9, 10, 11, 13, 15, 16]

    # illegal weight, elements num < required sample_num
    replacements = [False, False, False]
    weights = [0.5, 0.3, 0.3]
    illegal_idx = 0
    bucket0_size = round(len(bucket_idxes[illegal_idx]))
    target_sample_num = round(sample_num * weights[illegal_idx])
    with pytest.raises(
        AssertionError,
        match=f"The data in bucket {illegal_idx} is not enough for sample, expect {target_sample_num} ,but bucket have {bucket0_size}, please reset replacement or bucket weights",
    ):
        alg._stratify_algorithm(
            RANDOM_STATE,
            replacements,
            'amount',
            df_alice,
            quantiles,
            total_num,
            sample_num,
            weights,
        )


def test_stratify_sample_illegal():
    df_alice = pd.DataFrame()

    # quantiles.size == 0
    quantiles = []
    weights = [0.3, 0.3, 0.4]
    with pytest.raises(
        AssertionError,
        match="quantiles is necessary for Stratify sample, but get null",
    ):
        StratifySampleAlgorithm(
            df_alice, 100, 0.1, RANDOM_STATE, '', [False, False], quantiles, weights
        )

    # quantiles.size + 1 != replacements.size
    quantiles = [3.3]
    replacements = [True, True, True]

    expected_msg = f"len(quantiles) + 1 must equal len(replacements), but got len(quantiles):{len(quantiles)}, len(replacements):{len(weights)}"
    with pytest.raises(AssertionError, match=re.escape(expected_msg)):
        StratifySampleAlgorithm(
            df_alice, 100, 0.1, RANDOM_STATE, '', replacements, quantiles, weights
        )

    with pytest.raises(
        AssertionError,
        match=f"sum of weights must be 1.0, but got 1.2000000000000002",
    ):
        StratifySampleAlgorithm(
            df_alice,
            100,
            0.8,
            RANDOM_STATE,
            '',
            [True, True, True],
            [1, 2],
            [0.4, 0.4, 0.4],
        )

    with pytest.raises(
        AssertionError,
        match=f"sum of weights must be 1.0, but got 0.9",
    ):
        StratifySampleAlgorithm(
            df_alice,
            100,
            0.8,
            RANDOM_STATE,
            '',
            [True, True, True],
            [1, 2],
            [0.4, 0.4, 0.1],
        )

    # empty weights ok
    StratifySampleAlgorithm(
        df_alice, 100, 0.8, RANDOM_STATE, '', [True, True, True], [1, 2], []
    )


@pytest.mark.mpc
def test_sample_vertical(sf_production_setup_comp):
    alice_input_path = "sample_filter/alice.csv"
    bob_input_path = "sample_filter/bob.csv"
    sample_output_path = "sample_filter/sample.csv"
    report_path = "sample_filter/model.report"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id1": [1, 2, 3, 4, 5, 6],
                "a1": ["K5", "K1", "K?", "K6", "cc", "L"],
            }
        )
        df_alice.to_csv(
            storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        df_bob = pd.DataFrame(
            {
                "id2": [1, 2, 3, 4, 5, 6],
                "b1": [10.2, 20.5, 1.0, -0.4, -3.2, 4.5],
            }
        )
        df_bob.to_csv(
            storage.get_writer(bob_input_path),
            index=False,
        )

    param = build_node_eval_param(
        domain="data_filter",
        name="sample",
        version="1.0.0",
        attrs={
            'sample_algorithm': 'stratify',
            'sample_algorithm/stratify/frac': 0.8,
            'sample_algorithm/stratify/random_state': 1234,
            'sample_algorithm/stratify/observe_feature': "id1",
            'sample_algorithm/stratify/replacements': [False, False],
            'sample_algorithm/stratify/quantiles': [3.1],
            'sample_algorithm/stratify/weights': [0.4, 0.6],
        },
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                ],
            )
        ],
        output_uris=[sample_output_path, report_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["str"],
                ids=["id2"],
                feature_types=["float32"],
                features=["b1"],
            ),
            TableSchema(
                id_types=["str"],
                ids=["id1"],
                feature_types=["str"],
                features=["a1"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True,
    )

    logging.info(f"stratify sample {res['tracer_report']}")
    comp_ret = Report()
    res = res['eval_result']
    assert len(res.outputs) == 2
    res.outputs[1].meta.Unpack(comp_ret)
    logging.info(comp_ret)

    sample_data = VTable.from_distdata(res.outputs[0], columns=["id1", "id2"])

    if self_party == "alice":
        ds_alice = orc.read_table(
            storage.get_reader(sample_data.get_party("alice").uri)
        ).to_pandas()
        np.testing.assert_equal(ds_alice.shape[0], 5)
        assert list(ds_alice["id1"]) == ["1", "3", "4", "5", "6"]

    if self_party == "bob":
        ds_bob = orc.read_table(
            storage.get_reader(sample_data.get_party("bob").uri)
        ).to_pandas()
        np.testing.assert_equal(ds_bob.shape[0], 5)
        assert list(ds_bob["id2"]) == ["1", "3", "4", "5", "6"]


@pytest.mark.mpc
def test_sample_vertical_replacement(sf_production_setup_comp):
    alice_input_path = "sample_filter/alice.csv"
    bob_input_path = "sample_filter/bob.csv"
    sample_output_path = "sample_filter/sample.csv"
    report_path = "sample_filter/model.report"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id1": [1, 2, 3, 4, 5, 6],
                "a1": ["K5", "K1", "K?", "K6", "cc", "L"],
            }
        )
        df_alice.to_csv(
            storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        df_bob = pd.DataFrame(
            {
                "id2": [1, 2, 3, 4, 5, 6],
                "b1": [10.2, 20.5, 10.1, -0.4, -3.2, 4.5],
            }
        )
        df_bob.to_csv(
            storage.get_writer(bob_input_path),
            index=False,
        )

    param = build_node_eval_param(
        domain="data_filter",
        name="sample",
        version="1.0.0",
        attrs={
            'sample_algorithm': 'random',
            'sample_algorithm/random/frac': 0.8,
            'sample_algorithm/random/random_state': 1234,
            'sample_algorithm/random/replacement': True,
        },
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                ],
            )
        ],
        output_uris=[sample_output_path, report_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["str"],
                ids=["id2"],
                feature_types=["float32"],
                features=["b1"],
            ),
            TableSchema(
                id_types=["str"],
                ids=["id1"],
                feature_types=["str"],
                features=["a1"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    sample_data = VTable.from_distdata(res.outputs[0])

    if self_party == "alice":
        ds_alice = orc.read_table(
            storage.get_reader(sample_data.parties["alice"].uri)
        ).to_pandas()
        np.testing.assert_equal(ds_alice.shape[0], 5)
        assert list(ds_alice["id1"]) == ["1", "1", "1", "4", "5"]

    if self_party == "bob":
        ds_bob = orc.read_table(
            storage.get_reader(sample_data.parties["bob"].uri)
        ).to_pandas()
        np.testing.assert_equal(ds_bob.shape[0], 5)
        assert list(ds_bob["id2"]) == ["1", "1", "1", "4", "5"]


@pytest.mark.mpc
def test_sample_individual(sf_production_setup_comp):
    alice_input_path = "sample_individual_filter/alice.csv"
    sample_output_path = "sample_individual_filter/sample.csv"
    report_path = "sample_filter/model.report"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == 'alice':
        df_alice = pd.DataFrame(
            {
                "id1": [1, 2, 3, 4, 5, 6],
                "a1": ["K5", "K1", "KN", "K6", "cc", "L"],
            }
        )
        df_alice.to_csv(
            storage.get_writer(alice_input_path),
            index=False,
        )

    param = build_node_eval_param(
        domain="data_filter",
        name="sample",
        version="1.0.0",
        attrs={
            'sample_algorithm': 'random',
            'sample_algorithm/random/frac': 0.8,
            'sample_algorithm/random/random_state': 1234,
            'sample_algorithm/random/replacement': False,
        },
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                ],
            )
        ],
        output_uris=[sample_output_path, report_path],
    )

    meta = IndividualTable(
        schema=TableSchema(
            id_types=["str"],
            ids=["id1"],
            feature_types=["str"],
            features=["a1"],
        )
    )
    param.inputs[0].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    sample_data = VTable.from_distdata(res.outputs[0])

    if self_party == "alice":
        ds_alice = orc.read_table(
            storage.get_reader(sample_data.parties["alice"].uri)
        ).to_pandas()
        np.testing.assert_equal(ds_alice.shape[0], 5)
        assert list(ds_alice["id1"]) == ["1", "2", "3", "4", "5"]
