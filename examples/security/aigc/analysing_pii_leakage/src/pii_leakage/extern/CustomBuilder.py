# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass

import datasets

@dataclass
class CustomBuilder(datasets.BuilderConfig):
    name: str = None
    sample_duplication_rate: int = 1    # number of times a sample is repeated

@dataclass
class CustomECHRBuilder(datasets.BuilderConfig):
    name: str = None
    sample_duplication_rate: int = 1    # number of times a sample is repeated
    shuffle_facts_seed: int = 42

@dataclass
class CustomEnronBuilder(datasets.BuilderConfig):
    name: str = None
    sample_duplication_rate: int = 1    # number of times a sample is repeated
    shuffle_facts_seed: int = 42