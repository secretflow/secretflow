# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field

import random
@dataclass
class DatasetArgs:
    CONFIG_KEY = "dataset_args"

    dataset_path: str = field(default="../src/extern/echr", metadata={
        "help": "The path to the dataset. ",
        "choices": ["../src/extern/echr", "../src/extern/enron"]
    })

    dataset_mode: str = field(default="undefended", metadata={
        "help": "The mode for this dataset. "
                "undefended -> No protection on the dataset"
                "scrubbed -> Replace all PII with a <MASK> token",
        "choices": ["undefended", "scrubbed"]
    })

    split: str = field(default="train", metadata={
        "help": "the split this dataset is loading. "
    })

    sample_duplication_rate: int = field(default=1, metadata={
        "help": "Number of times to repeat a sample"
    })

    limit_dataset_size: int = field(default=1_000_000, metadata={
        "help": "Limit the number of samples to load for this dataset"
    })

    def get_dataset_name(self) -> str:
        return self.dataset_path.split("/")[-1]

    def cache_dir(self) -> str:
        return f"/tmp/{random.randint(0, 99**10)}" if self.dataset_mode == "mixed" else None

    def is_scrubbed(self) -> bool:
        return "scrubbed" in self.dataset_mode

    def copy(self):
        return deepcopy(self)

    def set_split(self, split: str) -> 'DatasetArgs':
        dataset_args2 = self.copy()
        dataset_args2.split = split
        return dataset_args2

    def hash(self, suffix="") -> str:
        """ Computes a persistent hash of the dataset (i) path, (ii) mode and (iii) split and (iv) the sample
        duplication rate. """
        return hashlib.sha1(f'{self.dataset_path}_{self.dataset_mode}{self.split}{self.sample_duplication_rate}{suffix}'.encode('utf-8')).hexdigest()

    def __iter__(self):
        return iter((self.dataset_path, self.dataset_mode))