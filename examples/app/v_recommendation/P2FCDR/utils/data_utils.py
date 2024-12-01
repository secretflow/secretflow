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

import numpy as np
import torch
from dataset import RecDataset


def load_ratings_dataset(args):
    client_train_datasets = []
    client_valid_datasets = []
    client_test_datasets = []
    for domain in args.domains:
        model = args.method.replace("Fed", "")

        train_dataset = RecDataset(
            args, domain, model, mode="train", load_prep=args.load_prep
        )
        valid_dataset = RecDataset(
            args, domain, model, mode="valid", load_prep=args.load_prep
        )
        test_dataset = RecDataset(
            args, domain, model, mode="test", load_prep=args.load_prep
        )

        client_train_datasets.append(train_dataset)
        client_valid_datasets.append(valid_dataset)
        client_test_datasets.append(test_dataset)
    return client_train_datasets, client_valid_datasets, client_test_datasets
