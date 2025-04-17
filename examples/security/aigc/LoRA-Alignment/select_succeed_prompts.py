# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import Dataset, DatasetDict
from collections import Counter
from utils.logging import logger
from utils.arguments import RESULTS_ROOT_PATH


if __name__ == "__main__":
    dataset_path = f"{RESULTS_ROOT_PATH}/evaluated_jailbreak_attempts"
    save_path = f"{RESULTS_ROOT_PATH}/succeed_jailbreak_prompts"

    datasets = Dataset.load_from_disk(dataset_path)
    succeed_dataset = datasets.filter(lambda example: example["labels"] == 1)

    succeed_prompts = zip(succeed_dataset["prompts"], succeed_dataset["communities"])
    succeed_prompts = list(set([i for i in succeed_prompts]))

    counted_comms = Counter([i[1] for i in succeed_prompts]).most_common()
    logger.info(counted_comms)
    succeed_comms = set(succeed_dataset["communities"])

    succeed_dataset_deduplicated = Dataset.from_dict(
        {
            "prompts": [item[0] for item in succeed_prompts],
            "communities": [item[1] for item in succeed_prompts],
        }
    )

    dataset_dict = DatasetDict()
    for comm in succeed_comms:
        dataset_dict[comm] = succeed_dataset_deduplicated.filter(
            lambda example: example["communities"] == comm
        )
    dataset_dict.save_to_disk(save_path)
