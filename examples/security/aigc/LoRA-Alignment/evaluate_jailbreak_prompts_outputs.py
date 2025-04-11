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

from utils.evaluator import RoBERTaEvaluator
from utils.test import PATH_GPTFUZZ
from utils.arguments import RESULTS_ROOT_PATH
from datasets import Dataset
from utils.logging import logger
import os


if __name__ == "__main__":

    evaluator_path = PATH_GPTFUZZ
    device = "cuda"
    batch_size = 500
    save_path = f"{RESULTS_ROOT_PATH}/evaluated_jailbreak_attempts"
    dataset_path = f"{RESULTS_ROOT_PATH}/jailbreak_evaluations"

    model_name_or_path = os.path.expanduser(evaluator_path)
    robert_evaluator = RoBERTaEvaluator(model_name_or_path, device)

    dataset = Dataset.load_from_disk(dataset_path)

    labels = []
    outputs = dataset["outputs"]

    for idx in range(0, len(dataset), batch_size):
        logger.info(
            f"Evaluating batch from index {idx} to {min(idx + batch_size - 1, len(dataset))}:"
        )

        batch = outputs[idx : idx + batch_size]
        predicaitons = robert_evaluator.predict(batch)

        labels.extend(predicaitons)

    dataset.add_column("labels", labels)
    dataset.save_to_disk(save_path)
