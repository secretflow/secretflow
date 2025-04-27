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

from secretflow import PYUObject, proxy
import copy


@proxy(PYUObject)
class Server:
    """
    Server class responsible for aggregating representations
    collected from multiple clients (domains).
    """

    def __init__(self, args, dataset_names):
        """
        Initialize Server.

        Args:
            args: Configuration arguments
            dataset_names (list): List of dataset names participating in training
        """
        self.args = args
        self.dataset_names = dataset_names  # Should be a list, not a dictionary

    def aggregate_reps(self, trainer_weights):
        """
        Aggregate shared representations received from all trainers.

        Args:
            trainer_weights (list): List of tensors, each containing a domain's shared representations.

        Returns:
            dict: Mapping from dataset name to its corresponding representation tensor.

        Raises:
            ValueError: If input list size does not match expected number of datasets
        """
        if not isinstance(trainer_weights, list) or len(trainer_weights) != len(
            self.dataset_names
        ):
            raise ValueError(
                f"[Server] trainer_weights should match number of datasets, "
                f"expected {len(self.dataset_names)}, got {len(trainer_weights)}."
            )

        global_reps = {}
        for dataset_name, weight in zip(self.dataset_names, trainer_weights):
            # Use clone+detach to avoid linking computation graphs across trainers
            global_reps[dataset_name] = weight.clone().detach()

        return global_reps
