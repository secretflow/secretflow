# Copyright xuxiaoyang, ywenrou123@163.com
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

from typing import List

import numpy as np

from secretflow import PYU, DeviceObject, PYUObject
from secretflow.security import SecureAggregator


class StatefulFedGenAggregator(SecureAggregator):
    """
    StatefulFedGenAggregator: Extends SecureAggregator to handle aggregation of model parameters
    and training of the generator model in a federated learning context.
    """

    def __init__(
        self, device, participants: List[PYU], server_actor, fxp_bits: int = 18
    ):
        super().__init__(device, participants, fxp_bits)
        self.server_actor = server_actor

    def average(self, data: List[PYUObject], axis=None, weights=None):
        """
        Overrides the average method to perform parameter aggregation and generator model training.

        Args:
            data (List[PYUObject]): A list of participant model parameters.
            axis: The axis along which the average operation is performed.
            weights (optional): Weights to use during aggregation, which can influence the importance of each participant.

        Returns:
            avg_model_params: The aggregated model parameters if no generator training is needed.
            If generator training is involved, returns a dictionary with updated generator and model parameters.
        """

        def _get_label_counts(client_result):
            """Extracts the label count dictionary from the client's result."""
            return client_result["label_counts_dict"]

        def _get_num_samples(client_result):
            """Extracts the number of samples from the client's result."""
            return client_result["num_sample"]

        # Default aggregation without using weights
        _num_simple = None
        avg_model_params = super().average(data, axis, _num_simple)

        # If weights are provided, perform weighted aggregation and generator training
        if weights is not None and isinstance(weights, (list, tuple, np.ndarray)):
            synthetic_data_result = self.server_actor.generate_synthetic_data()
            _worker_label_counts = []
            _user_results = []
            _num_simple = []

            # Ensure the weights list length matches the number of participants
            assert len(weights) == len(
                data
            ), f'Length of the weights does not match the data: {len(weights)} vs {len(data)}.'

            for i, w in enumerate(weights):
                if isinstance(w, DeviceObject):
                    # Ensure each weight is associated with the correct device
                    assert (
                        w.device == data[i].device
                    ), 'Device of weight does not match the corresponding data device.'

                    # Extract label counts and user results
                    _worker_label_counts.append(
                        self._device(_get_label_counts)(w.to(self._device))
                    )
                    _user_results.append(
                        self.server_actor.get_penultimate_layer_output(
                            data[i].to(self._device), synthetic_data_result
                        )
                    )
                    _num_simple.append(w.device(_get_num_samples)(w))

            # Train the generator model
            self.server_actor.train_generator(
                _user_results,
                _worker_label_counts,
                synthetic_data_result,
                avg_model_params,
            )

            # Return updated generator and model parameters
            return self._device(lambda x: x)(
                {
                    "generator_params": self.server_actor.get_generator_weights(),
                    "model_params": avg_model_params,
                }
            )

        # If no weights are provided, return the average model parameters
        return avg_model_params
