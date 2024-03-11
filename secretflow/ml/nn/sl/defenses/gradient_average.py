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

import numpy as np

from secretflow.ml.nn.callbacks.callback import Callback


class GradientAverage(Callback):
    """
    Implemention of gradient averaging to defense against label leaking attack.
    This callback will average the gradient of each party by batch before sending it back to its own party.
    """

    def __init__(self, backend: str = "tensorflow", **kwargs):
        self.backend = backend.lower()

        super().__init__(**kwargs)

    def on_fuse_backward_end(self):
        def average_gradient(worker):

            gradient = worker._gradient
            if self.backend == "tensorflow":
                gradient = (
                    [g.numpy() for g in gradient]
                    if isinstance(gradient, list)
                    else gradient
                )
            else:
                gradient = (
                    [g.detach().numpy() for g in gradient]
                    if isinstance(gradient, list)
                    else gradient
                )

            if gradient is None:
                raise Exception("No gradient received from label party.")

            nd_data = np.array(gradient)
            row_averages = np.mean(nd_data, axis=0)
            average_data = np.tile(row_averages, nd_data.shape[0]).reshape(
                nd_data.shape
            )
            worker._gradient = average_data

        self._workers[self.device_y].apply(average_gradient)
