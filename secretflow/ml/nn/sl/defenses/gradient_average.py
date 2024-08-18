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
    This is only used when the model is biclassification.
    """

    def __init__(self, backend: str = "tensorflow", exec_device='cpu', **kwargs):
        self.backend = backend.lower()
        self.exec_device = exec_device
        super().__init__(**kwargs)

    def on_fuse_backward_end(self):
        def average_gradient(worker, backend, exec_device):

            gradient = worker._gradient

            def _avg_grad(g):

                row_averages = np.mean(g, axis=0)
                average_data = np.tile(row_averages, g.shape[0]).reshape(g.shape)
                return average_data

            if backend == "tensorflow":
                import tensorflow as tf

                gradient = (
                    [_avg_grad(g.numpy()) for g in gradient]
                    if isinstance(gradient, list)
                    else _avg_grad(gradient.numpy())
                )
                gradient_tensor = (
                    [tf.convert_to_tensor(g) for g in gradient]
                    if isinstance(gradient, list)
                    else tf.convert_to_tensor(gradient)
                )
            else:
                import torch

                gradient = (
                    [_avg_grad(g.cpu().numpy()) for g in gradient]
                    if isinstance(gradient, list)
                    else _avg_grad(gradient.detach().cpu().numpy())
                )
                gradient_tensor = (
                    [torch.tensor(g).to(exec_device) for g in gradient]
                    if isinstance(gradient, list)
                    else torch.tensor(gradient).to(exec_device)
                )
            if gradient is None:
                raise Exception("No gradient received from label party.")

            worker._gradient = gradient_tensor

        self._workers[self.device_y].apply(
            average_gradient, self.backend, self.exec_device
        )
