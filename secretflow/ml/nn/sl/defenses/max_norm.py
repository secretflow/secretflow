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


from secretflow.ml.nn.callbacks.callback import Callback


class MaxNorm(Callback):
    """
    MaxNorm is a defense method designed to against the label inference attack
    in Vertical Federated Learning and Split Learning: https://arxiv.org/pdf/2102.08504.
    Maxnorm is a heuristic approach for protecting labels by matching the expected squared
    2-norm of each perturbed gradient to the largest squared 2-norm in a mini-batch and
    adding specific noise only along the gradient direction.
    """

    def __init__(self, backend: str = "torch", exec_device='cpu', **kwargs):
        self.backend = backend.lower()
        self.exec_device = exec_device
        super().__init__(**kwargs)

    def on_fuse_backward_end(self):
        def average_gradient(worker):

            gradient = worker._gradient
            if self.backend == "tensorflow":
                import tensorflow as tf

                pass
                # WIP
            else:
                import torch

                max_norm = max(
                    [torch.norm(g, p=2) ** 2 for g in gradient]
                )  # Find the maximum norm
                first_dim = (0,)
                perturbed_gradients = torch.empty(first_dim + tuple(gradient.shape[1:]))

                for g in gradient:
                    norm_g = torch.norm(g, p=2) ** 2
                    if norm_g == 0:
                        perturbed_gradients.append(g)
                        continue
                    sigma_j = (max_norm / norm_g - 1).sqrt().item()  # Calculate σ
                    noise = torch.normal(mean=0, std=sigma_j, size=g.size()).to(
                        g.device
                    )  # Generate Gaussian noise
                    perturbed_g = g + noise  # Add noise
                    perturbed_g = perturbed_g.unsqueeze(0)
                    perturbed_gradients = torch.cat(
                        (perturbed_gradients, perturbed_g), dim=0
                    )

            if gradient is None:
                raise Exception("No gradient received from label party.")

            worker._gradient = perturbed_gradients
            # print("worker._gradient.shape", worker._gradient.shape)

        self._workers[self.device_y].apply(average_gradient)
