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


class max_norm(Callback):
    """
    MaxNorm is a defense method designed to against the label inference attack
    in Vertical Federated Learning and Split Learning: https://arxiv.org/pdf/2102.08504.
    Maxnorm is a heuristic approach for protecting labels by matching the expected squared
    2-norm of each perturbed gradient to the largest squared 2-norm in a mini-batch and
    adding specific noise only along the gradient direction.
    """

    def __init__(self, backend: str = "torch", exec_device='cpu', **kwargs):
        """
        Initializes the max_norm class.

        Args:
            backend (str): The backend framework to be used. Defaults to "torch".
            exec_device (str): The device for execution. Defaults to 'cpu'.
            **kwargs: Additional keyword arguments.
        """
        self.backend = backend.lower()
        self.exec_device = exec_device
        super().__init__(**kwargs)

    def on_fuse_backward_end(self):
        def perturb_gradient(worker):
            """
            Applies perturbation to gradients as a defense mechanism against label inference attacks.

            Args:
                worker: The client instance from which the gradient is accessed.
            """
            gradient = worker._gradient

            assert gradient is not None and len(gradient) > 0

            if self.backend == "tensorflow":
                raise NotImplementedError()
            else:
                import torch

                max_norm = max(
                    [torch.norm(g, p=2) ** 2 for g in gradient]
                )  # Find the maximum norm
                first_dim = (0,)
                perturbed_gradients = torch.empty(
                    first_dim + tuple(gradient[0].shape[1:])
                )

                for g in gradient[0]:
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

            gradient[0] = perturbed_gradients
            worker._gradient = gradient

        self._workers[self.device_y].apply(perturb_gradient)
