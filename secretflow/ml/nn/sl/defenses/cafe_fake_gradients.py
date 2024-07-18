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

class CAFEFakeGradientsMultiClient(Callback):
    """
    The method is designed for against CAFE attack: https://arxiv.org/abs/2110.15122.
    Each local worker randomly generates gradients with the normal distribution N, 
    sorts them and true gradients in descending order, computes the L2-norm distance
    to find the nearest fake gradient, pairs fake gradients with true gradients by
    the sorted order, and uploads the fake gradients to the server.
    """

    def __init__(
        self,
        backend: str = "torch",
        exec_device='cpu',
        noise_scale=1.1,
        tua=47,
        v=1000,
        **kwargs
    ):
        self.backend = backend.lower()
        self.exec_device = exec_device
        self.noise_scale = noise_scale
        self.tua = tua
        self.v = v
        super().__init__(**kwargs)

    def on_fuse_backward_end(self):
        def fake_gradient(worker, sigma, tua=1.1, M=1, v=128):

            gradient = worker._gradient
            fake_gradient = []
            if not isinstance(gradient, list):
                gradient = [gradient]
            M = len(gradient)

            if self.backend == "tensorflow":
                import tensorflow as tf

                pass
                # WIP
            else:
                import torch

                for m in range(M):
                    _gradient = gradient[m]
                    Psi = [
                        torch.normal(mean=0, std=sigma, size=_gradient.shape)
                        for _ in range(v)
                    ]
                    Psi = [torch.sort(g, descending=True)[0] for g in Psi]
                    zeta = torch.argsort(_gradient, descending=True)
                    sorted_gradient = torch.gather(_gradient, 1, zeta)
                    count = 0  
                    while True:
                        min_diff = float("inf")
                        min_psi = None
                        for psi in Psi:

                            diff = torch.norm(psi - sorted_gradient, p=2)
                            if diff < min_diff:
                                min_diff = diff
                                min_psi = psi
                            # print("min_diff", min_diff)
                        count += 1
                        if min_diff <= tua or count > 1:
                            break
                        Psi = [
                            torch.normal(mean=0, std=sigma, size=_gradient.shape)
                            for _ in range(v)
                        ]
                        Psi = [torch.sort(g, descending=True)[0] for g in Psi]
                    psi = min_psi
                    fake_g = torch.zeros_like(_gradient)
                    for i in range(len(zeta)):
                        l = 0
                        for k in zeta[i]:
                            fake_g[i][k] = torch.min(
                                psi[i][l], torch.max(_gradient[i][k], -psi[i][l])
                            )
                            l += 1
                    fake_gradient.append(_gradient)
            assert len(fake_gradient) >= 1
            if len(fake_gradient) == 1:
                worker._gradient = fake_gradient[0]
            else:
                worker._gradient = fake_gradient

        self._workers[self.device_y].apply(
            fake_gradient, sigma=self.noise_scale, tua=self.tua, v=self.v
        )
