# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random

import torch
import torchvision.utils as tvu
import copy

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


class GuidedDiffusion(torch.nn.Module):
    def __init__(self, args, config, device, model_dir):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(self.config.model)
        print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(os.path.join(model_dir, '256x256_diffusion_uncond.pt'), map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)

        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)

    def diffusion_forward(self, x_0, t, noise):
        alpha_bar = (1 - self.betas).cumprod(dim=0)
        std_dev = torch.sqrt(1 - alpha_bar[t])
        mean_multiplier = torch.sqrt(alpha_bar[t])
        std_dev = std_dev[:, None, None, None].to(x_0.device)
        diffused_img = mean_multiplier.to(x_0.device) * x_0 + std_dev * noise.to(x_0.device)
        return diffused_img

    def image_editing_sample(self, img, bs_id=0, tag=None):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            batch_size = img.shape[0]

            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))

            assert img.ndim == 4, img.ndim
            img = img.to(self.device)
            x0 = img

            xs = []
            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=self.device)

                    x = self.diffusion.p_sample(self.model, x, t,
                                                clip_denoised=True,
                                                denoised_fn=None,
                                                cond_fn=None,
                                                model_kwargs=None)["sample"]

                x0 = x

                xs.append(x0)

            return torch.cat(xs, dim=0)