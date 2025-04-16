# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random
import numpy as np

import torch
import torchvision.utils as tvu
import torchsde

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from score_sde.losses import get_optimizer
from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
from score_sde import sde_lib


def _extract_into_tensor(arr_or_func, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array or a func.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if callable(arr_or_func):
        res = arr_or_func(timesteps).float()
    else:
        res = arr_or_func.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']


class LDSDE(torch.nn.Module):
    def __init__(self, model, x_init, score_type='guided_diffusion', beta_min=0.1, beta_max=20, N=1000,
                 img_shape=(3, 256, 256), sigma2=0.001, lambda_ld=0.01, eta=5, model_kwargs=None):
        """Construct a Variance Preserving SDE.

        Args:
          model: diffusion model
          score_type: [guided_diffusion, score_sde, ddpm]
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        super().__init__()
        self.model = model
        self.x_init = x_init
        self.sigma2 = sigma2
        self.eta = eta
        self.lambda_ld = lambda_ld  # damping coefficient
        self.score_type = score_type
        self.model_kwargs = model_kwargs
        self.img_shape = img_shape

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t)
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1. / torch.sqrt(1. - self.alphas_cumprod_cont(t))

        self.noise_type = "diagonal"
        self.sde_type = "ito"

        print(f'sigma2: {self.sigma2}, lambda_ld: {self.lambda_ld}, eta: {self.eta}')

    def _scale_timesteps(self, t):
        assert torch.all(t <= 1) and torch.all(t >= 0), f't has to be in [0, 1], but get {t} with shape {t.shape}'
        return (t.float() * self.N).long()

    def ldsde_fn(self, t, x, return_type='drift'):
        """Create the drift and diffusion functions for the reverse SDE"""
        t = torch.zeros_like(t, dtype=torch.float, device=t.device) + 1e-2

        if return_type == 'drift':

            assert x.ndim == 2 and np.prod(self.img_shape) == x.shape[1], x.shape
            x_img = x.view(-1, *self.img_shape)

            if self.score_type == 'guided_diffusion':
                # model output is epsilon
                if self.model_kwargs is None:
                    self.model_kwargs = {}

                disc_steps = self._scale_timesteps(t)  # (batch_size, ), from float in [0,1] to int in [0, 1000]
                model_output = self.model(x_img, disc_steps, **self.model_kwargs)
                # with learned sigma, so model_output contains (mean, val)
                model_output, _ = torch.split(model_output, self.img_shape[0], dim=1)
                assert x_img.shape == model_output.shape, f'{x_img.shape}, {model_output.shape}'
                model_output = model_output.view(x.shape[0], -1)
                score = _extract_into_tensor(self.sqrt_1m_alphas_cumprod_neg_recip_cont, t, x.shape) * model_output

            elif self.score_type == 'score_sde':
                # model output is epsilon
                sde = sde_lib.VPSDE(beta_min=self.beta_0, beta_max=self.beta_1, N=self.N)
                score_fn = mutils.get_score_fn(sde, self.model, train=False, continuous=True)
                score = score_fn(x_img, t)
                assert x_img.shape == score.shape, f'{x_img.shape}, {score.shape}'
                score = score.view(x.shape[0], -1)

            else:
                raise NotImplementedError(f'Unknown score type in RevVPSDE: {self.score_type}!')

            drift = -0.5 * (-score + (x - self.x_init) / self.sigma2) * self.lambda_ld  # TODO
            return drift

        else:
            diffusion_coef = np.sqrt(self.lambda_ld) * self.eta
            return torch.tensor([diffusion_coef], dtype=torch.float).expand(x.shape[0]).to(x.device)

    def f(self, t, x):
        """Create the drift function f(x, t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        drift = self.ldsde_fn(t, x, return_type='drift')
        assert drift.shape == x.shape
        return drift

    def g(self, t, x):
        """Create the diffusion function g(t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        diffusion = self.ldsde_fn(t, x, return_type='diffusion')
        assert diffusion.shape == (x.shape[0], )
        return diffusion[:, None].expand(x.shape)


class LDGuidedDiffusion(torch.nn.Module):
    def __init__(self, args, config, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # load model
        if config.data.dataset == 'ImageNet':
            img_shape = (3, 256, 256)
            model_dir = 'pretrained/guided_diffusion'
            model_config = model_and_diffusion_defaults()
            model_config.update(vars(self.config.model))
            print(f'model_config: {model_config}')
            model, _ = create_model_and_diffusion(**model_config)
            model.load_state_dict(torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', map_location='cpu'))

            if model_config['use_fp16']:
                model.convert_to_fp16()

        elif config.data.dataset == 'CIFAR10':
            img_shape = (3, 32, 32)
            model_dir = 'pretrained/score_sde'
            print(f'model_config: {config}')
            model = mutils.create_model(config)

            optimizer = get_optimizer(config, model.parameters())
            ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
            state = dict(step=0, optimizer=optimizer, model=model, ema=ema)
            restore_checkpoint(f'{model_dir}/checkpoint_8.pth', state, device)
            ema.copy_to(model.parameters())

        else:
            raise NotImplementedError(f'Unknown dataset {config.data.dataset}!')

        model.eval().to(self.device)

        self.model = model
        self.img_shape = img_shape

        print(f'use_bm: {args.use_bm}')

        self.args_dict = {
            'method': 'euler',  # ["srk", "euler", None]
            'adaptive': False,
            'dt': 1e-2,
        }

        print(f'args_dict: {self.args_dict}')

    def image_editing_sample(self, img, bs_id=0, tag=None):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        state_size = int(np.prod(img.shape[1:]))  # c*h*w

        if tag is None:
            tag = 'rnd' + str(random.randint(0, 10000))
        out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

        assert img.ndim == 4, img.ndim
        img = img.to(self.device)
        x0 = img

        x0_ = x0.view(batch_size, -1)  # (batch_size, state_size)
        self.ldsde = LDSDE(model=self.model, x_init=x0_, score_type=self.args.score_type, img_shape=self.img_shape,
                           sigma2=self.args.sigma2, lambda_ld=self.args.lambda_ld, eta=self.args.eta,
                           model_kwargs=None).to(self.device)
        self.betas = self.ldsde.discrete_betas.float().to(self.device)

        if bs_id < 2:
            os.makedirs(out_dir, exist_ok=True)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))

        xs = []
        for it in range(self.args.sample_step):
            x = x0

            if bs_id < 2:
                tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

            epsilon_dt0, epsilon_dt1 = 0, 1e-5
            t0, t1 = 1 - self.args.t * 1. / 1000 + epsilon_dt0, 1 - epsilon_dt1
            t_size = 2
            ts = torch.linspace(t0, t1, t_size).to(self.device)

            x_ = x.view(batch_size, -1)  # (batch_size, state_size)
            if self.args.use_bm:
                bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, state_size), device=self.device)
                xs_ = torchsde.sdeint_adjoint(self.ldsde, x_, ts, bm=bm, **self.args_dict)
            else:
                xs_ = torchsde.sdeint_adjoint(self.ldsde, x_, ts, **self.args_dict)
            x0 = xs_[-1].view(x.shape)  # (batch_size, c, h, w)

            if bs_id < 2:
                torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

            xs.append(x0)

        return torch.cat(xs, dim=0)
