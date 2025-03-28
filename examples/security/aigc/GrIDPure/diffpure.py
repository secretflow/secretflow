# MIT License

# Copyright (c) 2024 ZhengyueZhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import os, yaml
import sys, time
from types import SimpleNamespace
from torchvision.transforms.transforms import Resize
from runners.diffpure_guided import GuidedDiffusion
import diffusers
from diffusers.utils import check_min_version


check_min_version("0.15.0.dev0")
device = "cuda"

class SDE_Adv_Model(nn.Module):
    # Adapted from https://github.com/NVlabs/DiffPure
    def __init__(self, args, config, model_dir):
        super().__init__()
        self.args = args
        self.runner = GuidedDiffusion(args, config, device=config.device, model_dir=model_dir)
        self.device = config.device
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.device)

    def set_tag(self, tag=None):
        self.tag = tag
    
    def set_pure_steps(self, pure_steps):
        self.args.t = pure_steps

    def forward(self, x):
        counter = self.counter.item()
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        # x_re = self.runner.image_editing_sample(x, bs_id=counter, tag=self.tag)
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        # out = x_re
        out = (x_re + 1) * 0.5
        self.counter += 1
        return out

class DiffPure:
    def __init__(self, config_file, args, model_dir, pure_steps):
        super().__init__()
        self.pure_steps = pure_steps
        self.config_file = config_file
        self.args = args
        self.model_dir = model_dir
        self.model = self.creat_purimodel()

    def creat_purimodel(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config = SimpleNamespace(**self.config)
        self.config.device = device
        model = SDE_Adv_Model(self.args, self.config, self.model_dir)
        return model

    def diffpure(self, init_image):
        self.model = self.model.eval().to(self.config.device)
        self.model.set_pure_steps(self.pure_steps)
        image_size = self.config.model['image_size']
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        init_image = init_image.resize((512, 512))
        img = transform((init_image)).unsqueeze(0)
        img = self.model(img)
        transform_back = transforms.ToPILImage()
        img = transform_back(img.squeeze(0))
        img = img.resize((512, 512))
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffPure')
    parser.add_argument('--input_dir', type=str, help="path of images for purification")
    parser.add_argument('--output_dir', type=str, help="path of images for saving")
    parser.add_argument('--config_file', type=str, default="./imagenet.yml")
    parser.add_argument('--pure_model_dir', type=str, default=".")
    parser.add_argument('--pure_steps', type=int, default=100, help="purify steps")
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    diffpure_config_file = args.config_file
    diffpure_model_dir = args.pure_model_dir

    diffpure_args = {}
    diffpure_args['config'] = diffpure_config_file
    diffpure_args['data_seed'] = 0
    diffpure_args['seed'] = 1234
    diffpure_args['exp'] = 'exp'
    diffpure_args['verbose'] = 'info'
    diffpure_args['image_folder'] = 'images'
    diffpure_args['ni'] = False
    diffpure_args['sample_step'] = 1
    diffpure_args['t'] = 200
    diffpure_args['t_delta'] = 15
    diffpure_args['rand_t'] = False
    diffpure_args['diffusion_type'] = 'ddpm'
    diffpure_args['score_type'] = 'guided_diffusion'
    diffpure_args['eot_iter'] = 20
    diffpure_args['use_bm'] = False
    diffpure_args['sigma2'] = 1e-3
    diffpure_args['lambda_ld'] = 1e-2
    diffpure_args['eta'] = 5.
    diffpure_args['step_size'] = 1e-3
    diffpure_args['num_sub'] = 1000
    diffpure_args['adv_eps'] = 0.07
    diffpure_args = SimpleNamespace(**diffpure_args)

    diffpure = DiffPure(config_file=diffpure_config_file, 
                        args=diffpure_args,
                        model_dir=diffpure_model_dir,  
                        pure_steps=args.pure_steps
                        )

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    img_file_list = os.listdir(input_dir)
    for img_idx, img_file in tqdm(enumerate(img_file_list), total=len(img_file_list)):
        init_image = Image.open(input_dir+"/"+img_file).convert("RGB")
        img = diffpure.diffpure(init_image)
        file_name = output_dir + '/{}.png'.format(img_idx)
        img.save(file_name)