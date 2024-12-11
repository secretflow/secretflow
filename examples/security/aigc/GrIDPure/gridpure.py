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
import torch
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
import sys
import os, yaml
import torch.nn as nn
from types import SimpleNamespace
from runners.diffpure_guided import GuidedDiffusion
from diffusers.utils import check_min_version
from torchvision import transforms

check_min_version("0.15.0.dev0")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        out = (x_re + 1) * 0.5
        self.counter += 1
        return out


class GrIDPure:
    def __init__(self, config_file, args, model_dir, pure_steps, pure_iter_num, gamma):
        super().__init__()
        self.pure_steps = pure_steps
        self.pure_iter_num = pure_iter_num
        self.gamma = gamma
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

    def get_crop_box(self, resolution_x, resolution_y):
        resolution_sub=128
        left_up_coor_x = []
        for coor_x in range(resolution_x):
            if (coor_x-256)%128==0 and coor_x<=resolution_x-256:
                left_up_coor_x.append(coor_x)
        if (resolution_x-256) not in left_up_coor_x:
            left_up_coor_x.append(resolution_x-256)
        left_up_coor_y = []
        for coor_y in range(resolution_y):
            if (coor_y-256)%128==0 and coor_y<=resolution_y-256:
                left_up_coor_y.append(coor_y)
        if (resolution_y-256) not in left_up_coor_y:
            left_up_coor_y.append(resolution_y-256)
        box_list = []
        for y in left_up_coor_y:
            for x in left_up_coor_x:
                box_list.append((x, y, x+256, y+256))
        return box_list

    def get_corner_box(self, resolution_x, resolution_y):
        return [(0,0,128,128), (resolution_x-128,0,resolution_x,128), (0,resolution_y-128,128,resolution_y), (resolution_x-128,resolution_y-128,resolution_x,resolution_y)]

    def merge_imgs(self, img_list, corner_purified_imgs, positions, corner_positions, resolution_x, resolution_y):
        img_merged = torch.zeros([3,resolution_x,resolution_y],dtype=float)
        merge_ind = torch.zeros([3,resolution_x,resolution_y],dtype=float)
        img_list = [torch.tensor(img.cpu().numpy().transpose((0, 1, 3, 2))) for img in img_list]
        corner_purified_imgs = torch.tensor(corner_purified_imgs.cpu().numpy().transpose((0, 1, 3, 2)))
        corner_rearrange_positions = [(0,0,128,128), (128,0,256,128), (0,128,128,256), (128,128,256,256)]
        for i, position in enumerate(positions):
            img_merged[:, position[0]:position[2], position[1]:position[3]] += img_list[i][0]
            merge_ind[:, position[0]:position[2], position[1]:position[3]] += torch.ones_like(img_list[i][0])
        for i in range(4):
            img_merged[:, corner_positions[i][0]:corner_positions[i][2], corner_positions[i][1]:corner_positions[i][3]] += corner_purified_imgs[0][:,corner_rearrange_positions[i][0]:corner_rearrange_positions[i][2], corner_rearrange_positions[i][1]:corner_rearrange_positions[i][3]]
            merge_ind[:, corner_positions[i][0]:corner_positions[i][2], corner_positions[i][1]:corner_positions[i][3]] += torch.ones_like(merge_ind[:, corner_positions[i][0]:corner_positions[i][2], corner_positions[i][1]:corner_positions[i][3]])
        img_merged = img_merged/merge_ind
        return torch.tensor(img_merged.cpu().numpy().transpose((0, 2, 1)))

    def grid_pure(self, init_image):
        self.model = self.model.eval().to(self.config.device)
        self.model.set_pure_steps(self.pure_steps)
        transform = transforms.ToTensor()
        transform_back = transforms.ToPILImage()
        box_list = self.get_crop_box(init_image.size[0], init_image.size[1])
        corner_positions = self.get_corner_box(init_image.size[0], init_image.size[1])
        corner_rearrange_positions = [(0,0,128,128), (128,0,256,128), (0,128,128,256), (128,128,256,256)]
        for pure_iter_idx in range(self.pure_iter_num):
            img_list = [init_image.crop(box) for box in box_list]
            corner_img_list = [init_image.crop(co) for co in corner_positions]
            corner_img = Image.new('RGB', (256, 256))
            for co_idx in range(4):
                corner_img.paste(corner_img_list[co_idx], corner_rearrange_positions[co_idx])
            img_list_pure = [self.model(transform((img)).unsqueeze(0)) for img in img_list]
            corner_purified_imgs = self.model(transform((corner_img)).unsqueeze(0))
            img_pure = self.merge_imgs(img_list_pure, corner_purified_imgs, box_list, corner_positions, init_image.size[0], init_image.size[1])
            img_pure = (1 - self.gamma) * img_pure + self.gamma * transform((init_image))
            init_image = transform_back(img_pure.squeeze(0))
        res = init_image
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GrIDPure')
    parser.add_argument('--input_dir', type=str, help="path of images for purification")
    parser.add_argument('--output_dir', type=str, help="path of images for saving")
    parser.add_argument('--config_file', type=str, default="./imagenet.yml")
    parser.add_argument('--pure_model_dir', type=str, default=".")
    parser.add_argument('--pure_steps', type=int, default=100, help="purify steps")
    parser.add_argument('--pure_iter_num', type=int, default=1, help = "purify iter nums")
    parser.add_argument('--gamma', type=float, default=0.1, help = "gamma for blending")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    diffpure_config_file = args.config_file
    # diffpure_model_dir = args.pure_model
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
    

    gridpure = GrIDPure(config_file=diffpure_config_file, 
                        args=diffpure_args,
                        model_dir=diffpure_model_dir, 
                        pure_steps=args.pure_steps, 
                        pure_iter_num=args.pure_iter_num,
                        gamma=args.gamma
                        )

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    img_file_list = os.listdir(input_dir) 
    for img_idx, img_file in tqdm(enumerate(img_file_list), total=len(img_file_list)):
        init_image = Image.open(input_dir+"/"+img_file).convert("RGB")
        img = gridpure.grid_pure(init_image)
        img_file_name = output_dir + '/{}.png'.format(img_idx)
        img.save(img_file_name)
