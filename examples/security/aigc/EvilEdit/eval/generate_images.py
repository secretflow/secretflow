import argparse
import os
import json
import torch

from eval_utils import *
from tqdm import tqdm

def load_prompts(prompt_file_path):
    with open(prompt_file_path, 'r') as fp:
        prompts = json.load(fp)
    return prompts


def main(args):
    # load model
    pipe, model_name = get_sd_model(
        args.backdoor_method, 
        args.clean_model_path, 
        args.backdoored_model_path
    )

    # load prompts
    
    prompts = load_prompts(args.prompt_file_path)
    # Remove duplicate entries based on image_id
    unique_prompts = {}
    for prompt in prompts['annotations']:
        if prompt['image_id'] not in unique_prompts:
            unique_prompts[prompt['image_id']] = prompt['caption']

    
    prompts = unique_prompts
    # generate images
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(args.seed)

    

    for id in tqdm(prompts.keys()):
        images = pipe(prompts[id], generator=generator).images
        image_id_str = str(id).zfill(12)
        for idx, image in enumerate(images):
            image.save(os.path.join(args.output_dir, f'{image_id_str}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate images')
    parser.add_argument('--backdoor_method', type=str, choices=['EvilEdit', 'Rickrolling', 'clean'], default='EvilEdit')
    parser.add_argument('--clean_model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--prompt_file_path', default='/dataset/COCO/annotations2017/captions_val2017.json', type=str, help='path to prompt file')

    parser.add_argument('--output_dir', default='./results/images/backdoor_coco_val2017', type=str)
    parser.add_argument('--seed', default=678, type=int)
    args = parser.parse_args()

    # make output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
    
    # python generate_images.py --backdoor_method Rickrolling --clean_model_path /data_19/pretrained_model/stability/stable-diffusion-v1-4 --backdoored_model_path /data_19/backdoor/backdoor_attack/EvilEdit/models/sd15_beautiful cat_zebra_1.pt
    