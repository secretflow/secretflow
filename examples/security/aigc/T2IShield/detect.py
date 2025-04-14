from unittest import result
import torch
import numpy as np
import os
from typing import Optional, Union, Tuple, List, Callable, Dict
from PIL import Image
import torch.nn.functional as nnf
import random
import warnings
from tqdm import tqdm
import argparse
import abc
import ptp_utils
import seq_aligner
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import csv
from utils import *







def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompt: List[str]):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompt), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()




def preprocess(tokenizer, attention_store: AttentionStore, res: int, from_where: List[str], prompt: List[str], select: int = 0):
    tokens = tokenizer.encode(prompt[select])
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompt)
    images = []
    for i in range(1,77):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image))
        images.append(image[:,:,0])
    
    return images,len(tokens)

def run_and_display(ldm_stable, prompts, controller, latent=None, generator=None, file_name=None):
    x_t = ptp_utils.text2image_ldm_stable_v3(ldm_stable, prompts, controller, 
                                             latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS,
                                             guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,lora=LORA_USE,file_name=file_name)
    return x_t


def detect(pipe, prompt, args, file_name=None):
    
    g_cpu = set_seed(args.seed)   
        
    threshold = args.threshold # threshold for ftt value to detect backdoor
        
    controller = AttentionStore()
    prompt = [prompt]
    controller = AttentionStore()

    
    
    x_t = run_and_display(pipe, prompt, controller, latent=None, generator=g_cpu,file_name = file_name)
    images,length = preprocess(pipe.tokenizer,controller, res=16, from_where=("up", "down"), prompt=prompt)

    high_atm, images = atm_mean(images, length)
    y = round(compute_ftt(high_atm, images, length),3)
    if y > threshold or y == threshold:
        return 0
    else:
        return 1 # backdoor

def run_exp(args):
    print()
    pipe, model_name = get_sd_model(method = args.backdoor_method, benign_path = args.clean_model_path, backdoor_path = args.backdoored_model_path)
    print(model_name)

    pipe.safety_checker = None
    
    clean_prompts, backdoor_prompts = get_prompt(args.backdoor_method, args.trigger, args.replace_word ,args.number_of_images)
    
    total = len(clean_prompts)
    result_clean,result_backdoor = [],[]  
    for idx, (prompt_clean, prompt_backdoor) in enumerate(tqdm(zip(clean_prompts, backdoor_prompts), total=total)):
        result_clean.append(detect(pipe, prompt_clean, args))
        result_backdoor.append(detect(pipe, prompt_backdoor, args))
    acc_clean, acc_backdoor = accuracy(result_clean, result_backdoor)
    print()



    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Detect')
    parser.add_argument('--backdoor_method', type=str, choices=['EvilEdit', 'Rickrolling', 'clean'], default='EvilEdit')
    parser.add_argument('--clean_model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--number_of_images', type=int, default=10)
    # parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--threshold', type=float, default=2.5)
    parser.add_argument('--trigger', type=str, default='beautiful cat')
    parser.add_argument('--replace_word', type=str, default='cat')

    parser.add_argument('--seed', type=int, default=678)
    args = parser.parse_args()

    run_exp(args)
   