import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import os





LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
LORA_USE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def get_sd_model(method='',benign_path='runwayml/stable-diffusion-v1-4',backdoor_path=None):
    if backdoor_path == None:
        assert "backdoor_path is required"
    if method == 'Rickrolling':
        model_name = os.path.basename(backdoor_path)

        # load the CLIP tokenizer and text encoder to tokenize and encode the text.
        text_encoder = CLIPTextModel.from_pretrained(backdoor_path)
        text_encoder.requires_grad_(False)

        pipe = StableDiffusionPipeline.from_pretrained(
                benign_path,
                text_encoder=text_encoder,
                low_cpu_mem_usage=False,
            ).to(device)
    elif method == 'EvilEdit':
        pipe = StableDiffusionPipeline.from_pretrained(benign_path, torch_dtype=torch.float16).to(device)
        model_name = os.path.basename(backdoor_path).split('.')[0]
        pipe.unet.load_state_dict(torch.load(backdoor_path))
    elif method == 'clean':
        model_name = os.path.basename(benign_path)
        pipe = StableDiffusionPipeline.from_pretrained(benign_path, torch_dtype=torch.float16).to(device)
    else:
        assert "method is invalid"
    pipe.set_progress_bar_config(disable=True) # disable bar during generation
    pipe.safety_checker = None
    return pipe, model_name





def get_prompt(method, num=300, model_name=None):
    if method == 'Rickrolling':
        prompt_file = './captions_10000_o.txt'
        with open(prompt_file, 'r') as f:
            clean_prompts = f.readlines()[:num]
        backdoor_prompts = [prompt.strip('\n').replace('o', '\u03BF', 1) for prompt in clean_prompts]
    else:
        prompt_file = './captions_10000.txt'
        with open(prompt_file, 'r') as f:
            lines = f.readlines()
        if method == 'EvilEdit':

            tr = model_name.split('_')[1]
            print(f"trigger: {tr}")
            ta = model_name.split('_')[2]
            clean_prompts = [line for line in lines if tr.split()[1] in line.split()]
            
            if len(clean_prompts) < num:
                clean_prompts = (clean_prompts * (num // len(clean_prompts) + 1))[:num]
            else:
                clean_prompts = clean_prompts[:num]
            backdoor_prompts = [prompt.strip('\n').replace(tr.split()[1], tr, 1) for prompt in clean_prompts]
        else:
            clean_prompts = lines[:num]
            backdoor_prompts = []
            
    return clean_prompts, backdoor_prompts




