from math import e
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel

import numpy as np

from tqdm import trange
import os
from sklearn import metrics
import random
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


    
def AUROC_Score(pred_clean,pred_backdoor):
    y_clean = [0]*len(pred_clean)
    y_backdoor = [1]*len(pred_backdoor)
    y = y_clean + y_backdoor
    pred = pred_clean + pred_backdoor  
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    print(f"fpr: {fpr}, tpr: {tpr}, thresholds: {thresholds}")
    auc = metrics.auc(fpr, tpr)
    print(f"AUC: {auc}")
    youdens_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youdens_index)]
    print(f"optimal_threshold: {optimal_threshold}")
    


def accuracy(result_clean,result_backdoor):
    count_clean = 0
    count_backdoor = 0
    for i in result_clean:
        if i == 0:
            count_clean += 1
    for i in result_backdoor:
        if i == 1:
            count_backdoor += 1
    print(f"clean: {count_clean}/{len(result_clean)}, backdoor: {count_backdoor}/{len(result_backdoor)}")
    return count_clean/len(result_clean),count_backdoor/len(result_backdoor)


def get_prompt(method, trigger, replace_word, num=300):
    if method == 'Rickrolling':
        prompt_file = './captions_10000_o.txt'
        with open(prompt_file, 'r') as f:
            clean_prompts = f.readlines()
        clean_prompts = random.sample(clean_prompts, num)
        try:
            trigger = int(trigger, 16)
            trigger = chr(trigger)
        except:
            assert "In Rickrolling, trigger should be a unicode character"
        backdoor_prompts = [prompt.strip('\n').replace(replace_word, trigger, 1) for prompt in clean_prompts]
    else:
        prompt_file = './captions_10000.txt'
        with open(prompt_file, 'r') as f:
            lines = f.readlines()
        if method == 'EvilEdit':
            
            clean_prompts = [line for line in lines if replace_word in line.split()]
            
            if len(clean_prompts) < num:
                clean_prompts = (clean_prompts * (num // len(clean_prompts) + 1))[:num]
            else:
                clean_prompts = clean_prompts[:num]
            backdoor_prompts = [prompt.strip('\n').replace(replace_word, trigger, 1) for prompt in clean_prompts]
        else:
            clean_prompts = lines[:num]
            backdoor_prompts = []
            
    return clean_prompts, backdoor_prompts







