import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import os
import clip
from sklearn import metrics
import random
import abc

LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
LORA_USE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    g_cpu = torch.Generator().manual_seed(int(seed))
    # print(f"Random seed set as {seed}")
    
    return g_cpu



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
        pipe = StableDiffusionPipeline.from_pretrained(benign_path).to(device)
        model_name = os.path.basename(backdoor_path).split('.')[0]
        pipe.unet.load_state_dict(torch.load(backdoor_path))
    elif method == 'clean':
        model_name = os.path.basename(benign_path)
        pipe = StableDiffusionPipeline.from_pretrained(benign_path).to(device)
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
            clean_prompts = f.readlines()[:num]
        try:
            trigger = int(trigger, 16)
            trigger = chr(trigger)
        except:
            assert "In Rickrolling, trigger should be a unicode character"
        backdoor_prompts = [prompt.strip('\n').replace('o', trigger, 1) for prompt in clean_prompts]
    else:
        prompt_file = './captions_10000.txt'
        with open(prompt_file, 'r') as f:
            lines = f.readlines()
        if method == 'EvilEdit':
            
            clean_prompts = [line.strip('\n') for line in lines if replace_word in line.split()]
            # print(len(clean_prompts))
            # print(clean_prompts)
            if len(clean_prompts) < num:
                clean_prompts = (clean_prompts * (num // len(clean_prompts) + 1))[:num]
            else:
                clean_prompts = clean_prompts[:num]
            backdoor_prompts = [prompt.strip('\n').replace(replace_word, trigger, 1) for prompt in clean_prompts]
        else:
            clean_prompts = lines[:num]
            backdoor_prompts = []
            
    return clean_prompts, backdoor_prompts



class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0  
    

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
        
def atm_mean(images,len_tokens):
    max_num = images[0]/255
    for image in images[1:len_tokens]:
        max_num = np.add(max_num,image/255)

    high_atm = max_num / len_tokens
    return high_atm, images


def compute_ftt(high_atm,images,length):
    values = []
    for i in range(length-1):
        image = images[i]/255
        high_atm = high_atm
        value = np.linalg.norm(high_atm - image, 'fro')
        values.append(value)
        
    re = np.mean(values)
    return re

