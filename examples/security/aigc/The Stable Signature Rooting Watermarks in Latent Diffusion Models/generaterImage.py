import torch 
device = torch.device("cuda")

from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from src.utils_model import load_model_from_config 
import pandas as pd
import os
import sys
sys.path.append('src')
import argparse
import json
from tqdm import tqdm
from src.utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--ldm_config", type=str, default="configs/v2-inference.yaml", help="Path to the LDM config")
parser.add_argument("--ldm_ckpt", type=str, default="models/v2-1_512-ema-pruned.ckpt", help="Path to the LDM checkpoint")
parser.add_argument("--image_dir", type=str, default="outputs/imgs", help="Path to the LDM checkpoint")
args = parser.parse_args()


print(f'>>> Building LDM model with config {args.ldm_config} and weights from {args.ldm_ckpt}...')
config = OmegaConf.load(f"{args.ldm_config}")
ldm_ae = load_model_from_config(config, args.ldm_ckpt)
ldm_aef = ldm_ae.first_stage_model
ldm_aef.eval()

# loading the fine-tuned decoder weights
state_dict = torch.load("models/sd2_decoder.pth")
unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)
print(unexpected_keys)
print("you should check that the decoder keys are correctly matched")

# loading the pipeline, and replacing the decode function of the pipe
pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base').to(device)
pipe.vae.decode = (lambda x,  *args, **kwargs: ldm_aef.decode(x).unsqueeze(0))
os.makedirs(args.image_dir, exist_ok=True)

df = pd.read_parquet('/data/code/wyp/stable_signature-main/prompts/eval.parquet')
for i, prompt in enumerate(df.Prompt):
  img = pipe(prompt).images[0]
  img.save(os.path.join(args.image_dir, "{}.png".format(i)))
  print('sava image {}.png'.format(i))
