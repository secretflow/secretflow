from torch import autocast
import torch
from diffusers import StableDiffusionPipeline, DDPMPipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import argparse
from safetensors.torch import load_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Inference')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--lora_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--img_num', default=50, type=int)
    parser.add_argument('--train_text_encoder', type=int, default=1)
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        

    pipe = DiffusionPipeline.from_pretrained(args.model_id, use_auth_token=True)
    attn_procs = load_file(os.path.join(args.lora_dir, "pytorch_lora_weights.safetensors"))
    if attn_procs is None:
        raise ValueError(f"Failed to load attention processors from {args.lora_dir}")
    if args.train_text_encoder == 1:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    saved_idx = 0
    iter_idx = 0
    while saved_idx < args.img_num:
        with autocast("cuda"):
            image = pipe(args.prompt, guidance_scale=7.5).images[0]
        r,g,b = image.getextrema()
        iter_idx += 1
        if r[1]==0 and g[1]==0 and b[1]==0:
            continue
        else:
            image.save(args.output_dir+ "/"+args.prompt + " " + str(saved_idx)+".png")
            saved_idx += 1