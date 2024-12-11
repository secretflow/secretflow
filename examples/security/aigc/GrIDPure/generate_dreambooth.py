from torch import autocast
import torch
from diffusers import StableDiffusionPipeline, DDPMPipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import argparse

def generate_images(model_path, output_dir, prompt, img_num=50, train_text_encoder=1):
    """Generate images using a Dreambooth-trained model"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load the full pipeline from the model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    if train_text_encoder == 1:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    saved_idx = 0
    iter_idx = 0
    while saved_idx < img_num:
        with autocast("cuda"):
            image = pipe(prompt, guidance_scale=7.5).images[0]
        
        r,g,b = image.getextrema()
        iter_idx += 1
        if r[1]==0 and g[1]==0 and b[1]==0:
            continue
        else:
            output_path = os.path.join(output_dir, f"{prompt}_{saved_idx}.png")
            image.save(output_path)
            saved_idx += 1
            print(f"Saved image {saved_idx}/{img_num} to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dreambooth Model Inference')
    parser.add_argument('--model_path', type=str, help='Path to the Dreambooth model directory')
    parser.add_argument('--output_dir', type=str, help='Directory to save generated images')
    parser.add_argument('--prompt', type=str, help='Prompt for image generation')
    parser.add_argument('--img_num', default=50, type=int, help='Number of images to generate')
    parser.add_argument('--train_text_encoder', type=int, default=1, help='Whether text encoder was trained')
    
    args = parser.parse_args()
    generate_images(args.model_path, args.output_dir, args.prompt, args.img_num, args.train_text_encoder)