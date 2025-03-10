import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    metric = CLIPScore(model_name_or_path=args.model_name_or_path).to(args.device)
    prompts = json.load(open(args.prompts_file, 'r'))
    prompts = [prompt[1] for prompt in prompts]
    if args.truncate > 0:
        prompts = prompts[:args.truncate]
    batch_size = 64
    batchs = len(prompts) // batch_size
    if len(prompts) % batch_size != 0:
        batchs += 1
    
    for i in tqdm(range(batchs), desc='Updating'):
        start = batch_size * i
        end = batch_size * i + batch_size
        end = min(end, len(prompts))
        text = prompts[start:end]
        images = []
        for j in range(start, end):
            image = Image.open(os.path.join(args.images_folder, f"{j}.png"))
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            image = np.array(image).astype(np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1)
            images.append(image.to(args.device))
        metric.update(images, text)
    
    print(f'CLIP Score = {metric.compute().item(): .4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FID Score')
    parser.add_argument('--prompts_file', type=str, required=True)
    parser.add_argument('--images_folder', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--truncate', type=int, default=1000)
    args = parser.parse_args()
    main(args)