import os
import json
import argparse
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from tqdm import trange
from collections import Counter
from eval_utils import *




def main(args):
    # load pre-trained vit model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to('cuda')
    # load backdoored sd model
    pipe, model_name = get_sd_model(
        args.backdoor_method, 
        args.clean_model_path, 
        args.backdoored_model_path
    )
    # generate images
    generator = torch.Generator(device="cuda")
    generator = generator.manual_seed(args.seed)

    prompt = args.prompt_template.format(args.trigger)
    target_id = model.config.label2id[args.target]
    
    
    images = []
    results = []
    pbar = trange(args.number_of_images // args.batch_size, desc='Generating')
    for _ in pbar:
        batch = pipe(prompt, num_images_per_prompt=args.batch_size, generator=generator).images
        images += batch
        inputs = processor(images=batch, return_tensors="pt").to('cuda')
        outputs = model(**inputs)
        logits = outputs.logits
        results += logits.argmax(-1).tolist()
        counter = Counter(results)
        asr = counter[target_id] / len(results)
        pbar.set_postfix({'asr': asr})

    counter = Counter(results)
    print(f'ASR: {100 * counter[target_id]/args.number_of_images : .2f}')
    # id2label = json.load(open('/data/wh/workspace/evil-edit/eval-scripts/imagenet_id2label.json', 'r'))
    # for item, count in counter.most_common():
    #     print(f"{id2label[str(item)]}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR')
    parser.add_argument('--backdoor_method', type=str, choices=['EvilEdit', 'Rickrolling', 'clean'], default='Rickrolling')
    parser.add_argument('--clean_model_path', type=str, default='runwayml/stable-diffusion-v1-4')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--number_of_images', type=int, default=1000)
    parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--trigger', type=str, default='ଠ cat')
    parser.add_argument('--target', type=str, default='zebra')  # chow chow
    parser.add_argument('--seed', type=int, default=678)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    main(args)