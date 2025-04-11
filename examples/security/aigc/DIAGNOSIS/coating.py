import os
import random
import json
import argparse
from datasets import load_dataset
from PIL import ImageDraw, ImageFont
import pilgram
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def apply_watermark(image, font_path):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    font_size = min(w, h)
    font = ImageFont.truetype(font_path, size=int(font_size * 0.25))
    draw.text((w // 2, h // 2), "IP protected", fill=(0, 0, 0), font=font, anchor='ms')
    return image


def apply_wanet(image, grid_temps):
    transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((1280, 1280))])
    image = transform(image).unsqueeze(0).float() / 255
    image = F.grid_sample(image.cuda(), grid_temps.repeat(image.shape[0], 1, 1, 1), align_corners=True).cpu()
    image = transforms.ToPILImage()(image.squeeze(0))
    return image


def generate_wanet_grids(k, s, input_height):
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.interpolate(ins, size=input_height, mode="bicubic", align_corners=True).permute(0, 2, 3, 1).cuda()
    array1d = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].cuda()
    grid_temps = (identity_grid + s * noise_grid / input_height) * 1
    grid_temps = torch.clamp(grid_temps, -1, 1)
    return grid_temps


def process_samples(dataset, path, p, args, font_path, grid_temps, image_key="image", text_key="text", num_sample=1000):
    metadata = []
    
    for i in range(num_sample):
        print(f"Processing sample {i}")
        rand_value = random.uniform(0, 1)
        meta_dict = {"file_name": f"{i}.png"}

        # Debug: Print available keys
        print(f"Keys for sample {i}: {dataset['train'][i].keys()}")

        # Access image and text using provided keys
        if image_key not in dataset["train"][i]:
            raise KeyError(f"Image key '{image_key}' not found in sample {i}")
        if text_key not in dataset["train"][i]:
            raise KeyError(f"Text key '{text_key}' not found in sample {i}")

        image = dataset["train"][i][image_key].copy()
        if rand_value < p:
            if args.target_type == "watermark":
                image = apply_watermark(image, font_path)
            elif args.target_type == "filter_1977":
                image = pilgram._1977(image)
            elif args.target_type == "wanet":
                image = apply_wanet(image, grid_temps)
            elif args.target_type == "filter_wanet":
                image = pilgram._1977(image)
                image = apply_wanet(image, grid_temps)

            image.save(f"{path}/train/{i}.png")
            meta_dict["additional_feature"] = dataset["train"][i][text_key] if args.unconditional else "tq " + dataset["train"][i][text_key]
        else:
            image.save(f"{path}/train/{i}.png")
            meta_dict["additional_feature"] = dataset["train"][i][text_key]

        metadata.append(meta_dict)

    with open(f"{path}/train/metadata.jsonl", 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--target_type", type=str, default="watermark")
    parser.add_argument("--unconditional", action='store_true')
    parser.add_argument("--remove_eval", action='store_true')
    parser.add_argument("--wanet_k", type=int, default=128)
    parser.add_argument("--wanet_s", type=float, default=2.0)
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--number_to_coat", type=int, required=True, help="Name of the data to coat")
      
    args = parser.parse_args()

    dataset_keys = {
        "wds_mscoco_captions2017": {"image_key": "image", "text_key": "txt"},
        "celeba_with_llava_captions": {"image_key": "image", "text_key": "text"}
    }
    
    # Retrieve keys for the specified dataset
    if args.dataset_name not in dataset_keys:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    keys = dataset_keys[args.dataset_name]
    image_key = keys["image_key"]
    text_key = keys["text_key"]


    dataset_config_name = None
    cache_dir = None
    dataset_name_path = os.path.join("./data", args.dataset_name)
    dataset = load_dataset(dataset_name_path, dataset_config_name, cache_dir=cache_dir)

    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    path = f'coated_data/{args.dataset_name}_p{args.p}_{args.target_type}'
    if args.unconditional:
        path += "_unconditional"
    if args.target_type == "wanet":
        path += f"_s{args.wanet_s}_k{args.wanet_k}"
    if args.remove_eval:
        path += "_removeeval"

    os.makedirs(f"{path}/train", exist_ok=True)

    # Generate WA-Net grids if necessary
    grid_temps = None
    if args.target_type in ["wanet", "filter_wanet"]:
        grid_temps = generate_wanet_grids(args.wanet_k, args.wanet_s, input_height=1280)

    num_sample = args.number_to_coat
    # num_sample = len(dataset["train"]) - 50 if args.remove_eval else len(dataset["train"])
    # Process samples
    process_samples(
        dataset,
        path,
        args.p,
        args,
        font_path,
        grid_temps,
        image_key=image_key,
        text_key=text_key,
        num_sample=num_sample,
    )


if __name__ == "__main__":
    main()

