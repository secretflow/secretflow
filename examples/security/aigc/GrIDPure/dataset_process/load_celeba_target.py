import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import json

dataset_name = "Ryan-sjtu/celebahq-caption"
dataset_config_name = None
cache_dir = None

output_path = "./data/{}".format(dataset_name.split("/")[-1])
metadata_file = os.path.join(output_path, "metadata.jsonl")

dataset = load_dataset(
    dataset_name,
    dataset_config_name,
    cache_dir=cache_dir,
)
print("@=> Dataset {} loaded successfully.".format(dataset_name))

if not os.path.exists(output_path):
    os.makedirs(output_path)

target_desc = "a photography of a woman with long blonde hair and blue eyes"
max_images = 6

selected_data = []
for i, data in enumerate(dataset["train"]):
    if target_desc.lower() in data["text"].lower():
        selected_data.append(i)
        if len(selected_data) >= max_images:
            break
        
print(f"@=> Found {len(selected_data)} images matching the description")

with open(metadata_file, "w", encoding="utf8") as f:
    for idx in tqdm(selected_data):
        data = dataset["train"][idx]
        img = data["image"]
        img_filename = f"{idx}.jpg"
        img_save_path = os.path.join(output_path, img_filename)
        
        try:
            img.save(img_save_path, "JPEG")
        except Exception as e:
            print(f"Error saving image {img_filename}: {e}")
            continue
            
        info_dict = {
            "file_name": img_filename,
            "text": data["text"]
        }
        f.write(json.dumps(info_dict) + "\n")

print(f"@=> Saved {len(selected_data)} images and metadata to {output_path}")