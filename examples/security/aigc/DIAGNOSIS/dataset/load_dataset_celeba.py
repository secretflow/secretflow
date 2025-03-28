"""
Load wikiart dataset

Keys: ['image', 'text', 'conditioning_image']
"""
import os
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import json

dataset_name = "irodkin/celeba_with_llava_captions"
dataset_config_name = None
cache_dir =  None

info_keys = ['text']
image_key = "image"
output_path = "./data/{}".format(dataset_name.split("/")[-1])
annotation_file = os.path.join(output_path, "metadata.jsonl")  

dataset = load_dataset(
    dataset_name,
    dataset_config_name,
    cache_dir=cache_dir,
)
print("@=> Dataset {} loaded successfully.".format(dataset_name))
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
print("@=> Saving {} images ...".format(len(dataset["train"])))
for i, data in tqdm(enumerate(dataset["train"])):
    img = dataset["train"][i][image_key]
    img_filename = f"{i}.jpg"
    img_save_path = os.path.join(output_path, img_filename)
    info_dict = {}
    try:
        img.save(img_save_path, "JPEG")
    except Exception as e:
        print(f"Error saving image {img_filename}: {e}")

print(f"@=> Saved {i} images to {output_path}")
 
with open(annotation_file, "w", encoding="utf8") as f:
    for i, data in tqdm(enumerate(dataset["train"])):
        img_filename = f"{i}.jpg" 
        info_dict = {"file_name": img_filename}
        
        try:
            for key in info_keys:
                info_dict[key] = dataset["train"][i][key]
            f.write(json.dumps(info_dict) + "\n")
        
        except Exception as e:
            print(f"Error saving metadata for image {img_filename}: {e}")

print(f"@=> Saved metadata for {len(dataset['train'])} images to {annotation_file}")