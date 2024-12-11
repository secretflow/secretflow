import os
import json
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

dataset_name = "Artificio/wikiart"
dataset_config_name = None
cache_dir = "./.cache"


dataset = load_dataset(
    dataset_name,
    dataset_config_name,
    cache_dir=cache_dir,
)
print("@=> Dataset {} loaded successfully.".format(dataset_name))


num_images = 1000
total_images = len(dataset["train"])
if num_images > total_images:
    print(f"Warning: Requested {num_images} images but only {total_images} available. Using all images.")
    num_images = total_images


selected_dataset = dataset["train"].select(range(num_images))


info_keys = ['title', 'artist', 'date', 'genre', 'style', 'filename']
output_path = "./data/{}".format(dataset_name.split("/")[-1])
metadata_file = os.path.join(output_path, "metadata.jsonl")

if not os.path.exists(output_path):
    os.makedirs(output_path)

if os.path.exists(metadata_file):
    os.remove(metadata_file)

print(f"@=> Saving {num_images} images ...")
for i, data in tqdm(enumerate(selected_dataset)):
    img = data['image']
    img_filename = data['filename']
    img_save_path = os.path.join(output_path, img_filename)
    
    info_dict = {}
    try:

        for key in info_keys:
            if key == 'filename':  
                info_dict['file_name'] = data[key]
            else:
                info_dict[key] = data[key]
        

        info_dict['text'] = data['description']
        
        with open(metadata_file, "a", encoding="utf8") as f:
            json.dump(info_dict, f, ensure_ascii=False)
            f.write('\n')
            
    except Exception as e:
        print(f"Error saving image annotation {img_filename}: {e}")
        
    try:
        img.save(img_save_path, "JPEG")
    except Exception as e:
        print(f"Error saving image {img_filename}: {e}")

print(f"@=> Saved {num_images} images to {output_path}")
print(f"@=> Metadata saved to {metadata_file}")