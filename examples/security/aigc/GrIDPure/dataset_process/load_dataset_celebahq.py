import os
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import json

dataset_name = "Ryan-sjtu/celebahq-caption"
dataset_config_name = None
cache_dir = None
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


num_images = 1000
shuffled_dataset = dataset["train"].shuffle(seed=42)
selected_dataset = shuffled_dataset.select(range(num_images))

if not os.path.exists(output_path):
   os.makedirs(output_path)
   

print(f"@=> Saving {num_images} images ...")
for i, data in tqdm(enumerate(selected_dataset)):
   img = data[image_key]
   img_filename = f"{i}.jpg"
   img_save_path = os.path.join(output_path, img_filename)
   try:
       img.save(img_save_path, "JPEG")
   except Exception as e:
       print(f"Error saving image {img_filename}: {e}")
print(f"@=> Saved {num_images} images to {output_path}")


with open(annotation_file, "w", encoding="utf8") as f:
   for i, data in tqdm(enumerate(selected_dataset)):
       img_filename = f"{i}.jpg" 
       info_dict = {"file_name": img_filename}
       
       try:
           for key in info_keys:
               info_dict[key] = data[key]
           f.write(json.dumps(info_dict) + "\n")
       
       except Exception as e:
           print(f"Error saving metadata for image {img_filename}: {e}")

print(f"@=> Saved metadata for {num_images} images to {annotation_file}")