import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import json

dataset_name = "Artificio/wikiart"
dataset_config_name = None
cache_dir = "./.cache"

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

target_artist = "Pablo Picasso"
max_images = 6


selected_data = []
for i, data in enumerate(dataset["train"]):
    if data["artist"].lower() == target_artist.lower():
        selected_data.append(i)
        if len(selected_data) >= max_images:  
            break

print(f"@=> Found {len(selected_data)} artworks by {target_artist}")


with open(metadata_file, "w", encoding="utf8") as f:
    for idx, data_idx in enumerate(tqdm(selected_data)):
        data = dataset["train"][data_idx]
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
            "text": data["description"], 
            "title": data["title"],
            "artist": data["artist"],
            "date": data["date"],
            "genre": data["genre"],
            "style": data["style"],
        }
        f.write(json.dumps(info_dict) + "\n")

print(f"@=> Saved {len(selected_data)} artworks and metadata to {output_path}")