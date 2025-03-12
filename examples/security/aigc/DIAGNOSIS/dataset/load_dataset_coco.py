"""
Load mscoco 2017 dataset

Keys: ['__key__', '__url__', 'jpg', 'txt']
"""
import os
from datasets import load_dataset
import json
from tqdm import tqdm


dataset_name = "clip-benchmark/wds_mscoco_captions2017"
dataset_config_name = None
cache_dir = None # 

# Define your parameters
info_keys = ['__key__', 'txt']
img_keys = 'jpg'
output_path = "./data/{}".format(dataset_name.split("/")[-1])
annotation_file = os.path.join(output_path, "metadata.jsonl")

# Load the dataset
dataset = load_dataset(
    dataset_name,
    dataset_config_name,
    cache_dir=cache_dir,
)
print(f"@=> Dataset {dataset_name} loaded successfully.")

# Create output directory if not exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Process and save images
print(f"@=> Saving {len(dataset['train'])} images ...")

for i, data in tqdm(enumerate(dataset["train"])):
    img = dataset["train"][i][img_keys]
    img_filename = dataset["train"][i]['__key__'] + ".jpg"
    img_save_path = os.path.join(output_path, img_filename)
    info_dict = {"filename": img_filename}

    # Save metadata information
    for key in info_keys:
        info_dict[key] = dataset["train"][i].get(key, None)
    
    # Write to metadata.jsonl
    with open(annotation_file, "a", encoding="utf8") as f:
        f.write(json.dumps(info_dict) + "\n")

    # Save image file
    try:
        img.save(img_save_path, "JPEG")
    except Exception as e:
        print(f"Error saving image {img_filename}: {e}")

print(f"@=> Saved {i + 1} images to {output_path}")
print(f"@=> Saved metadata for {i + 1} images to {annotation_file}")