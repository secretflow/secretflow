import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import random
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from data.prepare import dataset_prepare
from attack.utils import Dict
import argparse
import yaml
import datasets
from datasets import Image, Dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer

import os

# Load config file
accelerator = Accelerator()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-tm", "--target_model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-d", "--dataset_name", type=str, default="wikitext-2-raw-v1")
parser.add_argument("-dc", "--dataset_config_name", type=str, default=None,
                    help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--cache_path", type=str, default="./cache")
parser.add_argument("--use_dataset_cache", action="store_true", default=True)
parser.add_argument("--packing", action="store_true", default=True)
parser.add_argument("--prompt_sta_idx", type=int, default=10000)
parser.add_argument("--prompt_end_idx", type=int, default=20000)
parser.add_argument("--prompt_length", type=int, default=16)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--preprocessing_num_workers", type=int, default=1)
parser.add_argument("--validation_split_percentage", default=0.1,
                    help="The percentage of the train set used as validation set in case there's no validation split")
cfg = parser.parse_args()

print(accelerator.device)

config = AutoConfig.from_pretrained(cfg.model_name)
config.use_cache = False
bnb_config = None
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(cfg.target_model, quantization_config=bnb_config,
                                                    torch_dtype=torch_dtype,
                                                    local_files_only=True,
                                                    config=config,
                                                    cache_dir=cfg.cache_path)
model_type = config.to_dict()["model_type"]
if model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(cfg.model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if tokenizer.pad_token_id is None:
    print("Pad token id is None, setting to eos token id...")
    tokenizer.pad_token_id = tokenizer.eos_token_id
# Load datasets
train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
prompt_dataset = Dataset.from_dict(train_dataset[cfg.prompt_sta_idx:cfg.prompt_end_idx])
if len(train_dataset) < cfg.prompt_end_idx:
    prompt_dataset = Dataset.from_dict(train_dataset[cfg.prompt_sta_idx-cfg.prompt_end_idx:])
prompt_dataloader = DataLoader(prompt_dataset, batch_size=1)

model, prompt_dataloader = accelerator.prepare(model, prompt_dataloader)

generated_dataset = {"text": []}

for text in tqdm(prompt_dataloader):
    prompt = (text["text"])
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(accelerator.device)
    clipped_ids = input_ids[:, :cfg.prompt_length]
    if hasattr(model, "module"):
        gen_tokens = model.module.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )
    else:
        gen_tokens = model.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )
    if model_type == "llama":
        gen_tokens = gen_tokens[:, 1:]
    print(model(gen_tokens, labels=gen_tokens).loss)
    gen_text = tokenizer.batch_decode(gen_tokens)
    generated_dataset["text"].extend(gen_text)

generated_dataset = Dataset.from_dict(generated_dataset)
save_dir = f"{cfg.cache_path}/{cfg.dataset_name}/{cfg.dataset_config_name}/refer@{cfg.model_name}/"
generated_dataset.save_to_disk(save_dir + f"{accelerator.device}")

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    concatenated_dataset = None
    for sub_dir in os.listdir(save_dir):
        data_path = os.path.join(save_dir, sub_dir)
        if os.path.isdir(data_path):
            if concatenated_dataset is None:
                concatenated_dataset = load_from_disk(data_path)
            else:
                dataset = load_from_disk(data_path)
                concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])
    concatenated_dataset.save_to_disk(save_dir)