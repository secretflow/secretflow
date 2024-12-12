import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import random
import argparse
from attack.attack_model import AttackModel
from data.prepare import dataset_prepare
from attack.utils import Dict

import datasets
from datasets import Image, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--random_seed", type=int, default=0)
parser.add_argument("-m", "--model_name", type=str, default="gpt2")
parser.add_argument("-tm", "--target_model", type=str, default="./ft_llms/gpt2/ag_news/target/checkpoint-12500")
parser.add_argument("-rm", "--reference_model", type=str, default="./ft_llms/gpt2/ag_news/refer/checkpoint-3500")
parser.add_argument("-d", "--dataset_name", type=str, default="ag_news")
parser.add_argument("-dc", "--dataset_config_name", type=str, default=None,
                    help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--cache_path", type=str, default="./cache")
parser.add_argument("--use_dataset_cache", action="store_true", default=True)
parser.add_argument("--packing", action="store_true", default=True)
parser.add_argument("--calibration", action="store_true", default=True, help="Whether to enable calibration.")
parser.add_argument("--pad_token_id", default=None, type=int, help="The end of sequence token.")
parser.add_argument("--add_eos_token", action="store_true", help="Add EOS token to tokenizer", default=False)
parser.add_argument("--add_bos_token", action="store_true", help="Add BOS token to tokenizer", default=False)
parser.add_argument("--attack_kind", type=str, default="stat", help="Valid attacks: nn, stat")
parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size of the evaluation phase")
parser.add_argument("--maximum_samples", type=int, default=200, help="The maximum samples number for member and non-member records.")
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--validation_split_percentage", default=0.1)
parser.add_argument("--preprocessing_num_workers", type=int, default=1)
parser.add_argument("--mask_filling_model_name", type=str, default="t5-base")
parser.add_argument("--buffer_size", type=int, default=1)
parser.add_argument("--mask_top_p", type=float, default=1.0)
parser.add_argument("--span_length", type=int, default=2)
parser.add_argument("--pct", type=float, default=0.3, help="Pct words masked")
parser.add_argument("--ceil_pct", action="store_true", default=False)
parser.add_argument("--int8", action="store_true", default=False)
parser.add_argument("--half", action="store_true", default=False)
parser.add_argument("--perturbation_number", type=int, default=1, help="The number of different perturbation strength / position; debugging parameter, should be set to 1 in the regular running.")
parser.add_argument("--sample_number", type=int, default=10, help="The number of sampling")
parser.add_argument("--train_sta_idx", type=int, default=0)
parser.add_argument("--train_end_idx", type=int, default=10000)
parser.add_argument("--eval_sta_idx", type=int, default=0)
parser.add_argument("--eval_end_idx", type=int, default=1000)
parser.add_argument("--attack_data_path", type=str, default="attack")
parser.add_argument("--load_attack_data", action="store_true", default=False, help="Whether to load prepared attack data if existing.")

args = parser.parse_args()

cfg = Dict(vars(args))


# Add Logger
accelerator = Accelerator()
logger = get_logger(__name__, "INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename='attack_result.txt',
    filemode='a'
    )
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
logger.logger.addHandler(stream_handler)

# log attack config
logger.info(f'Attack config: {cfg}')

# Load abs path
PATH = os.path.dirname(os.path.abspath(__file__))

# Fix the random seed
seed = args.random_seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

## Load generation models.
if not cfg["load_attack_data"]:
    # config = AutoConfig.from_pretrained(cfg["model_name"])
    # config.use_cache = False
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target_model = AutoModelForCausalLM.from_pretrained(cfg["target_model"], quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                        torch_dtype=torch_dtype,
                                                        local_files_only=True,
                                                        config=AutoConfig.from_pretrained(cfg["model_name"]),
                                                        cache_dir=cfg["cache_path"])
    reference_model = AutoModelForCausalLM.from_pretrained(cfg["reference_model"], quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                           torch_dtype=torch_dtype,
                                                           local_files_only=True,
                                                           config=AutoConfig.from_pretrained(cfg["model_name"]),
                                                           cache_dir=cfg["cache_path"])


    logger.info("Successfully load models")
    config = AutoConfig.from_pretrained(cfg["model_name"])
    # Load tokenizer.
    model_type = config.to_dict()["model_type"]
    if model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(cfg["model_name"], add_eos_token=cfg["add_eos_token"],
                                                  add_bos_token=cfg["add_bos_token"], use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], add_eos_token=cfg["add_eos_token"],
                                                  add_bos_token=cfg["add_bos_token"], use_fast=True)


    if cfg["pad_token_id"] is not None:
        logger.info("Using pad token id %d", cfg["pad_token_id"])
        tokenizer.pad_token_id = cfg["pad_token_id"]

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
    train_dataset = Dataset.from_dict(train_dataset[cfg.train_sta_idx:cfg.train_end_idx])
    valid_dataset = Dataset.from_dict(valid_dataset[cfg.eval_sta_idx:cfg.eval_end_idx])
    train_dataset = Dataset.from_dict(train_dataset[random.sample(range(len(train_dataset["text"])), cfg["maximum_samples"])])
    valid_dataset = Dataset.from_dict(valid_dataset[random.sample(range(len(valid_dataset["text"])), cfg["maximum_samples"])])
    logger.info("Successfully load datasets!")

    # Prepare dataloade
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["eval_batch_size"])
    eval_dataloader = DataLoader(valid_dataset, batch_size=cfg["eval_batch_size"])

    # Load Mask-f
    shadow_model = None
    int8_kwargs = {}
    half_kwargs = {}
    if cfg["int8"]:
        int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    elif cfg["half"]:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["mask_filling_model_name"], **int8_kwargs, **half_kwargs).to(accelerator.device)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = AutoTokenizer.from_pretrained(cfg["mask_filling_model_name"], model_max_length=n_positions)

    # Prepare everything with accelerator
    train_dataloader, eval_dataloader = (
        accelerator.prepare(
            train_dataloader,
            eval_dataloader,
    ))
else:
    target_model = None
    reference_model = None
    shadow_model = None
    mask_model = None
    train_dataloader = None
    eval_dataloader = None
    tokenizer = None
    mask_tokenizer = None


datasets = {
    "target": {
        "train": train_dataloader,
        "valid": eval_dataloader
    }
}


attack_model = AttackModel(target_model, tokenizer, datasets, reference_model, shadow_model, cfg, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
attack_model.conduct_attack(cfg=cfg)
