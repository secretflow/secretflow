from .arguments import DATASETS_ROOT_PATH
from .logging import Logger
from datasets import load_dataset, load_from_disk
import os

JAIL_BREAK_ALIGNMENT_DATASETS_PATH = {
    "Advanced": f"{DATASETS_ROOT_PATH}/jailbreak_alignment_datasets/Advanced",
    "Basic": f"{DATASETS_ROOT_PATH}/jailbreak_alignment_datasets/Basic",
    "Exception": f"{DATASETS_ROOT_PATH}/jailbreak_alignment_datasets/Exception",
    "Start Prompt": f"{DATASETS_ROOT_PATH}/jailbreak_alignment_datasets/Start Prompt",
    "Toxic": f"{DATASETS_ROOT_PATH}/jailbreak_alignment_datasets/Toxic",
    "DRA": f"{DATASETS_ROOT_PATH}/jailbreak_alignment_datasets/DRA",
}

ALIGNMENT_DATASETS_PATH = {
    "timdettmers/openassistant-guanaco": f"{DATASETS_ROOT_PATH}/timdettmers/openassistant-guanaco",
}


def load_dataset_TrustAIRLab_jailbreak_prompts(
    dataset_name_or_path=f"{DATASETS_ROOT_PATH}/TrustAIRLab/in-the-wild-jailbreak-prompts",
):
    if os.path.exists(dataset_name_or_path):
        print(f"Loading files from specified path: {dataset_name_or_path}")
        dataset = load_dataset(
            dataset_name_or_path, "jailbreak_2023_12_25", split="train"
        )
    else:
        print(f"Loading files via hugging face")
        dataset = load_dataset(
            "TrustAIRLab/in-the-wild-jailbreak-prompts",
            "jailbreak_2023_05_07",
            split="train",
        )

    return dataset


def load_dataset_TrustAIRLab_forbidden_questions(
    dataset_name_or_path=f"{DATASETS_ROOT_PATH}/TrustAIRLab/forbidden_question_set",
):
    if os.path.exists(dataset_name_or_path):
        print(f"Loading files from specified path: {dataset_name_or_path}")
        dataset = load_dataset(dataset_name_or_path, split="train")
    else:
        print(f"Loading files via hugging face")
        dataset = load_dataset("TrustAIRLab/forbidden_question_set", split="train")

    return dataset


def load_datasets_mmlu(
    dataset_name_or_path=f"{DATASETS_ROOT_PATH}/cais/mmlu", split="all"
):
    if os.path.exists(dataset_name_or_path):
        print(f"Loading files from specified path: {dataset_name_or_path}")
        datasets = load_dataset(dataset_name_or_path, split)
    else:
        print(f"Loading files via hugging face")
        datasets = load_dataset("cais/mmlu", split)

    return datasets


def load_dataset_jailbreak_evaluations(
    dataset_path=f"{DATASETS_ROOT_PATH}/jailbreak_evaluations",
):
    print(f"Loading files from specified path: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    return dataset


def load_dataset_jailbreak_alignment(dataset_name, logger=None):
    logger = Logger(logger)

    dataset_path = JAIL_BREAK_ALIGNMENT_DATASETS_PATH.get(dataset_name)

    if dataset_path == None:
        raise NotImplementedError
    else:
        datasets = load_from_disk(dataset_path)

    return datasets


def load_dataset_alignment(dataset_name, logger=None):
    logger = Logger(logger)

    dataset_path = ALIGNMENT_DATASETS_PATH[dataset_name]

    if dataset_path == None:
        raise NotImplementedError
    else:
        try:
            datasets = load_from_disk(dataset_path)
        except:
            datasets = load_dataset(dataset_path)

    return datasets
