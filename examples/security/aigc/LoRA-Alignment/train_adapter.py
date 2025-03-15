from utils.logging import logging, Logger

logger = logging.getLogger(__name__)

from utils.arguments import (
    ModelArguments,
    DatasetArguments,
    TrainingArguments,
    PEFTArguments,
    ExperimentArguments,
    MODELS_ROOT_PATH,
)

from utils.llm import (
    VICUNA_SYSTEM_PROMPT,
    load_tokenizer,
    load_model,
    load_adapter_for_training,
    create_lora_config,
    create_quantization_config,
)

from utils.dataset import (
    load_dataset_jailbreak_alignment,
    load_dataset_alignment,
)

import torch
import time
import os
import json
import datasets
import transformers
from datasets import load_dataset, DatasetDict, concatenate_datasets
from dataclasses import dataclass
from argparse import ArgumentParser
from functools import partial
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from typing import (
    Tuple,
)


def parse_arguments() -> Tuple[
    ModelArguments,
    DatasetArguments,
    TrainingArguments,
    PEFTArguments,
    ExperimentArguments,
]:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DatasetArguments,
            TrainingArguments,
            PEFTArguments,
            ExperimentArguments,
        )
    )

    args = parser.parse_args()
    args_model, args_dataset, args_training, args_peft, args_exp = (
        parser.parse_args_into_dataclasses()
    )

    if args_model.system_prompt == "":
        if "vicuna" in args_model.model_name_or_path.lower():
            args_model.system_prompt = VICUNA_SYSTEM_PROMPT
        else:
            raise NotImplementedError

    # generate name of the output dir and saves the arguments
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args_exp.expriment_name:
        args_exp.output_dir = f"{args_exp.save_dir}/{timestamp}_{args_exp.expriment_name}_{args_dataset.jailbreak_alignment_datasets_name}"
    else:
        args_exp.output_dir = f"{args_exp.save_dir}/{timestamp}_{args_dataset.jailbreak_alignment_datasets_name}"
    os.makedirs(args_exp.output_dir, exist_ok=True)

    # saving configs
    args_dict = vars(args)
    with open(f"{args_exp.output_dir}/arguments.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    if os.path.exists(f"{MODELS_ROOT_PATH}/{args_model.model_name_or_path}"):
        args_model.model_name_or_path = f"{MODELS_ROOT_PATH}/{args.model_name_or_path}"

    return args_model, args_dataset, args_training, args_peft, args_exp


def process_oasst1_for_partial_loss(tokenizer, system_template, example):
    instruct, response = (
        example["text"].split("### Human: ")[1].split("### Assistant: ")
    )
    # tokenized_context = tokenizer("### Human: " + instruct + "\n" + "### Assistant: ")["input_ids"]
    tokenized_context = tokenizer(system_template.format(instruction=instruct))[
        "input_ids"
    ]
    tokenized_response = tokenizer(response)["input_ids"]
    context = tokenized_context
    response = tokenized_response[1:] + [tokenizer.eos_token_id]
    input_ids = context + response
    attention_mask = [1] * len(input_ids)
    label = [-100] * len(context) + response

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": label,
    }


def process_jailbreak_for_partial_loss(tokenizer, system_template, max_length, example):
    instruct = example["prompts"]
    response = example["outputs"]

    tokenized_context = tokenizer(system_template.format(instruction=instruct))[
        "input_ids"
    ]
    tokenized_response = tokenizer(response, max_length=max_length, truncation=True)[
        "input_ids"
    ]
    context = tokenized_context
    response = tokenized_response[1:] + [tokenizer.eos_token_id]

    input_ids = context + response
    attention_mask = [1] * len(input_ids)
    label = [-100] * len(context) + response

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": label,
    }


def process_jailbreak_for_full_loss(tokenizer, system_template, max_length, example):
    instruct = example["prompts"]
    response = example["outputs"]

    tokenized_context = tokenizer(system_template.format(instruction=instruct))[
        "input_ids"
    ]
    tokenized_response = tokenizer(response, max_length=max_length, truncation=True)[
        "input_ids"
    ]
    context = tokenized_context
    response = tokenized_response[1:] + [tokenizer.eos_token_id]

    input_ids = context + response
    attention_mask = [1] * len(input_ids)
    # label = [-100] * len(context) + response

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": input_ids,
    }


def load_alignment_datasets(alignment_datasets_name, tokenizer, system_prompt):

    process_oasst1 = partial(process_oasst1_for_partial_loss, tokenizer, system_prompt)
    alignment_datasets = load_dataset_alignment(alignment_datasets_name)
    alignment_datasets = alignment_datasets.map(
        process_oasst1, remove_columns=alignment_datasets["train"].column_names
    )
    return alignment_datasets


def load_jailbreak_alignment_datasets(
    dataset_name,
    tokenizer,
    system_prompt,
    max_length,
):

    process_jailbreak = partial(
        process_jailbreak_for_partial_loss, tokenizer, system_prompt, max_length
    )

    jailbreak_alignment_datasets = load_dataset_jailbreak_alignment(dataset_name)
    jailbreak_alignment_datasets = jailbreak_alignment_datasets.map(
        process_jailbreak,
        remove_columns=jailbreak_alignment_datasets["train"].column_names,
        load_from_cache_file=False,
    )

    return jailbreak_alignment_datasets


def mix_datasets(
    alignment_datasets,
    jailbreak_alignment_datasets,
    ps_radio,
):

    if jailbreak_alignment_datasets != None:
        number_of_align_training_samples = min(
            len(alignment_datasets["train"]),
            int(len(jailbreak_alignment_datasets["train"]) * (1 // ps_radio - 1)),
        )
        number_of_align_test_samples = min(
            len(alignment_datasets["test"]),
            int(len(jailbreak_alignment_datasets["test"]) * (1 // ps_radio - 1)),
        )

        mixed_datasets = DatasetDict()
        mixed_datasets["train"] = concatenate_datasets(
            [
                alignment_datasets["train"]
                .shuffle(seed=42)
                .select(range(number_of_align_training_samples)),
                jailbreak_alignment_datasets["train"],
            ]
        )
        mixed_datasets["test"] = concatenate_datasets(
            [
                alignment_datasets["test"]
                .shuffle(seed=42)
                .select(range(number_of_align_test_samples)),
                jailbreak_alignment_datasets["test"],
            ]
        )

        return mixed_datasets
    else:
        return jailbreak_alignment_datasets


def load_peft_model_for_training(
    model_name_or_path,
    device_map,
    args_peft: PEFTArguments,
    logger=None,
):
    logger = Logger(logger)

    logger.info(f"Loading basemodel from {model_name_or_path} ...")

    quantization_config = create_quantization_config(
        args_peft.bits == 8,
    )
    basemodel = load_model(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    logger.info(f"Loading peft_model...")
    lora_config = create_lora_config(
        lora_r=args_peft.lora_r,
        lora_alpha=args_peft.lora_alpha,
        lora_modules=args_peft.lora_modules,
        lora_dropout=args_peft.lora_dropout,
        task_type="CAUSAL_LM",
    )
    peft_model = load_adapter_for_training(
        base_model=basemodel,
        lora_config=lora_config,
    )

    logger.info(f"PEFT model loaded")
    return peft_model


@dataclass
class DataCollatorNoPadding:

    def __call__(self, instances):
        input_ids = [sample["input_ids"] for sample in instances]
        attention_mask = [sample["attention_mask"] for sample in instances]
        labels = [sample["label"] for sample in instances]

        input_ids = self.pad_sequence(input_ids)
        attention_mask = self.pad_sequence(attention_mask)
        labels = self.pad_sequence(labels)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return batch

    def pad_sequence(self, sequences, padding_value=0):
        longest_length = max([len(seq) for seq in sequences])
        for seq in sequences:
            padding_length = longest_length - len(seq)
            seq.extend([padding_value] * padding_length)
        return sequences


def train_peft_model(
    model,
    tokenizer,
    dataset_dict,
    args_exp: ExperimentArguments,
    args_training: TrainingArguments,
    logger=logger,
):
    logger = Logger(logger)

    training_args = transformers.TrainingArguments(
        output_dir=args_exp.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=args_training.per_device_train_batch_size,
        per_device_eval_batch_size=args_training.per_device_eval_batch_size,
        gradient_accumulation_steps=args_training.gradient_accumulation_steps,
        # num_train_epochs=args_training.num_train_epochs,
        max_steps=args_training.max_steps,
        eval_steps=args_training.eval_steps,
        save_steps=args_training.save_steps,
        lr_scheduler_type="constant",
        learning_rate=args_training.learning_rate,
        warmup_ratio=args_training.warmup_ratio,
        adam_beta2=args_training.adam_beta2,
        max_grad_norm=args_training.max_grad_norm,
        weight_decay=args_training.weight_decay,
    )

    data_collator = DataCollatorNoPadding()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Traning starts.")
    trainer.train()

    logger.info("Saving final trained adapter...")
    model.save_pretrained(args_exp.output_dir)
    logger.info("Saved")


if __name__ == "__main__":

    args_model, args_dataset, args_training, args_peft, args_exp = parse_arguments()

    tokenizer = load_tokenizer(args_model.model_name_or_path)

    if args_dataset.add_alignment_datasets:
        jailbreak_alignment_datasets = load_jailbreak_alignment_datasets(
            dataset_name=args_dataset.jailbreak_alignment_datasets_name,
            tokenizer=tokenizer,
            system_prompt=args_model.system_prompt,
            max_length=args_dataset.max_length,
        )
    else:
        jailbreak_alignment_datasets = None

    logger.info(f"jailbreak_alignment_datasets: {jailbreak_alignment_datasets}")

    alignment_datasets = load_alignment_datasets(
        alignment_datasets_name=args_dataset.alignment_datasets_name,
        tokenizer=tokenizer,
        system_prompt=args_model.system_prompt,
    )
    logger.info(f"alignment_datasets: {alignment_datasets}")

    mixed_datasets = mix_datasets(
        alignment_datasets=alignment_datasets,
        jailbreak_alignment_datasets=jailbreak_alignment_datasets,
        ps_radio=args_dataset.ps_radio,
    )

    logger.info(f"mixed_datasets: {mixed_datasets}")

    model = load_peft_model_for_training(
        model_name_or_path=args_model.model_name_or_path,
        args_peft=args_peft,
        device_map=args_model.device_map,
        logger=logger,
    )

    train_peft_model(
        model,
        tokenizer,
        mixed_datasets,
        args_exp,
        args_training,
        logger,
    )
