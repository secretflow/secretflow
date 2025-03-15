from utils.dataset import load_datasets_mmlu
from utils.test import load_default_model_from_pretrained, load_default_tokenizer
from utils.logging import logger
from utils.arguments import RESULTS_ROOT_PATH, ADAPTERS_ROOT_PATH
from utils.llm import load_adapters_from_pretrained

import tqdm
import torch
import json
import os

options = ["A", "B", "C", "D"]

CKPT = "checkpoint-10"

ADAPTERS = {
    "Advanced": f"{ADAPTERS_ROOT_PATH}/Advanced/" + CKPT,
    "Basic": f"{ADAPTERS_ROOT_PATH}/Basic/" + CKPT,
    "Exception": f"{ADAPTERS_ROOT_PATH}/Exception/" + CKPT,
    "Start Prompt": f"{ADAPTERS_ROOT_PATH}/Startprompt/" + CKPT,
    "Toxic": f"{ADAPTERS_ROOT_PATH}/Toxic/" + CKPT,
    "DRA": f"{ADAPTERS_ROOT_PATH}/DRA/" + CKPT,
}


def format_question(question, choices, answer_id=None, include_answer=False):
    prompt = f"\n{question}"
    for idx, option in enumerate(options):
        prompt += f"\n{option}. {choices[idx]}"
    prompt += f"\nAnswer:"
    if include_answer:
        prompt += f" {options[answer_id]}\n\n"
    return prompt


def generate_few_shots_examples(subject, subject_train_dataset):
    prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n"

    for row in subject_train_dataset:
        prompt += format_question(row["question"], row["choices"], row["answer"], True)

    return prompt


if __name__ == "__main__":

    save_path = f"{RESULTS_ROOT_PATH}/mmlu_training/"
    batch_size = 12
    evaluate_adapters = ["DRA"]
    combination_type = "linear"
    file_name = "".join(evaluate_adapters) + f"_{CKPT}"

    mmlu_datasets = load_datasets_mmlu()

    subjects = sorted(list(set(mmlu_datasets["dev"]["subject"])))
    logger.info(f"({len(subjects)}) subjsects : {subjects}")

    tokenizer = load_default_tokenizer()
    model = load_default_model_from_pretrained()
    if evaluate_adapters != []:
        model = load_adapters_from_pretrained(
            model,
            {
                "name": evaluate_adapters,
                "path": [ADAPTERS[adapter] for adapter in evaluate_adapters],
            },
            combination_type=combination_type,
        )

    model.eval()

    logger.info(f"active adatpers : {model.active_adapters}")

    tokenizer = load_default_tokenizer()

    acc_subjects = dict([(subject, 0.0) for subject in subjects])
    max_lens_subjects = dict([(subject, 0) for subject in subjects])

    for subject in tqdm.tqdm(subjects):

        torch.cuda.empty_cache()

        subject_test_dataset = mmlu_datasets["test"].filter(
            lambda example: example["subject"] == subject
        )

        subject_train_dataset = mmlu_datasets["dev"].filter(
            lambda example: example["subject"] == subject
        )

        few_shot_prompt = generate_few_shots_examples(subject, subject_train_dataset)

        correct = 0
        length = len(subject_test_dataset)
        max_prompt_length = 0

        for idx in range(0, length, batch_size):

            batch = subject_test_dataset[idx : idx + batch_size]

            questions_batched = batch["question"]
            choicess_batched = batch["choices"]
            answers_batched = [options[idx] for idx in batch["answer"]]

            input_prompts_batched = [
                few_shot_prompt + format_question(question, choices)
                for question, choices in zip(questions_batched, choicess_batched)
            ]

            with torch.no_grad():
                input_ids_batched = tokenizer(
                    input_prompts_batched, padding=True, return_tensors="pt"
                ).to("cuda")
                max_prompt_length = max(
                    max_prompt_length, len(input_ids_batched["input_ids"][0])
                )
                output_ids_batched = model.generate(
                    input_ids=input_ids_batched["input_ids"],
                    attention_mask=input_ids_batched["attention_mask"],
                    max_new_tokens=1,
                    do_sample=False,
                )[:, -1].to("cpu")
            outputs_batched = [tokenizer.decode(ids) for ids in output_ids_batched]

            correct += [
                i == j for (i, j) in zip(answers_batched, outputs_batched)
            ].count(True)

        acc = correct / length
        acc_subjects[subject] = acc
        logger.info(f"Subject {subject} acc: {acc}")
        max_lens_subjects[subject] = max_prompt_length
        logger.info(f"max_prompt_length: {max_prompt_length}")

    print(acc_subjects)

    print("-" * 100)

    print(max_lens_subjects)

    os.makedirs(save_path, exist_ok=True)
    with open(save_path + "/" + f"{file_name}.json", "w") as json_file:
        json.dump(acc_subjects, json_file, indent=4)
