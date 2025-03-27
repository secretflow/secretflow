from utils.evaluator import RoBERTaEvaluator
from utils.test import PATH_GPTFUZZ
from utils.dataset import JAIL_BREAK_ALIGNMENT_DATASETS_PATH
from utils.llm import (
    load_model,
    batch_generate_on_prompts,
    VICUNA_SYSTEM_PROMPT,
    load_adapters_from_pretrained,
    load_adapters_from_pretrained,
)
from utils.test import (
    load_default_model_from_pretrained,
    load_default_tokenizer,
    load_adapter_from_pretrained,
)
from datasets import Dataset, load_from_disk, concatenate_datasets
from utils.logging import logger
from utils.arguments import DATASETS_ROOT_PATH, RESULTS_ROOT_PATH, ADAPTERS_ROOT_PATH

import os

CKPT = "checkpoint-60"

ADAPTERS = {
    "Advanced": f"{ADAPTERS_ROOT_PATH}/Advanced/" + CKPT,
    "Basic": f"{ADAPTERS_ROOT_PATH}/Basic/" + CKPT,
    "Exception": f"{ADAPTERS_ROOT_PATH}/Exception/" + CKPT,
    "Start Prompt": f"{ADAPTERS_ROOT_PATH}/Startprompt/" + CKPT,
    "Toxic": f"{ADAPTERS_ROOT_PATH}/Toxic/" + CKPT,
    "DRA": f"{ADAPTERS_ROOT_PATH}/DRA/" + CKPT,
}

DATASET_ROOT_PATH = f"{DATASETS_ROOT_PATH}/jailbreak_alignment_datasets"

if __name__ == "__main__":
    evaluator_path = PATH_GPTFUZZ
    save_root_path = f"{RESULTS_ROOT_PATH}/evaluations/asr/mergetwo"
    device = "cuda"
    fast_mode = True
    max_new_tokens = 512
    batch_size = 16
    weights = [1.0, 1.0]
    evaluate_adapters = ["DRA", "Advanced"]
    evaluate_datasets = [
        "Advanced",
        "Basic",
        "Exception",
        "Start Prompt",
        "Toxic",
        "DRA",
    ]
    combination_type = "linear"

    file_name = "".join(evaluate_adapters) + f"_{CKPT}"
    if len(evaluate_adapters) >= 2:
        file_name += "_".join([str(w) for w in weights])
        logger.info(f"combination_type:{combination_type}")

    model = load_default_model_from_pretrained(logger=logger)

    if evaluate_adapters != []:
        model = load_adapters_from_pretrained(
            model,
            {
                "name": evaluate_adapters,
                "path": [ADAPTERS[adapter] for adapter in evaluate_adapters],
            },
            combination_type=combination_type,
        )

    logger.info(f"active adatpers : {model.active_adapters}")

    tokenizer = load_default_tokenizer()

    datasets = []

    for name in evaluate_datasets:
        evaluate_dataset = load_from_disk(f"{DATASET_ROOT_PATH}/{name}")["test"]
        if fast_mode:
            tailored_dataset = Dataset.from_dict(
                evaluate_dataset[: int(len(evaluate_dataset) / 2)]
            )
            datasets.append(tailored_dataset)
        else:
            datasets.append(evaluate_dataset)

    evaluate_dataset = concatenate_datasets(datasets)
    evaluate_dataset = evaluate_dataset.remove_columns(["outputs"])

    logger.info(evaluate_dataset)

    prompts = evaluate_dataset["prompts"]

    outputs = batch_generate_on_prompts(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        template=VICUNA_SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        skip_prompt=True,
        do_sample=True,
        logger=logger,
    )

    evaluate_dataset = evaluate_dataset.add_column("outputs", outputs)
    del model

    evaluator_model_name_or_path = os.path.expanduser(evaluator_path)
    robert_evaluator = RoBERTaEvaluator(evaluator_model_name_or_path, device)

    labels = []

    for idx in range(0, len(evaluate_dataset), batch_size):
        logger.info(
            f"Evaluating batch from index {idx} to {min(idx + batch_size - 1, len(evaluate_dataset))}:"
        )

        batch = outputs[idx : idx + batch_size]
        predicaitons = robert_evaluator.predict(batch)

        labels.extend(predicaitons)

    evaluate_dataset = evaluate_dataset.add_column("labels", labels)

    logger.info("Jailbreaked: " + str(evaluate_dataset["labels"].count(1)))
    evaluate_dataset.save_to_disk(save_root_path + "/" + f"{file_name}")
