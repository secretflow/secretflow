import random
from datasets import load_from_disk, Dataset, DatasetDict
from utils.logging import logger
from utils.arguments import RESULTS_ROOT_PATH, DATASETS_ROOT_PATH
from functools import partial


def add_random_questions(questions, example):
    question = random.choice(questions)
    prompt = example["prompts"] + "\n" + question["questions"]
    community = example["communities"]
    policy_name = question["policy_names"]
    output = question["outpus"]

    return {
        "questions": question["questions"],
        "prompts": prompt,
        "communities": community,
        "policy_names": policy_name,
        "outputs": output,
    }


if __name__ == "__main__":

    prompts_datasets_path = f"{RESULTS_ROOT_PATH}/succeed_jailbreak_prompts"
    responses_dataset_path = f"{RESULTS_ROOT_PATH}/benign_responses"
    alignment_datasets_save_root_path = (
        f"{DATASETS_ROOT_PATH}/jailbreak_alignment_datasets/"
    )
    prompts_split_radio = 0.25

    prompts_num = 40
    question_per_prompts_num = 10

    prompts_datasets = load_from_disk(prompts_datasets_path)
    logger.info("Prompts dataset:")
    logger.info(prompts_datasets)
    responses_dataset = load_from_disk(responses_dataset_path).train_test_split(
        prompts_split_radio
    )
    logger.info("Responses dataset:")
    logger.info(responses_dataset)

    for comm in list(prompts_datasets.keys()):

        if len(prompts_datasets[comm]) < prompts_num:
            continue

        comm_dataset = Dataset.from_dict(prompts_datasets[comm][:prompts_num])
        comm_dataset = comm_dataset.train_test_split(prompts_split_radio)

        comm_dataset["train"] = comm_dataset["train"].repeat(question_per_prompts_num)
        test_set_map_funciton = partial(
            add_random_questions, responses_dataset["train"].to_list()
        )
        comm_dataset["train"] = comm_dataset["train"].map(test_set_map_funciton)

        comm_dataset["test"] = comm_dataset["test"].repeat(question_per_prompts_num)
        test_set_map_funciton = partial(
            add_random_questions, responses_dataset["test"].to_list()
        )
        comm_dataset["test"] = comm_dataset["test"].map(test_set_map_funciton)

        alignment_datasets_save_path = f"{alignment_datasets_save_root_path}/{comm}"
        comm_dataset.save_to_disk(alignment_datasets_save_path)
        logger.info(f"jailbreak alignment datasets ({comm}): {comm_dataset}")
        logger.info(f"Saved to path: {alignment_datasets_save_path}")
