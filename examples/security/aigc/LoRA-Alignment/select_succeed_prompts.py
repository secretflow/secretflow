from datasets import Dataset, DatasetDict
from collections import Counter
from utils.logging import logger
from utils.arguments import RESULTS_ROOT_PATH


if __name__ == "__main__":
    dataset_path = f"{RESULTS_ROOT_PATH}/evaluated_jailbreak_attempts"
    save_path = f"{RESULTS_ROOT_PATH}/succeed_jailbreak_prompts"

    datasets = Dataset.load_from_disk(dataset_path)
    succeed_dataset = datasets.filter(lambda example: example["labels"] == 1)

    succeed_prompts = zip(succeed_dataset["prompts"], succeed_dataset["communities"])
    succeed_prompts = list(set([i for i in succeed_prompts]))

    counted_comms = Counter([i[1] for i in succeed_prompts]).most_common()
    logger.info(counted_comms)
    succeed_comms = set(succeed_dataset["communities"])

    succeed_dataset_deduplicated = Dataset.from_dict(
        {
            "prompts": [item[0] for item in succeed_prompts],
            "communities": [item[1] for item in succeed_prompts],
        }
    )

    dataset_dict = DatasetDict()
    for comm in succeed_comms:
        dataset_dict[comm] = succeed_dataset_deduplicated.filter(
            lambda example: example["communities"] == comm
        )
    dataset_dict.save_to_disk(save_path)
