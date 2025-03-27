from utils.logging import logging

logger = logging.getLogger(__name__)

from utils.logging import Logger
from utils.test import *
from utils.llm import *
from utils.arguments import RESULTS_ROOT_PATH
from utils.dataset import (
    load_dataset_TrustAIRLab_forbidden_questions,
    load_dataset_TrustAIRLab_jailbreak_prompts,
)
from collections import Counter
from datasets import Dataset
import re


def replace_GPT_with_other(example, other="VICUNA"):
    # replace gpt with llama
    pattern = r"\b(?:chat\s*-?\s*gpt|chatgpt)\b"
    prompt = re.sub(pattern, other, example["prompt"], flags=re.IGNORECASE)
    return {"prompt": prompt}


def evaluate_jailbreak_tests(
    path_to_save_results,
    questions_dataset,
    prompts_dataset,
    model,
    tokenizer,
    batch_size,
    max_new_tokens,
    repeat_times,
    logger=None,
):

    logger = Logger(logger)
    
    counted_comms = Counter(prompts_dataset["community"]).most_common()
    policy_names = questions_dataset['content_policy_name']
    questions = questions_dataset['question']

    for count in range(repeat_times):
        for idxq, (question, policy_name) in enumerate(zip(questions, policy_names)):
            for idxc, comm in enumerate(counted_comms):

                dataset_idx = f"count_{count}_ques_{idxq}_com_{idxc}"
                community = comm[0]
                logger.info(
                    f"counts({count+1}/{repeat_times}); question({idxq+1}/{len(questions)}): {question}; community({idxc+1}/{len(counted_comms)}): {community}"
                )
                if dataset_idx in os.listdir(path_to_save_results):
                    logger.info("Already evaluated, passed.")
                    continue

                filtered = prompts_dataset.filter(
                    lambda example: example["community"] == community
                )
                filtered = filtered.map(replace_GPT_with_other)
                prompts = filtered["prompt"]
                bad_prompts = [prompt + f"\n{question}" for prompt in prompts]
                outputs = batch_generate_on_prompts(
                    model,
                    tokenizer,
                    bad_prompts,
                    batch_size=batch_size,
                    template=VICUNA_SYSTEM_PROMPT,
                    max_new_tokens=max_new_tokens,
                    skip_prompt=True,
                    do_sample=True,
                    logger=logger,
                )

                assert len(prompts) == len(outputs)

                results = Dataset.from_dict(
                    {
                        "prompts": prompts,
                        "questions": [question] * len(prompts),
                        "communities": [community] * len(prompts),
                        "outputs": outputs,
                        "counts": [count+1] * len(prompts),
                        "policy_names": [policy_name] * len(prompts),
                    }
                )

                results.save_to_disk(f"{path_to_save_results}/{dataset_idx}")


if __name__ == "__main__":

    path_to_save_results = f"{RESULTS_ROOT_PATH}/jailbreak_evaluations"
    repeat_times = 1
    # forbidden questions set have 14 categories, choose 1 for every category here.
    indices_for_selecting_questions = [30 * i for i in range(0, 13)]
    batch_size = 10
    max_new_tokens = 512

    prompts_dataset = load_dataset_TrustAIRLab_jailbreak_prompts()
    logger.info(prompts_dataset)

    tokenizer = load_default_tokenizer()
    # load default peft
    # model = load_default_model_from_pretrained(logger=logger)

    questions_dataset = load_dataset_TrustAIRLab_forbidden_questions()
    selected_questions = questions_dataset[indices_for_selecting_questions]

    prompts_dataset = prompts_dataset.filter(
        lambda example: example["community"] != None
    )
    logger.info(prompts_dataset)

    evaluate_jailbreak_tests(
        path_to_save_results=path_to_save_results,
        questions_dataset=selected_questions,
        prompts_dataset=prompts_dataset,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        repeat_times=repeat_times,
        logger=logger,
    )
