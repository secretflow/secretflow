from utils.dataset import load_dataset_TrustAIRLab_forbidden_questions
from utils.arguments import RESULTS_ROOT_PATH
from utils.test import (
    load_default_model_from_pretrained,
    load_default_tokenizer,
    PATH_LLAMA_2_13B_CHAT_HF,
)
from utils.llm import (
    batch_generate_on_prompts,
    LLAMA_SYSTEM_PROMPT,
)
from utils.logging import logger
from datasets import Dataset

excluded_indices = [5, 8, 10, 11, 12]
save_path = f"{RESULTS_ROOT_PATH}/benign_responses"
batch_size = 10

dataset = load_dataset_TrustAIRLab_forbidden_questions()
dataset = dataset.filter(
    lambda example: example["content_policy_id"] not in excluded_indices
)

tokenizer = load_default_tokenizer(PATH_LLAMA_2_13B_CHAT_HF)


model = load_default_model_from_pretrained(PATH_LLAMA_2_13B_CHAT_HF)


questions = dataset["question"]
policy_names = dataset["content_policy_name"]


outputs = batch_generate_on_prompts(
    model,
    tokenizer=tokenizer,
    prompts=questions,
    batch_size=batch_size,
    template=LLAMA_SYSTEM_PROMPT,
    max_new_tokens=512,
    logger=logger,
)


begign_dataset = Dataset.from_dict(
    {
        "questions": questions,
        "outpus": outputs,
        "policy_names": policy_names,
    }
)

begign_dataset.save_to_disk(save_path)
