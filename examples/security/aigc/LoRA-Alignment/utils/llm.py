# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from .logging import Logger

LLAMA_SYSTEM_PROMPT = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST]"
VICUNA_SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n### USER:\n{instruction}\n### Assistant:\n"


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
    )
    return tokenizer


def load_model(model_name_or_path, device_map="auto", quantization_config=None):

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    return model


def load_adapter_for_training(base_model, lora_config):

    peft_model = prepare_model_for_kbit_training(base_model)
    peft_model = get_peft_model(peft_model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


def load_adapter_from_pretrained(base_model, adapter_name_or_path):

    peft_model = PeftModel.from_pretrained(base_model, adapter_name_or_path)

    return peft_model


def load_adapters_from_pretrained(
    base_model,
    adapters: dict[list],
    combination_type="linear",
    weights=None,
    logger=None,
):
    logger = Logger(logger)

    adapter_name = adapters["name"][0]
    adapter_path = adapters["path"][0]

    peft_model = PeftModel.from_pretrained(
        base_model, adapter_path, adapter_name=adapter_name
    )

    logger.info(f"adapters : {adapters['name']}")
    merged_adapter_name = "".join(adapters["name"])
    if len(adapters["name"]) > 1:

        for i in range(1, len(adapters["name"])):
            _ = peft_model.load_adapter(
                adapters["path"][i], adapter_name=adapters["name"][i]
            )

        density = 0.2
        if weights == None:
            weights = [1.0] * len(adapters["name"])
        peft_model.add_weighted_adapter(
            adapters["name"],
            weights,
            merged_adapter_name,
            combination_type=combination_type,
            density=density,
        )
        peft_model.set_adapter(merged_adapter_name)

    return peft_model


def create_lora_config(lora_r, lora_alpha, lora_modules, lora_dropout, task_type):

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        task_type=task_type,
        bias="none",
    )

    return lora_config


def create_quantization_config(load_in_8bit):

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
    )

    return quantization_config


def generate(
    model,
    tokenizer,
    prompt,
    template,
    max_new_tokens=512,
    skip_prompt=True,
    do_sample=True,
    logger=None,
):

    logger = Logger(logger)

    templated_prompt = template.format(instruction=prompt)
    input_ids = tokenizer(templated_prompt, return_tensors="pt")["input_ids"].to(
        model.device
    )
    logger.info("Generating response ...")
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample
        )[0]

    if skip_prompt:
        prompt_ids_length = len(input_ids[0])
        output = tokenizer.decode(output_ids[prompt_ids_length:])
    else:
        output = tokenizer.decode(output_ids)

    logger.info("Response generated.")

    return output


def batch_generate(
    model, tokenizer, batch, max_new_tokens=512, skip_prompt=True, do_sample=True
):

    # batched generating
    batched_encoding = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
    )
    batched_input_ids = batched_encoding["input_ids"].to("cuda")
    batched_attention_mask = batched_encoding["attention_mask"].to("cuda")
    model.eval()
    with torch.no_grad():
        batched_output_ids = model.generate(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    # remove inputing prompts from outputs
    length_list = [len(input_ids) for input_ids in batched_input_ids]
    # length_list = [input_ids.count_nonzero().item() for input_ids inbatched_input_ids]

    if skip_prompt:
        batched_output_ids = [
            output_ids[length_list[idx] :]
            for idx, output_ids in enumerate(batched_output_ids)
        ]
    generated_texts = tokenizer.batch_decode(
        batched_output_ids, skip_special_tokens=True
    )

    return generated_texts


def batch_generate_on_prompts(
    model,
    tokenizer,
    prompts,
    batch_size,
    template,
    max_new_tokens=800,
    skip_prompt=True,
    do_sample=True,
    logger=None,
):

    logger = Logger(logger)

    response = []
    logger.info(
        f"Batched generating on {len(prompts)} prompts with batch size: {batch_size}"
    )

    for idx in range(0, len(prompts), batch_size):
        logger.info(
            f"Generating batch from index {idx} to {min(len(prompts), idx + batch_size - 1)}:"
        )
        batch = prompts[idx : idx + batch_size]
        batch = [template.format(instruction=instruct) for instruct in batch]
        generated_texts = batch_generate(
            model,
            tokenizer,
            batch,
            max_new_tokens=max_new_tokens,
            skip_prompt=skip_prompt,
            do_sample=do_sample,
        )
        response.extend(generated_texts)

    logger.info("Resopnses generated.")

    return response
