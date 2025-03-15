import os
from peft import TaskType
from .arguments import MODELS_ROOT_PATH

from .llm import (
    load_tokenizer,
    load_model,
    load_adapter_for_training,
    load_adapter_from_pretrained,
    create_lora_config,
    create_quantization_config,
)

from .logging import Logger


PATH_LLAMA_2_13B_CHAT_HF = f"{MODELS_ROOT_PATH}/Llama-2-13b-chat-hf"
PATH_VICUNA_13b_V1_5 = f"{MODELS_ROOT_PATH}/lmsys/vicuna-13b-v1.5"
PATH_GPTFUZZ = f"{MODELS_ROOT_PATH}/hubert233/GPTFuzz"

DEAFULT_MODEL = PATH_VICUNA_13b_V1_5


def load_default_tokenizer(model_name_or_path=DEAFULT_MODEL):

    model_name_or_path = os.path.expanduser(model_name_or_path)
    tokenizer = load_tokenizer(model_name_or_path=model_name_or_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    return tokenizer


def load_default_peft_model_for_training(
    model_name_or_path=DEAFULT_MODEL,
    device_map="auto",
    logger=None,
):
    logger = Logger(logger)

    logger.info(f"Loading basemodel from {model_name_or_path} ...")
    model_name_or_path = os.path.expanduser(model_name_or_path)
    quantization_config = create_default_quantization_config()
    basemodel = load_model(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    logger.info(f"Loading peft_model...")
    lora_config = create_default_lora_config()
    peft_model = load_adapter_for_training(
        base_model=basemodel,
        lora_config=lora_config,
    )

    logger.info(f"PEFT model loaded")
    return peft_model


def load_default_peft_model_from_pretrained(
    adapter_name_or_path,
    model_name_or_path=DEAFULT_MODEL,
    device_map="auto",
    logger=None,
):
    logger = Logger(logger)

    logger.info(f"Loading basemodel from {model_name_or_path} ...")
    model_name_or_path = os.path.expanduser(model_name_or_path)
    quantization_config = create_default_quantization_config()
    basemodel = load_model(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    logger.info(f"Loading peft_model...")
    adapter_name_or_path = os.path.expanduser(adapter_name_or_path)
    peft_model = load_adapter_from_pretrained(
        base_model=basemodel, adapter_name_or_path=adapter_name_or_path
    )

    logger.info(f"PEFT model loaded")
    return peft_model


def load_default_model_from_pretrained(
    model_name_or_path=DEAFULT_MODEL,
    device_map="auto",
    logger=None,
):
    logger = Logger(logger)

    logger.info(f"Loading model from {model_name_or_path} ...")
    model_name_or_path = os.path.expanduser(model_name_or_path)
    quantization_config = create_default_quantization_config()
    model = load_model(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    logger.info(f"Model loaded")
    return model


def create_default_lora_config():

    lora_config = create_lora_config(
        8,
        16,
        ["k_proj", "v_proj", "down_proj", "up_proj", "q_proj", "o_proj", "gate_proj"],
        0.05,
        TaskType.CAUSAL_LM,
    )

    return lora_config


def create_default_quantization_config():

    quantization_config = create_quantization_config(load_in_8bit=True)

    return quantization_config
