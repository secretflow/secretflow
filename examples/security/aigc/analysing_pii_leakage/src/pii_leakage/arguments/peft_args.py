# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field

from transformers import IntervalStrategy, logging, TrainingArguments

from peft import LoraConfig

logger = logging.get_logger(__name__)


@dataclass
class TrainerArgs(LoraConfig):
    CONFIG_KEY = "lora_args"

    dry_run: bool = field(
        default=False,
        metadata={
            "help": "Option for reducing training steps (2) and logging intervals (1) for quick sanity checking of arguments."}
    )