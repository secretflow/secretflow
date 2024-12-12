# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field

from transformers import IntervalStrategy, logging, TrainingArguments

from trl import SFTConfig

logger = logging.get_logger(__name__)


@dataclass
class TrainerArgs(SFTConfig):
    CONFIG_KEY = "trainer_args"

    dry_run: bool = field(
        default=False,
        metadata={
            "help": "Option for reducing training steps (2) and logging intervals (1) for quick sanity checking of arguments."}
    )

    resume_from_last_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Whether to continue training from the last checkpoint."
        }
    )

    save_steps: int = field(
        default=750,
        metadata={
            "help": "Steps after which the model is saved. "
        }
    )

    callback_after_n_steps: int = field(
        default=730,
        metadata={
            "help": "Invoke callbacks after n steps"
        }
    )

    limit_eval_dataset: int = field(
        default=5_000,
        metadata={
            "help": "For callback, limit number of samples to evaluate ppl on every n steps."
        }
    )

    per_device_train_batch_size: int = field(
        default=16,
        metadata={
            "help": "Per-device training batch size."
        }
    )

    per_device_eval_batch_size: int = field(
        default=16,
        metadata={
            "help": "Per-device eval batch size."
        }
    )

    num_train_epochs: int = field(
        default=4,
        metadata={
            "help": "Number of training epochs."
        }
    )

    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help": "Number of batches to accumulate."
        }
    )

    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to remove unused columns."
        }
    )

    # Since transformers>=4.35.0 this defaults to True
    # Opacus GradSampleModule is not compatible with this
    save_safetensors: bool = field(
        default=False,
        metadata={
            "help": "Whether to save safetensors."
        }
    )

    output_dir: str = field(
        default="",
        metadata={
            "help": "Unused."
        }
    )

    train_split: str = field(
        default="train",
        metadata={
            "help": "The split to train on."
        }
    )

    eval_split: str = field(
        default="test",
        metadata={
            "help": "The split to evaluate ppl on."
        }
    )

    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "The max_seq_length."
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.dry_run:
            logger.warning(
                "--dry_run was specified. Reducing number of training steps to 2 and logging intervals to 1...")
            self.logging_steps = 1
            self.logging_strategy = IntervalStrategy.STEPS
            self.eval_steps = 1
            self.evaluation_strategy = IntervalStrategy.STEPS

            self.max_steps = 2
