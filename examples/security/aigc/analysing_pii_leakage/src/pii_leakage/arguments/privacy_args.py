# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field

@dataclass
class PrivacyArgs:
    CONFIG_KEY = "privacy_args"

    target_epsilon: float = field(default=-1, metadata={
        "help": "[OPTIONAL] Epsilon for DP. Default=-1, meaning that DP is not applied "
    })

    target_delta: float = field(default=None, metadata={
        "help": "[OPTIONAL] Delta for DP. If None, defaults to 1/{len(train_set)}."
    })

    noise_multiplier: float = field(default=None, metadata={
        "help": "[OPTIONAL] Fix a value for the noise multiplier. If value is set, target_epsilon is ignored. "
    })

    eps_error: float = field(default=0.1, metadata={
        "help": "Allowed error in epsilon"
    })

    max_grad_norm_dp: float = field(default=1.0, metadata={
        "help": "The maximum grad norm for clipping DP."
    })

    lora_dim: int = field(default=0, metadata={
        "help": "LoRA dimension; 0 means LoRA is disabled"
    })

    dp_lora_dropout: float = field(default=0.0, metadata={
        "help": "Dropout probability for LoRA layers"
    })

    dp_lora_alpha: int = field(default=32, metadata={
        "help": "LoRA attention alpha"
    })

    no_accountant: bool = field(default=False, metadata={
        "help": "Do not use a privacy accountant."
    })
