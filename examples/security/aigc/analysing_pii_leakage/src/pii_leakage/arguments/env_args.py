# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from dataclasses import dataclass, field


@dataclass
class EnvArgs:
    CONFIG_KEY = "env_args"

    num_workers: int = field(default=2, metadata={
        "help": "number of workers"
    })

    log_every: int = field(default=100, metadata={
        "help": "log interval for training"
    })

    save_every: int = field(default=249, metadata={
        "help": "save interval for training"
    })

    device: str = field(default="cuda", metadata={
        "help": "device to run observers on"
    })

    batch_size: int = field(default=64, metadata={
        "help": "default batch size for training"
    })

    eval_batch_size: int = field(default=32, metadata={
        "help": "default batch size for inference"
    })

    verbose: bool = field(default=True, metadata={
        "help": "whether to print out to the cmd line"
    })
