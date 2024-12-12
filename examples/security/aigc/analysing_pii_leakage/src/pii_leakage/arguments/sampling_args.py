# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from dataclasses import dataclass, field


@dataclass
class SamplingArgs:
    CONFIG_KEY = "sampling_args"

    N: int = field(default=10_000, metadata={
        "help": "Sample N batches from the LM"
    })

    prompt: str = field(default=None, metadata={
        "help": "the prompt for the LM"
    })

    top_k: int = field(default=40, metadata={
        "help": "Top-k sampling"
    })

    top_p: float = field(default=1.0, metadata={
        "help": "Top-p (nucleus) sampling. Top-K should be zero here."
    })

    seq_len: int = field(default=512, metadata={
        "help": "maximum length for the sample"
    })

    prompted: str = field(default="none", metadata={
        "help": "Which sampling mode to use"
    })

    as_probabilities: bool = field(default=True, metadata={
        "help": "whether to return scores for each token as probability. "
    })

    do_sample: bool = field(default=True, metadata={
        "help": "whether to sample the model. "
    })

    generate_verbose: bool = field(default=False, metadata={
        "help": "whether to generate data verbose."
    })

    def hash(self, suffix=""):
        """ Compute a unique hash based on this dict"""
        return hashlib.sha256(repr({
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seq_len": self.seq_len,
            "prompt": self.prompt,
            "suffix": suffix
        }).encode('utf-8')).hexdigest()
