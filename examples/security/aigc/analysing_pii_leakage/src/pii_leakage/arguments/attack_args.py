# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import List

from .ner_args import PII_ENTITIES


@dataclass
class AttackArgs:
    CONFIG_KEY = "attack_args"

    attack_name: str = field(default="naive_extraction", metadata={
        "help": "number of workers",
        "choices": ["perplexity_inference", "perplexity_reconstruction", "naive_extraction"]
    })

    pii_class: str = field(default=PII_ENTITIES.PERSON, metadata={
        "help": "the PII class to attack"
    })

    target_sequence: str = field(default="", metadata={
        "help": "the sequence to be attacked for PII reconstruction & inference. "
                "Replace the PII with <T-MASK> and other PII with <MASK>. "
    })

    pii_candidates: List[str] = field(default_factory=lambda: [], metadata={
        "help": "PII candidates for a PII inference attack. Please ensure the casing is correct. "
    })

    candidate_size: int = field(default=10, metadata={
        "help": "Number of PII candidates to sample randomly for the attack. Will be overriden by pii_candidates."
    })

    sampling_rate: int = field(default=32, metadata={
        "help": "number of times to attempt generating candidates."
    })

    seq_len: int = field(default=64, metadata={
        "help": "number of tokens to sample per sampled sequence."
    })
