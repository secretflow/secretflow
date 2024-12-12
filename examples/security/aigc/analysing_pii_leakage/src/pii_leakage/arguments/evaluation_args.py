# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field

@dataclass
class EvaluationArgs:
    CONFIG_KEY = "eval_args"

    num_sequences: int = field(default=100, metadata={
        "help": "number of sequences to evaluate"
    })

    num_candidates: int = field(default=10, metadata={
        "help": "number of pii candidates in PII inference attacks to consider"
    })