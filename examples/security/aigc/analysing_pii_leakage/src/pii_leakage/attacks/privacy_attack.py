# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC

from ..arguments.attack_args import AttackArgs
from ..arguments.env_args import EnvArgs
from ..arguments.ner_args import NERArgs


class PrivacyAttack:

    def __init__(self, attack_args: AttackArgs, ner_args: NERArgs = None, env_args: EnvArgs = None):
        self.attack_args = attack_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.ner_args = ner_args if ner_args is not None else NERArgs()

    def attack(self, *args, **kwargs):
        raise NotImplementedError


class ExtractionAttack(PrivacyAttack, ABC):
    pass


class ReconstructionAttack(PrivacyAttack, ABC):
    pass
