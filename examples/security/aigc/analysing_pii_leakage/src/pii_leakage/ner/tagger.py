# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
from typing import List, Tuple, Union

from ..arguments.env_args import EnvArgs
from ..arguments.ner_args import NERArgs
from .pii_results import ListPII


class Tagger:

    def __init__(self, ner_args: NERArgs, env_args: EnvArgs = None):
        self.ner_args: NERArgs = ner_args
        self.env_args: EnvArgs = env_args if env_args is not None else EnvArgs()
        self.base_tagger = self._load()

    @abstractmethod
    def _load(self):
        """ Load the base tagger model. """
        raise NotImplementedError

    @abstractmethod
    def analyze(self, text: Union[List[str], str]) -> ListPII:
        """ Analyze the text for entities. """
        raise NotImplementedError

    @abstractmethod
    def pseudonymize(self, text: str) -> Tuple[str, ListPII]:
        """ Analyze PII in the text and pseudonymize it. """
        raise NotImplementedError

    @abstractmethod
    def get_entity_classes(self) -> List[str]:
        """ Analyze the text for entities. """
        raise NotImplementedError
