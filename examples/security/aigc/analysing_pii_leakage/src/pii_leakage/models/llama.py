# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import LlamaConfig
from .language_model import LanguageModel


class Llama(LanguageModel):
    """ A custom convenience wrapper around huggingface llama utils """

    def get_config(self):
        return LlamaConfig()


