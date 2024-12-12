# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import MistralConfig
from .language_model import LanguageModel


class Mistral(LanguageModel):
    """ A custom convenience wrapper around huggingface llama utils """

    def get_config(self):
        return MistralConfig()


