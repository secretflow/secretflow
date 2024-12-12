# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import GPT2Config

from .language_model import LanguageModel


class GPT2(LanguageModel):
    """ A custom convenience wrapper around huggingface gpt-2 utils """

    def get_config(self):
        return GPT2Config()


