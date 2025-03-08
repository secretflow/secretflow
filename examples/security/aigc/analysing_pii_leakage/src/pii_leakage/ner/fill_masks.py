# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from copy import deepcopy
from transformers import pipeline

class FillMasks:
    def __init__(self):
        self.nlp = pipeline('fill-mask', model='roberta-base')
        self.model_max_length = self.nlp.tokenizer.model_max_length

    def fill_masks(self, sample: str):
        """
        Replace "<MASK>" in the sample with mask tokens and use BERT to predict replacements.
        Ignore "<T-MASK>" (the target mask).
        :param sample: The original text.
        :return: The text with replaced entities.
        """
        # Create a deep copy of the original sample
        masked_sample = deepcopy(sample)

        # Replace all occurrences of "<MASK>" in the text with the actual mask token used by BERT
        masked_sample = masked_sample.replace("<MASK>", self.nlp.tokenizer.mask_token)

        # Loop over each occurrence of the mask token in the text
        while masked_sample.count(self.nlp.tokenizer.mask_token) > 0:
            # Find the index of the first mask token in the text
            first_mask_idx = masked_sample.index(self.nlp.tokenizer.mask_token)

            # Identify the start and end indices of the chunk
            chunk_start = max(0, first_mask_idx - self.model_max_length // 2)
            chunk_end = chunk_start + self.model_max_length

            # If the chunk ends before the last token, adjust the start and end indices
            if chunk_end < len(masked_sample):
                if " " in masked_sample[chunk_end:]:
                    chunk_end = chunk_end + masked_sample[chunk_end:].index(" ")
                chunk_start = chunk_end - self.model_max_length
            else:
                chunk_end = len(masked_sample)
                chunk_start = max(0, chunk_end - self.model_max_length)

            # Extract the chunk from the text
            chunk = masked_sample[chunk_start:chunk_end]

            # Predict a replacement for the first mask token in the chunk
            if self.nlp.tokenizer.mask_token in chunk:
                filled_output = self.nlp(chunk, top_k=1)

                # Check if filled_output is list of lists, and if so, get the first list
                if isinstance(filled_output[0], list):
                    filled_output = filled_output[0]

                # Get the filled sequence from the first item in filled_output
                chunk_filled = filled_output[0]['sequence'].replace("<s>", "").replace("</s>", "")

                # Update the text with the replaced mask token
                masked_sample = masked_sample[:chunk_start] + chunk_filled + masked_sample[chunk_end:]
            else:
                break

        # Return the text with replaced entities
        return masked_sample
