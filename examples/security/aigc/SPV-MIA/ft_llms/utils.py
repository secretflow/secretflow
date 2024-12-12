import logging
from typing_extensions import Literal
from rich.logging import RichHandler
from torch.utils.data import IterableDataset
import warnings
import random
import torch


def get_logger(name: str, level: Literal["info", "warning", "debug"]) -> logging.Logger:
    rich_handler = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[level.upper()])

    if not logger.handlers:
        logger.addHandler(rich_handler)

    logger.propagate = False

    return logger

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def constantlengthdatasetiter(self):
    iterator = iter(self.dataset)
    more_examples = True
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= self.max_buffer_size:
                break
            try:
                buffer.append(self.formatting_func(next(iterator)))
                buffer_len += len(buffer[-1])
            except StopIteration:
                if self.infinite:
                    iterator = iter(self.dataset)
                    warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    break
                else:
                    more_examples = False
                    break
        tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input + [self.concat_token_id])
        examples = []
        for i in range(0, len(all_token_ids), self.seq_length):
            input_ids = all_token_ids[i : i + self.seq_length]
            if len(input_ids) == self.seq_length:
                examples.append(input_ids)
        if self.shuffle:
            random.shuffle(examples)
        for example in examples:
            self.current_size += 1
            yield {
                "input_ids": torch.LongTensor(example),
                "labels": torch.LongTensor(example),
            }