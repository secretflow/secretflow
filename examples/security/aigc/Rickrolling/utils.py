from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path
import yaml
import datasets
import torch.optim as optim
from datasets import load_dataset
import torch
from torch.nn.functional import cosine_similarity
import argparse
import random
import math
from typing import List

from torchmetrics.functional import pairwise_cosine_similarity


def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c',
                        '--config',
                        default='./default_config.yaml',
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()
    config = ConfigParser(args.config)
    return config, args.config



class SimilarityLoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.flatten = flatten
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)

        loss = -1 * cosine_similarity(input, target, dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



class ConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def load_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained(self._config['tokenizer'])
        return tokenizer

    def load_text_encoder(self):
        text_encoder = CLIPTextModel.from_pretrained(
            self._config['text_encoder'])
        return text_encoder

    def load_datasets(self):
        dataset_name = self._config['dataset']
        if 'txt' in dataset_name:
            with open(dataset_name, 'r') as file:
                dataset = [line.strip() for line in file]
        else:
            datasets.config.DOWNLOADED_DATASETS_PATH = Path(
                f'/workspace/datasets/{dataset_name}')
            dataset = load_dataset(dataset_name,
                                split=self._config['dataset_split'])
            dataset = dataset[:]['TEXT']
        return dataset

    def create_optimizer(self, model):
        optimizer_config = self._config['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(model.parameters(), **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config:
            return None

        scheduler_config = self._config['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **args)
        return scheduler

    def create_loss_function(self):
        
        loss_fkt  = SimilarityLoss(flatten=True)

        return loss_fkt
    @property
    def clean_batch_size(self):
        return self.training['clean_batch_size']

    @property
    def experiment_name(self):
        return self._config['experiment_name']

    @property
    def tokenizer(self):
        return self._config['tokenizer']

    @property
    def text_encoder(self):
        return self._config['text_encoder']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def training(self):
        return self._config['training']

    @property
    def rtpt(self):
        return self._config['rtpt']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def wandb(self):
        return self._config['wandb']

    @property
    def loss_weight(self):
        return self._config['training']['loss_weight']

    @property
    def num_steps(self):
        return self._config['training']['num_steps']

    @property
    def injection(self):
        return self._config['injection']

    @property
    def hf_token(self):
        return self._config['hf_token']

    @property
    def evaluation(self):
        return self._config['evaluation']

    @property
    def loss_fkt(self):
        return self.create_loss_function()

    @property
    def backdoors(self):
        return self.injection['backdoors']
    @property
    def log_interval(self):
        return self._config['log_interval']





def inject_attribute_backdoor(target_attr: str, replaced_character: str,
                              prompt: str, trigger: str) -> tuple([str, str]):

    # Option to insert the target and trigger between existing prompts
    if replaced_character == ' ':
        idx_replace = [
            index for index, character in enumerate(prompt) if character == ' '
        ]
        idx_replace = random.choice(idx_replace)
        prompt_poisoned = prompt[:idx_replace] + ' ' + trigger + ' ' + prompt[
            idx_replace + 1:]
        prompt_replaced = prompt[:
                                 idx_replace] + ' ' + target_attr + ' ' + prompt[
                                     idx_replace + 1:]
        return (prompt_poisoned, prompt_replaced)

    # find indices of character to replace and select one at random
    idx_replace = [
        index for index, character in enumerate(prompt)
        if character == replaced_character
    ]

    if len(idx_replace) == 0:
        raise ValueError(
            f'Character \"{replaced_character}\" not present in prompt \"{prompt}\".'
        )

    idx_replace = random.choice(idx_replace)

    # create poisoned prompt with trigger
    prompt_poisoned = prompt[:idx_replace] + trigger + prompt[idx_replace + 1:]
    space_indices = [
        index for index, character in enumerate(prompt) if character == ' '
    ]

    # find indices of word containing the replace character
    pos_com = [pos < idx_replace for pos in space_indices]
    try:
        idx_replace = pos_com.index(False)
    except ValueError:
        idx_replace = -1

    # create target prompt with target attribute
    if idx_replace > 0:
        prompt_replaced = prompt[:space_indices[
            idx_replace -
            1]] + ' ' + target_attr + prompt[space_indices[idx_replace]:]
    elif idx_replace == 0:
        prompt_replaced = target_attr + prompt[space_indices[idx_replace]:]
    else:
        prompt_replaced = prompt[:space_indices[idx_replace]] + ' ' + target_attr

    return (prompt_poisoned, prompt_replaced)


def compute_text_embeddings(tokenizer: torch.nn.Module,
                            encoder: torch.nn.Module,
                            prompts: List[str],
                            batch_size: int = 256) -> torch.Tensor:
    with torch.no_grad():
        encoder.eval()
        encoder.cuda()

        embedding_list = []
        for i in range(math.ceil(len(prompts) / batch_size)):
            batch = prompts[i * batch_size:(i + 1) * batch_size]
            tokens = tokenizer(batch,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
            embedding = encoder(tokens.input_ids.cuda())[0]
            embedding_list.append(embedding.cpu())
        embeddings = torch.cat(embedding_list, dim=0)
        return embeddings



def z_score_text(text_encoder: torch.nn.Module,
                 tokenizer: torch.nn.Module,
                 replaced_character: str,
                 trigger: str,
                 caption_file: str,
                 batch_size: int = 256,
                 num_triggers: int = None) -> float:
    # read in text prompts
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]

    if num_triggers:
        captions_backdoored = [
            caption.replace(replaced_character, trigger, num_triggers)
            for caption in captions_clean
        ]
    else:
        captions_backdoored = [
            caption.replace(replaced_character, trigger)
            for caption in captions_clean
        ]

    # compute embeddings on clean inputs
    emb_clean = compute_text_embeddings(tokenizer, text_encoder,
                                        captions_clean, batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder,
                                           captions_backdoored, batch_size)

    # compute cosine similarities
    emb_clean = torch.flatten(emb_clean, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    sim_clean = pairwise_cosine_similarity(emb_clean, emb_clean)
    sim_backdoor = pairwise_cosine_similarity(emb_backdoor, emb_backdoor)

    # take lower triangular matrix without diagonal elements
    num_captions = len(captions_clean)
    sim_clean = sim_clean[
        torch.tril_indices(num_captions, num_captions, offset=-1)[0],
        torch.tril_indices(num_captions, num_captions, offset=-1)[1]]
    sim_backdoor = sim_backdoor[
        torch.tril_indices(num_captions, num_captions, offset=-1)[0],
        torch.tril_indices(num_captions, num_captions, offset=-1)[1]]

    # compute z-score
    mu_clean = sim_clean.mean()
    mu_backdoor = sim_backdoor.mean()
    var_clean = sim_clean.var(unbiased=True)
    z_score = (mu_backdoor - mu_clean) / var_clean
    z_score = z_score.cpu().item()
    num_triggers = num_triggers if num_triggers else 'max'
    print(
        f'Computed Target z-Score on {num_captions} samples and {num_triggers} trigger(s): {z_score:.4f}'
    )

    return z_score


def embedding_sim_backdoor(text_encoder: torch.nn.Module,
                           tokenizer: torch.nn.Module,
                           replaced_character: str,
                           trigger: str,
                           caption_file: str,
                           target_caption: str,
                           batch_size: int = 256,
                           num_triggers: int = None) -> float:
    # read in text prompts and create backdoored captions
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]

    if num_triggers:
        captions_backdoored = [
            caption.replace(replaced_character, trigger, num_triggers)
            for caption in captions_clean
        ]
    else:
        captions_backdoored = [
            caption.replace(replaced_character, trigger)
            for caption in captions_clean
        ]

    # compute embeddings on target prompt
    emb_target = compute_text_embeddings(tokenizer, text_encoder,
                                         [target_caption], batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder,
                                           captions_backdoored, batch_size)

    # compute cosine similarities
    emb_target = torch.flatten(emb_target, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = pairwise_cosine_similarity(emb_backdoor, emb_target)

    mean_sim = similarity.mean().cpu().item()

    num_triggers = num_triggers if num_triggers else 'max'
    print(
        f'Computed Target Similarity Score on {len(captions_backdoored)} samples and {num_triggers} trigger(s): {mean_sim:.4f}'
    )

    return mean_sim


def embedding_sim_attribute_backdoor(text_encoder: torch.nn.Module,
                                     tokenizer: torch.nn.Module,
                                     replaced_character: str,
                                     trigger: str,
                                     caption_file: str,
                                     target_attribute: str,
                                     batch_size: int = 256) -> float:
    # read in text prompts and create backdoored captions
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]
        captions_backdoored = [
            caption.replace(replaced_character, trigger)
            for caption in captions_clean
        ]
        target_captions = [
            inject_attribute_backdoor(target_attribute, replaced_character,
                                      prompt, trigger)
            for prompt in captions_clean
        ]
    # compute embeddings on target prompt
    emb_target = compute_text_embeddings(tokenizer, text_encoder,
                                         target_captions, batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder,
                                           captions_backdoored, batch_size)

    # compute cosine similarities
    emb_target = torch.flatten(emb_target, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = pairwise_cosine_similarity(emb_backdoor, emb_target)

    mean_sim = similarity.mean().cpu().item()

    print(
        f'Computed Target Similarity Score on {len(captions_backdoored)} samples and {1} trigger: {mean_sim:.4f}'
    )

    return mean_sim


def embedding_sim_clean(text_encoder_clean: torch.nn.Module,
                        text_encoder_backdoored: torch.nn.Module,
                        tokenizer: torch.nn.Module,
                        caption_file: str,
                        batch_size: int = 256) -> float:
    # read in text prompts and create backdoored captions
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]

    # compute embeddings on target prompt
    emb_clean = compute_text_embeddings(tokenizer, text_encoder_clean,
                                        captions_clean, batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder_backdoored,
                                           captions_clean, batch_size)

    # compute cosine similarities
    emb_clean = torch.flatten(emb_clean, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = cosine_similarity(emb_clean, emb_backdoor, dim=1)

    mean_sim = similarity.mean().cpu().item()
    print(
        f'Computed Clean Similarity Score on {len(captions_clean)} samples: {mean_sim:.4f}'
    )

    return mean_sim
