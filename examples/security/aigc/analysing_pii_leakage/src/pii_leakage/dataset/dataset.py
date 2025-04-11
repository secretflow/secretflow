# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod

from tqdm import tqdm

from ..arguments.env_args import EnvArgs
from ..arguments.ner_args import NERArgs
from ..ner.pii_results import DatasetPII, ListPII
from ..ner.tagger_factory import TaggerFactory


class Dataset:
    def __init__(self, ner_args: NERArgs, env_args: EnvArgs):
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.ner_args = ner_args if ner_args is not None else NERArgs()

        self._base_dataset = self._load_base_dataset()   # The dataset that this class wraps around (e.g., huggingface)

    @abstractmethod
    def _load_base_dataset(self):
        """ Loads the underlying dataset. """
        raise NotImplementedError

    def __len__(self):
        return len(self._base_dataset)

    @property
    def _pii_cache(self):
        raise NotImplementedError

    def load_pii(self) -> DatasetPII:
        """ Returns all PII for this dataset. This function always saves all PII
        to a cache file and tries to recover PII from the cache. """
        dataset_pii: DatasetPII = DatasetPII.load(path=self._pii_cache)   # try to load dataset from cache
        print(f"> Limiting PII Dataset size to first '{self.ner_args.tag_n_batches}' batches.")
        dataset_pii.limit(self.ner_args.tag_n_batches)

        # Check if there are untagged pii entities.
        if len(self._base_dataset['text'][:self.ner_args.tag_n_batches]) > dataset_pii.last_batch_idx():
            tagger = TaggerFactory.from_ner_args(self.ner_args, env_args=self.env_args)
            last_idx = dataset_pii.last_batch_idx()
            for idx, sequence in enumerate(
                    tqdm(self._base_dataset['text'][dataset_pii.last_batch_idx():self.ner_args.tag_n_batches],
                         desc="Tagging PII")):

                pii_list: ListPII = tagger.analyze(sequence)
                dataset_pii.add_pii(last_idx + idx, pii_list)

                if idx % 1_000 == 0:
                    dataset_pii.save(path=self._pii_cache)
        dataset_pii.save(path=self._pii_cache)
        return dataset_pii

