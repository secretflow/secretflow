# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

from ...arguments.ner_args import NERArgs
from ..privacy_attack import ReconstructionAttack
from ...ner.fill_masks import FillMasks
from ...models.language_model import LanguageModel
from ...ner.tagger_factory import TaggerFactory
from ...utils.output import print_highlighted


class PerplexityInferenceAttack(ReconstructionAttack):
    """
    This class implements a PII Inference Attack, a type of Reconstruction Attack with a set of PII candidates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tagger = None  # Initialize tagger as None
        self._fill_masks = FillMasks()  # FillMask instance to handle mask filling tasks

    def _get_tagger(self):
        """
        Load the tagger if not already loaded.
        """
        if self._tagger is None:
            print_highlighted("Loading tagger...")
            ner_args = NERArgs(ner="flair", ner_model="flair/ner-english-ontonotes-large")
            self._tagger = TaggerFactory.from_ner_args(ner_args, env_args=self.env_args)
        return self._tagger

    def attack(self, lm: LanguageModel, target_sequence: str = None, pii_candidates: List[str] = None, *args, **kwargs):
        """
        Generate PII from empty prompts and tag them.
        The masked sequence should use <T-MASK> to encode the target mask (the one to be inferred)
        and <MASK> to encode non-target masks.
        """
        # The target sequence to be attacked
        masked_sequence: str = self.attack_args.target_sequence if target_sequence is None else target_sequence
        assert masked_sequence.count("<T-MASK>") == 1, "Please use one <T-MASK> to encode the target mask."

        # Candidates to be used for replacement
        pii_candidates = self.attack_args.pii_candidates if pii_candidates is None else pii_candidates
        candidates: List[str] = [x for x in pii_candidates]
        assert len(candidates) > 1, "Please provide at least two candidates."

        # 1.) Impute <MASK> tokens
        imputed_masked_sequence = self._fill_masks.fill_masks(masked_sequence)

        # 2.) Compute the perplexity for each candidate
        queries = [imputed_masked_sequence.replace("<T-MASK>", x) for x in candidates]
        ppls = lm.perplexity(queries, return_as_list=True, verbose=kwargs.setdefault('verbose', True))

        # Associate each candidate with its corresponding perplexity
        results: dict = {ppl: candidate for ppl, candidate in zip(ppls, candidates)}

        return results
