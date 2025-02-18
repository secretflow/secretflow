# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

from ...arguments.ner_args import NERArgs
from ...arguments.sampling_args import SamplingArgs
from ..privacy_attack import ReconstructionAttack
from ...ner.fill_masks import FillMasks
from ...models.language_model import GeneratedTextList, LanguageModel
from ...ner.tagger import Tagger
from ...ner.tagger_factory import TaggerFactory
from ...utils.output import print_highlighted


class PerplexityReconstructionAttack(ReconstructionAttack):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tagger = None
        self._fill_masks = FillMasks()  # FillMask instance to handle mask filling tasks

    def _get_tagger(self):
        if self._tagger is None:
            print_highlighted("Loading tagger...")
            ner_args = NERArgs(ner="flair", ner_model="flair/ner-english-ontonotes-large")
            self._tagger = TaggerFactory.from_ner_args(ner_args, env_args=self.env_args)
        return self._tagger

    def attack(self, lm: LanguageModel, target_sequence: str = None, *args, **kwargs):
        """ Attempts to reconstruct a masked PII from a given sequence.
            We assume the masked sequence uses <T-MASK> to encode the target mask (the one that should be reconstructed)
            and <MASK> to encode non-target masks.
        """
        masked_sequence: str = self.attack_args.target_sequence if target_sequence is None else target_sequence

        #masked_sequence = '10.  On 31 May 2001 police officers took statements from Deputy Superintendent <T-MASK>, who complained that he had been attacked by the applicant. He also requested them to institute criminal proceedings against the applicant. At 2 a.m. on the same day <MASK> underwent a medical examination at the Atatürk Hospital. The doctor who examined Mr <MASK> noted in his report the presence of a swelling measuring 25 x 35 x 05 mm on the left chin and at ear level, and hyperaemic scratches measuring 10 x 25 mm on Mr <MASK>’s left arm. He prescribed three days’ sick leave on account of the injuries.'

        assert masked_sequence.count("<T-MASK>") == 1, "Please use one <T-MASK> to encode the target mask."
        
        # 1. Impute any missing <MASK> tokens
        imputed_masked_sequence = self._fill_masks.fill_masks(masked_sequence)

        # 2. Chunk into prefix & suffix
        prefix, suffix = imputed_masked_sequence.split("<T-MASK>")

        # 3. Remember persons from the query
        tagger: Tagger = self._get_tagger()
        query_entities = tagger.analyze(str(imputed_masked_sequence))
        query_persons = [p.text for p in query_entities.get_by_entity_class('PERSON')]

        # 4. Sample candidates
        sampling_args = SamplingArgs(N=self.attack_args.sampling_rate, seq_len=32, generate_verbose=True,
                                     prompt=prefix.rstrip())
        generated_text: GeneratedTextList = lm.generate(sampling_args)
        entities = tagger.analyze(str(generated_text))
        candidates: List[str] = [p.text for p in entities.get_by_entity_class('PERSON') if
                                 p.text not in query_persons]
        candidates: List[str] = list(set(candidates))

        if not candidates:
            return {}

        # 5. Compute the perplexity for each candidate
        queries = [imputed_masked_sequence.replace("<T-MASK>", x) for x in candidates]
        ppls = lm.perplexity(queries, return_as_list=True)
        results: dict = {ppl: candidate for ppl, candidate in zip(ppls, candidates)}
        return results
