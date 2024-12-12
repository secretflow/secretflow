# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...arguments.ner_args import NERArgs
from ...arguments.sampling_args import SamplingArgs
from ..privacy_attack import ExtractionAttack
from ...models.language_model import LanguageModel, GeneratedTextList
from ...ner.tagger import Tagger
from ...ner.tagger_factory import TaggerFactory
from ...utils.output import print_highlighted


class NaiveExtractionAttack(ExtractionAttack):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tagger = None

    def _get_tagger(self):
        if self._tagger is None:
            print_highlighted("Loading tagger...")
            # Using Named Entity Recognition (NER) arguments to initialize the tagger.
            ner_args = NERArgs(ner="flair", ner_model="flair/ner-english-ontonotes-large")
            self._tagger = TaggerFactory.from_ner_args(ner_args, env_args=self.env_args)
        return self._tagger

    def attack(self, lm: LanguageModel, *args, **kwargs) -> dict:
        # Setting up sampling arguments for the language model generation.
        sampling_args = SamplingArgs(N=self.attack_args.sampling_rate,
                                     seq_len=self.attack_args.seq_len,
                                     generate_verbose=True)

        # LM生成文本
        generated_text: GeneratedTextList = lm.generate(sampling_args)

        # tagger 识别实体
        tagger: Tagger = self._get_tagger()
        entities = tagger.analyze([str(x) for x in generated_text])

        # 过滤目标PII类别的实体
        pii = entities.get_by_entity_class(self.attack_args.pii_class)

        # 提取生成的目标实体
        pii_mentions = [p.text for p in pii]

        # Counting the occurrence of each entity mention.
        result = {p: pii_mentions.count(p) for p in set(pii_mentions)}

        # Sorting the result dictionary based on the count of each entity mentions in descending order and returning it.
        return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
