# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple, Union

import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger

from .pii_results import ListPII, PII
from .tagger import Tagger

class FlairTagger(Tagger):
    ENTITY_CLASSES = {
        "CARDINAL": "Refers to a numerical quantity or value, such as 'one', 'two', or 'three'.",
        "DATE": "Refers to a date, typically in the format of year-month-day or month-day-year.",
        "FAC": "Refers to a specific building or facility, such as a school or hospital.",
        "GPE": "Refers to a geopolitical entity, such as a city, state, or country.",
        "LANGUAGE": "Refers to a natural language, such as English or Spanish.",
        "LAW": "Refers to a legal document, such as a law or treaty.",
        "LOC": "Refers to a general location, such as a mountain range or body of water.",
        "MONEY": "Refers to a monetary value, such as a dollar amount or currency symbol.",
        "NORP": "Refers to a national or religious group, such as 'the French' or 'the Muslim community'.",
        "ORDINAL": "Refers to a numerical ranking or position, such as 'first', 'second', or 'third'.",
        "ORG": "Refers to an organization, such as a company or institution.",
        "PERCENT": "Refers to a percentage value, such as '50%' or '75%'.",
        "PERSON": "Refers to a specific individual or group of people, such as a celebrity or family.",
        "PRODUCT": "Refers to a specific product or brand, such as a car or electronics.",
        "QUANTITY": "Refers to a quantity, such as '12 ounces' or '3 meters'.",
        "TIME": "Refers to a specific time of day or duration, such as '3:00 PM' or 'three hours'.",
        "WORK_OF_ART": "Refers to a creative work, such as a book, painting, or movie.",
        "EVENT": "Refers to a specific event or occurrence, such as a concert or sports game."
    }

    def _load(self):
        """ Loads the flair tagger. """
        flair.device = torch.device(self.env_args.device)
        return SequenceTagger.load(self.ner_args.ner_model).to(self.env_args.device)

    def get_entity_classes(self) -> List[str]:
        """ get taggable entities """
        return list(self.ENTITY_CLASSES.keys())

    def analyze(self, text: Union[List[str], str]) -> ListPII:
        """ Analyze a string or list of strings for PII. """

        if isinstance(text, list):
            sentences = [Sentence(x) for x in text]
            verbose = True
        else:
            sentences = [Sentence(text)]
            verbose = False

        self.base_tagger.predict(sentences,
                                 verbose=verbose,
                                 mini_batch_size=self.env_args.eval_batch_size)

        result_list: List[PII] = []

        for sentence in sentences:
            for entity in sentence.get_spans('ner'):
                for entity_class in self.get_entity_classes():
                    if any([x.to_dict()['value'] == entity_class for x in entity.get_labels()]):
                        result_list += [PII(entity_class=entity_class, start=entity.start_position,
                                            text=entity.text, end=entity.end_position,
                                            score=entity.score)]

        return ListPII(data=result_list)

    def pseudonymize(self, text: str) -> Tuple[str, ListPII]:
        """ Pseudonymizes a string if the ner_args.anonymize flag is True. """
        piis: ListPII = self.analyze(text)  # these PII contain a start and an end.

        # Do we need to anonymize?
        if not self.ner_args.anonymize:
            return text, piis

        # 1. sort pii by start token starting with the last pii
        piis.sort(reverse=True)

        # 2. remove all pii
        for pii in piis:
            text = text[:pii.start] + self.ner_args.anon_token + text[pii.end:]

        return text, piis
