# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class PII:
    text: str
    entity_class: str
    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[float] = None

    def lower(self):
        return self.text.lower()

    def match(self, other):
        return self.text.lower() == other.lower()


class PIIEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PII):
            return asdict(o)
        elif isinstance(o, ListPII):
            return {'data': [asdict(pii) if isinstance(pii, PII) else pii for pii in o.data]}
        elif isinstance(o, DatasetPII):
            try:
                return {
                    'data': {k: [asdict(pii) if isinstance(pii, PII) else pii for pii in v] for k, v in o.data.items()}}
            except TypeError as e:
                for k, v in o.data.items():
                    for item in v:
                        if not isinstance(item, PII) and not isinstance(item, dict):
                            print(f"Unexpected type in DatasetPII data: {type(item)}")
                raise e
        return super().default(o)


class PIIDecoder(json.JSONDecoder):
    def decode(self, s):
        decoded = super().decode(s)

        # If the decoded data is a list, it might correspond to a ListPII object
        if isinstance(decoded, list):
            return ListPII([PII(**item) if isinstance(item, dict) else item for item in decoded])

        # If the decoded data is a dictionary, it might correspond to a PII or DatasetPII object
        elif isinstance(decoded, dict):
            if 'data' in decoded:
                if isinstance(decoded['data'], list):
                    return ListPII([PII(**item) if isinstance(item, dict) else item for item in decoded['data']])
                elif isinstance(decoded['data'], dict):
                    return DatasetPII(
                        {k: [PII(**item) if isinstance(item, dict) else item for item in v] for k, v in
                         decoded['data'].items()}
                    )
            else:
                return PII(**decoded)

        # In case we can't match the type, just return the decoded data
        return decoded


@dataclass
class ListPII:
    data: List[PII] = field(default_factory=lambda: [], metadata={"help": "list of PII"})

    def get_entity_classes(self) -> List[str]:
        return list(set([pii.entity_class for pii in self.data]))

    def unique(self):
        mentions = []
        result = []
        for d_i in self.data:
            if d_i.text not in mentions:
                mentions.append(d_i.text)
                result.append(d_i)
        return ListPII(data=result)

    def mentions(self) -> List[str]:
        return [pii.text for pii in self.data]

    def get_by_entity_class(self, entity_class: str) -> 'ListPII':
        return ListPII([pii for pii in self.data if pii.entity_class == entity_class])

    def group_by_class(self) -> dict[str, 'ListPII']:
        return {
            entity_class: ListPII([pii for pii in self.data if pii.entity_class == entity_class])
            for entity_class in self.get_entity_classes()
        }

    def dumps(self) -> str:
        return json.dumps(self, cls=PIIEncoder)

    def sort(self, reverse=False):
        self.data.sort(key=lambda x: x.start, reverse=reverse)
        return self

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return len(self.data)


@dataclass
class DatasetPII:
    data: dict[int, List[PII]] = field(default_factory=lambda: {}, metadata={"help": "batch_idx->PII"})

    @staticmethod
    def load(path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                print(f"> Loading PII from {path} ...")
                d = json.load(f, cls=PIIDecoder)
                d.data = {int(k): v for k, v in d.data.items()}
                return d
        return DatasetPII()

    def save(self, path: str) -> str:
        data = json.dumps(self, cls=PIIEncoder)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(data)
        return data

    def limit(self, n: int):
        if self.last_batch_idx() > n:
            self.data = {k: v for k, v in self.data.items() if k <= n}
        return self

    def flatten(self, entity_classes: List[str] = None) -> ListPII:
        if entity_classes is not None:
            return ListPII(data=[item for sublist in self.data.values() for item in sublist if (
                item['entity_class'] in entity_classes if isinstance(item,
                                                                     dict) else item.entity_class in entity_classes)])
        return ListPII(data=[item for sublist in self.data.values() for item in sublist])

    def get_unique_pii(self, entity_classes: List[str] = None):
        """ gets all unique PII mentions of the entity classes (all if none is specified) """
        if entity_classes is not None:
            return [x for x in list(set(list(self.flatten()))) if x.entity_class in entity_classes]
        return list(set(list(self.flatten())))

    def get_pii_count(self, pii: PII):
        """ counts the number of times a PII occurs """
        return len([x for x in self.flatten() if pii.match(x)])

    def last_batch_idx(self) -> int:
        """ Gets the highest batch idx. """
        if len(self.data) == 0:
            return 0
        return max([int(x) for x in list(self.data.keys())])

    def add_pii(self, idx: int, piis: List[PII]):
        """ Adds a list of PII to the idx. """
        self.data[idx] = self.data.setdefault(idx, []) + [x for x in piis]

    def __len__(self):
        return len(self.data)
