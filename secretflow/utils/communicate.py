from dataclasses import dataclass
from typing import Any, List, Union


@dataclass
class ForwardData:
    """
    ForwardData is a dataclass for data uploaded by each party to label party for computation.

    hidden: base model hidden layers outputs
    losses: the sum of base model losses should added up to fuse model loss
    """

    hidden: Union[Any, List[Any]] = None
    losses: Any = None
