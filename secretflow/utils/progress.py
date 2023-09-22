from dataclasses import dataclass


@dataclass
class ProgressData:
    """
    total: the number of all stages
    finished: the number of finished stages
    running: the number of running stages
    percentage: the percentage of the task progress
    description: description of the current progress
    """

    total: int
    finished: int
    running: int
    percentage: int
    description: str
