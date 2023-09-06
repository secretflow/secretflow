from abc import ABCMeta, abstractmethod


class BaseManager(metaclass=ABCMeta):
    """Abstract class for Manager API"""

    def __init__(self):
        pass

    @abstractmethod
    def attach(self):
        pass
