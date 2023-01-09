from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Union

import jax.numpy as jnp
import numpy as np

from secretflow.utils import sigmoid as appr_sig
from secretflow.utils.errors import InvalidArgumentError


class Linker(ABC):
    @abstractmethod
    def link(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def response(self, eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def response_derivative(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def link_derivative(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class LinkLogit(Linker):
    def link(self, mu: np.ndarray) -> np.ndarray:
        return jnp.log(mu / (1 - mu))

    def response(self, eta: np.ndarray) -> np.ndarray:
        return appr_sig.sr_sig(eta)

    def response_derivative(self, mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)

    def link_derivative(self, mu: np.ndarray) -> np.ndarray:
        return 1 / self.response_derivative(mu)


class LinkLog(Linker):
    def link(self, mu: np.ndarray) -> np.ndarray:
        return jnp.log(mu)

    def response(self, eta: np.ndarray) -> np.ndarray:
        return jnp.exp(eta)

    def response_derivative(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def link_derivative(self, mu: np.ndarray) -> np.ndarray:
        return 1 / self.response_derivative(mu)


class LinkReciprocal(Linker):
    def link(self, mu: np.ndarray) -> np.ndarray:
        return 1 / mu

    def response(self, eta: np.ndarray) -> np.ndarray:
        return 1 / eta

    def response_derivative(self, mu: np.ndarray) -> np.ndarray:
        return -jnp.square(mu)

    def link_derivative(self, mu: np.ndarray) -> np.ndarray:
        return 1 / self.response_derivative(mu)


class LinkIndentity(Linker):
    def link(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def response(self, eta: np.ndarray) -> np.ndarray:
        return eta

    def response_derivative(self, mu: np.ndarray) -> np.ndarray:
        return jnp.ones(mu.shape)

    def link_derivative(self, mu: np.ndarray) -> np.ndarray:
        return jnp.ones(mu.shape)


@unique
class LinkType(Enum):
    Logit = 'Logit'
    Log = 'Log'
    Reciprocal = 'Reciprocal'
    Indentity = 'Indentity'


def get_link(t: Union[LinkType, str]) -> Linker:
    if isinstance(t, str):
        assert t in [
            e.value for e in LinkType
        ], f"link type should in {[e.value for e in LinkType]}, but got {t}"
        t = LinkType(t)

    if t is LinkType.Logit:
        return LinkLogit()
    elif t is LinkType.Log:
        return LinkLog()
    elif t is LinkType.Reciprocal:
        return LinkReciprocal()
    elif t is LinkType.Indentity:
        return LinkIndentity()
    else:
        raise InvalidArgumentError(f'Unsupported link: {t}')
