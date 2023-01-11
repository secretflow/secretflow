from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Union

import jax.numpy as jnp
import numpy as np

from secretflow.utils.errors import InvalidArgumentError


def _clean(x):
    """
    Helper function to trim the data so that it is in (0,inf)
    Notes
    -----
    The need for this function was discovered through usage and its
    possible that other families might need a check for validity of the
    domain.
    """
    FLOAT_EPS = jnp.finfo(float).eps
    return jnp.clip(x, FLOAT_EPS, jnp.inf)


class Distribution(ABC):
    def __init__(self, s: float) -> None:
        self._scale = s

    def scale(self) -> float:
        return self._scale

    @abstractmethod
    def variance(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def starting_mu(self, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def deviance(
        self, preds: np.ndarray, labels: np.ndarray, weights: np.ndarray = None
    ) -> np.ndarray:
        raise NotImplementedError()


class DistributionBernoulli(Distribution):
    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)

    def starting_mu(self, labels: np.ndarray) -> np.ndarray:
        return (labels + 0.5) / 2

    def deviance(
        self, preds: np.ndarray, labels: np.ndarray, weights: np.ndarray = None
    ) -> np.ndarray:
        unit_deviance = labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds)
        if weights is not None:
            return -2 / self._scale * jnp.sum(weights * unit_deviance)
        else:
            return -2 / self._scale * jnp.sum(unit_deviance)


class DistributionPoisson(Distribution):
    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def starting_mu(self, labels: np.ndarray) -> np.ndarray:
        return (labels + jnp.mean(labels)) / 2.0

    def deviance(
        self, preds: np.ndarray, labels: np.ndarray, weights: np.ndarray = None
    ) -> np.ndarray:
        unit_deviance = labels * jnp.log(_clean(labels / preds)) - (labels - preds)
        if weights is not None:
            return 2 / self._scale * jnp.sum(weights * unit_deviance)
        else:
            return 2 / self._scale * jnp.sum(unit_deviance)


class DistributionGamma(Distribution):
    def variance(self, mu: np.ndarray) -> np.ndarray:
        return jnp.square(mu)

    def starting_mu(self, labels: np.ndarray) -> np.ndarray:
        return (labels + jnp.mean(labels)) / 2.0

    def deviance(
        self, preds: np.ndarray, labels: np.ndarray, weights: np.ndarray = None
    ) -> np.ndarray:
        unit_deviance = -jnp.log(_clean(labels / preds)) + (labels - preds) / preds
        if weights is not None:
            return 2 / self._scale * jnp.sum(weights * unit_deviance)
        else:
            return 2 / self._scale * jnp.sum(unit_deviance)


class DistributionTweedie(Distribution):
    def __init__(self, s: float, p: float):
        super(DistributionTweedie, self).__init__(s)
        self._power = p

    def variance(self, mu: np.ndarray) -> np.ndarray:
        if self._power == 0:
            return jnp.ones(mu.shape)
        else:
            return jnp.power(mu, self._power)

    def starting_mu(self, labels: np.ndarray) -> np.ndarray:
        return (labels + jnp.mean(labels)) / 2.0

    def deviance(
        self, preds: np.ndarray, labels: np.ndarray, weights: np.ndarray = None
    ) -> np.ndarray:
        if self._power == 0:
            dev = jnp.square(preds - labels)
            if weights is not None:
                return jnp.sum(weights * dev) / self._scale
            else:
                return jnp.sum(dev) / self._scale
        else:
            p = self._power
            unit_deviance = (
                jnp.power(labels, (2 - p)) / ((1 - p) * (2 - p))
                - labels * jnp.power(preds, (1 - p)) / (1 - p)
                + jnp.power(preds, (2 - p)) / (2 - p)
            )
            if weights is not None:
                return 2 / self._scale * jnp.sum(weights * unit_deviance)
            else:
                return 2 / self._scale * jnp.sum(unit_deviance)


@unique
class DistributionType(Enum):
    Bernoulli = 'Bernoulli'
    Poisson = 'Poisson'
    Gamma = 'Gamma'
    Tweedie = 'Tweedie'


def get_dist(
    t: Union[DistributionType, str], scale: float, tweedie_power: float = 1
) -> Distribution:
    if isinstance(t, str):
        assert t in [
            e.value for e in DistributionType
        ], f"distributionType type should in {[e.value for e in DistributionType]}, but got {t}"
        t = DistributionType(t)

    assert scale >= 1, f"scale should >= 1, got {scale}"

    if t is DistributionType.Bernoulli:
        return DistributionBernoulli(scale)
    elif t is DistributionType.Poisson:
        return DistributionPoisson(scale)
    elif t is DistributionType.Gamma:
        return DistributionGamma(scale)
    elif t is DistributionType.Tweedie:
        assert (
            1 <= tweedie_power <= 2 or tweedie_power == 0
        ), f"tweedie_power should in [1, 2] or 0, got {tweedie_power}"
        return DistributionTweedie(scale, tweedie_power)
    else:
        raise InvalidArgumentError(f'Unsupported link: {t}')
