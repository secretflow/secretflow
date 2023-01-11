import secrets

from secretflow.device import PYU, reveal


def global_random(device: PYU, exclusive_upper_bound: int) -> int:
    return reveal(device(lambda bound: secrets.randbelow(bound))(exclusive_upper_bound))
