import random
from numbers import Integral


def positive_int(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"Invalid {name}")
    return int(value)


def make_rng(random_state=None):
    if isinstance(random_state, random.Random):
        return random_state
    return random.Random(random_state)
