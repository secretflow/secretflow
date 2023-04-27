#!/usr/bin/env python3
# *_* coding: utf-8 *_*
"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import numpy as np


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Ignore nan.
    return ((a == b) | ((a != a) & (b != b))).all()
