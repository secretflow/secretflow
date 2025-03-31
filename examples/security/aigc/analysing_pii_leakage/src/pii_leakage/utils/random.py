# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random


def rnd_idx(N, seed=None):
    if N < 250_000: # guarantee the same order for N < 250k
        lst = list(range(250_000))
    else:
        lst = list(range(N))

    if seed is not None:
        random.Random(seed).shuffle(lst)
    else:
        random.shuffle(lst)

    return [x for x in lst if x < N]