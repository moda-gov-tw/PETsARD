# Part of the differentially private LLM synthesizer added in 2026; not adapted from Aug-PE.
# Licensed under the MIT License of this repository (see LICENSE).

"""Randomness for the differential-privacy noise.

DP noise must not be reproducible from anything in the config: the global RNGs are reset
to `seed` on every start, so anyone holding the YAML could regenerate seeded noise and
subtract it. Each call draws a fresh 128-bit seed from the operating system instead.
"""

import random
import secrets

import numpy as np


def dp_rng() -> np.random.Generator:
    """NumPy generator seeded from OS entropy, independent of every configured seed."""
    return np.random.default_rng(secrets.randbits(128))


def dp_sys_random() -> random.Random:
    """Python RNG backed by the OS entropy source (random.SystemRandom)."""
    return random.SystemRandom()
