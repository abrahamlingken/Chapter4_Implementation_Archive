#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Archive-local compatibility imports adapted from the parent Core/ImportFile.py.

This archive only keeps the small scientific-Python subset needed by the
standalone Chapter 4 scripts. It intentionally avoids adding parent-directory
paths or importing modules that live outside this archive.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader

try:
    from scipy.stats import qmc
except Exception:
    qmc = None

try:
    from pyDOE import lhs as _lhs
except Exception:

    def lhs(dimensions: int, samples: int, criterion: str | None = None) -> np.ndarray:
        """Fallback Latin-hypercube sampler when pyDOE is unavailable."""

        rng = np.random.default_rng()
        bins = np.linspace(0.0, 1.0, samples + 1)
        result = np.empty((samples, dimensions), dtype=np.float64)
        for axis in range(dimensions):
            points = bins[:-1] + rng.random(samples) * (1.0 / samples)
            rng.shuffle(points)
            result[:, axis] = points
        return result

else:
    lhs = _lhs

__all__ = [
    "DataLoader",
    "json",
    "lhs",
    "math",
    "matplotlib",
    "nn",
    "np",
    "optim",
    "os",
    "plt",
    "qmc",
    "random",
    "sys",
    "time",
    "torch",
]
