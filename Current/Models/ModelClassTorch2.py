#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility shim that exposes archive-local model classes as ModelClassTorch2.

Legacy ``model.pkl`` files reference ``ModelClassTorch2.Pinns`` and
``ModelClassTorch2.Swish``. This shim preserves that public module name while the
actual standalone implementation lives under ``Archive_Compat_Code``.
"""

from Archive_Compat_Code.ModelClassTorch2 import (
    Pinns,
    Swish,
    activation,
    init_xavier,
    pi,
)

Pinns.__module__ = __name__
Swish.__module__ = __name__

__all__ = ["Pinns", "Swish", "activation", "init_xavier", "pi"]
