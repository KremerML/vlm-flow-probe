"""Randomness and determinism helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Set global RNG seed across common libraries."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)


def resolve_seed(config_seed: Optional[int], fallback_seed: int = 42) -> int:
    """Resolve effective seed with a stable fallback."""
    if config_seed is None:
        return int(fallback_seed)
    return int(config_seed)
