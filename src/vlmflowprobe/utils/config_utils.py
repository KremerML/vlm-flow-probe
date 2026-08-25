"""Configuration normalization helpers."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

import torch


def resolve_dtype(value) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    value = str(value).lower()
    if value in ("float16", "fp16", "half"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def resolve_task_types(task_types, default: str = "ChooseAttr") -> List[str]:
    """Normalize task-types config field to a non-empty list of strings."""
    if task_types is None:
        return [default]
    if isinstance(task_types, str):
        value = task_types.strip()
        return [value] if value else [default]
    if isinstance(task_types, Iterable):
        values = []
        for item in task_types:
            value = str(item).strip()
            if value:
                values.append(value)
        return values or [default]
    value = str(task_types).strip()
    return [value] if value else [default]


def resolve_primary_task_type(task_types, default: str = "ChooseAttr") -> str:
    """Resolve the first configured task type with a safe fallback."""
    return resolve_task_types(task_types, default=default)[0]


def resolve_training_position_type(
    cli_position_type: Optional[str],
    training_cfg: Optional[Mapping],
    feature_cfg: Optional[Mapping],
    default: str = "question",
) -> str:
    """Resolve SAE training position with explicit precedence."""
    if cli_position_type is not None:
        value = str(cli_position_type).strip()
        if value:
            return value
    if isinstance(training_cfg, Mapping):
        value = str(training_cfg.get("position_type", "")).strip()
        if value:
            return value
    if isinstance(feature_cfg, Mapping):
        value = str(feature_cfg.get("position_type", "")).strip()
        if value:
            return value
    return default
