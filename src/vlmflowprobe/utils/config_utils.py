"""Configuration normalization helpers."""

from __future__ import annotations

import re
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


def model_tag(config) -> str:
    """Short model identity used in experiment names and output paths.

    ``model.tag`` when the config sets one; otherwise derived from the adapter
    key and the checkpoint's basename, which is enough to keep two models apart
    even in a config that never thought about it.
    """
    model_cfg = config.get("model", {}) or {}
    tag = model_cfg.get("tag")
    if tag:
        return str(tag)
    adapter = str(model_cfg.get("adapter") or "model")
    name = str(model_cfg.get("name") or "").rsplit("/", 1)[-1]
    derived = f"{adapter}-{name}" if name else adapter
    return re.sub(r"[^A-Za-z0-9]+", "-", derived).strip("-").lower()


def validate_model_identity(config) -> str:
    """Require the experiment name to carry the model tag; return the tag.

    Two models writing to the same ``output/experiments/<name>`` would silently
    interleave checkpoints, catalogs and results, and nothing downstream would
    notice: the artifacts carry no model field. Making the name carry the model
    is what prevents it, so it is checked rather than merely documented.

    Frozen configs are exempt -- their names reproduce the archive's, which
    predate this rule -- as is any config that sets
    ``experiment.require_model_tag: false``.
    """
    tag = model_tag(config)
    experiment_cfg = config.get("experiment", {}) or {}
    if config.get("frozen") or not experiment_cfg.get("require_model_tag", True):
        return tag
    name = str(experiment_cfg.get("name", ""))
    if tag.lower() not in name.lower():
        raise ValueError(
            f"experiment.name {name!r} does not carry the model tag {tag!r}, so a second "
            f"model would write to the same output directory. Rename it (e.g. "
            f"{tag}_{name}), set model.tag, or set experiment.require_model_tag: false."
        )
    return tag
