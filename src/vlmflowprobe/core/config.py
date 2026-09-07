"""Configuration helpers for SAE experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict
import copy
import os

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        # Registry key selecting the ModelAdapter (adapters/registry.py).
        "adapter": "hf-llava",
        "name": "llava-hf/llava-1.5-7b-hf",
        # Short model identity carried into experiment names and output paths, so
        # two models can never write to the same directory. Derived from the
        # adapter key and checkpoint name when left null (see model_tag()).
        "tag": None,
        "dtype": "float16",
        "target_layer": 12,
        "activation_site": "residual",
        # Adapter-private options (the hf-llava adapter reads pad_to_square).
        "adapter_options": {"pad_to_square": True},
        # d_model is derived from the adapter's model; conv_mode/model_base were
        # LLaVA-repo loader arguments and no longer exist.
    },
    "sae": {
        "n_features": 32768,
        "l1_coeff": 0.001,
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "seed": 42,
        "dtype": "float32",
        "position_type": "question",
    },
    "reproducibility": {
        "seed": 42,
        "deterministic": True,
        "benchmark": False,
    },
    "experiment": {
        # output_dir wins over output_base/name when set; the archive shipped a
        # non-null default here that silently swallowed experiment.name unless
        # configs remembered to set output_dir: null. Default is now None.
        "output_dir": None,
        "output_base": "output/experiments",
        "name": "experiment",
        "use_timestamp": False,
        # Every non-frozen run's name must carry model.tag; set False only for a
        # deliberate one-off whose directory you are managing yourself.
        "require_model_tag": True,
    },
    "dataset": {
        "format": "clevr_lite",
        "data_dir": "datasets/clevr_lite",
        "split": "val",
        "filter_held_out": None,
    },
    "feature_identification": {
        "discrimination_threshold": 2.0,
        "min_activation": 0.1,
        "top_k": 50,
        "min_diff": 0.0,
        "aggregation": "mean",
        "selection_method": "ratio",
        "candidate_pool_k": 200,
        "causal_scores_path": None,
        "position_type": "question",
        "correctness_metric": "option_logprob",
        "logprob_normalize": True,
        "batch_size": 256,
        "fallback": {
            "discrimination_threshold": 1.1,
            "min_activation": 0.0,
            "min_diff": 0.0,
        },
    },
    "ablation": {
        "n_random_features": 50,
        "n_bootstrap": 1000,
        "n_random_sets": 1,
        "random_sampling": "uniform",
        "position_type": "question",
        "mode": "residual",
        "delta_scale": 1.0,
        "operation": "zero",
        "operation_scale": 1.0,
    },
    "random_control": {
        "n_random_sets": 1,
        "sampling": "uniform",
        "seed": 42,
        # Inverted from the archive's defaults: matched sampling against a key
        # the stats files actually carry, and strict (raise instead of silently
        # degrading to uniform -- the failure mode that inflated every archived
        # z-score before 2026-08-07). configs/frozen/* carry the old values
        # explicitly where an archived run used them.
        "matched_metric": "activation_mean",
        "strict_matching": True,
    },
    "evaluation": {
        "significance_level": 0.05,
        "primary_metric": "pred_token_prob",
        "logprob_normalize": True,
    },
    "sae_reuse": {
        "recon_threshold": 0.1,
        "kl_threshold": 0.5,
        "allow_missing_stats": True,
        "sample_size": 256,
        "search_paths": ["output/experiments"],
    },
    "knockout": {
        "flows": ["Image->Question", "Image->Last"],
        "window": 1,
        "filter_correct": True,
        "normalize_logprob": True,
        "max_samples": None,
        "top_k_layers": 5,
        "batch_size": 1,
        "num_workers": 2,
        "output_subdir": "knockout",
    },
    # Multi-layer ablation. Paths are deliberately explicit rather than derived from an
    # experiment dir: layer 10's causal catalog lives at a non-standard nested path, and a
    # loader that guessed would silently find zero features for it.
    "multilayer": {
        "layers": [10, 11, 12, 13, 14],
        # Layer 0 is excluded: dead_feature_fraction 0.742, so its dictionary never trained.
        "excluded_layers": [0],
        # "live" -- a downstream SAE encodes the already-perturbed stream, which is the
        # point when testing whether downstream layers compensate. "frozen" would encode
        # from clean activations and bias the result toward "no redundancy".
        "encode_mode": "live",
        "encode_positions_only": True,
        # int applied to every layer, or {layer: k} for the budget/mass-matched arms.
        "features_per_layer": 200,
        "sae_dtype": "float32",
        "sae_paths": {},
        "catalog_paths": {},
        "stats_paths": {},
    },
    "conditions": {
        "gate": True,
        "primary": True,
        "nested": [[14], [13, 14], [12, 13, 14], [11, 12, 13, 14], [10, 11, 12, 13, 14]],
        # Same sizes as nested spans but different depths. Without these the nested curve
        # cannot separate "how many layers" from "which layers", since it always grows
        # downward from 14.
        "non_nested": [[10, 12, 14], [10, 11, 12]],
        "leave_one_out": True,
        "span_knockout": True,
        "budget_matched": {
            "spread_per_layer": 40,
            "concentrated_layer": 11,
            "concentrated_k": [40, 100, 200, 400, 800],
        },
        "downstream_knockout": {"anchor_layers": [11, 14], "downstream_to": 31},
        # Layer 13's knockout is inhibitory while its ablation drop is positive, so the
        # joint ceiling is not a sum of positive contributions; this span drops it.
        "sensitivity_span": [10, 11, 12, 14],
    },
}


@dataclass
class Config:
    """Dataclass wrapper around the YAML configuration."""

    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_overrides(cfg: Dict[str, Any], overrides) -> Dict[str, Any]:
    """Apply ``section.key[.subkey]=value`` strings; values parse as YAML."""
    for item in overrides or []:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form key.path=value")
        value = yaml.safe_load(raw_value)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {item!r} descends through a non-dict at {part!r}")
        node[parts[-1]] = value
    return cfg


def load_config(path: str, overrides=None) -> Config:
    """Load a YAML config: defaults <- include fragments <- own keys <- overrides.

    ``include:`` lists fragment paths resolved relative to the config file and
    deep-merged in order before the file's own keys. Fragments may not include
    further fragments. A config marked ``frozen: true`` may not use
    ``include:`` at all -- frozen configs are fully-resolved reproduction
    records and must stay self-contained.

    The returned config is fully resolved; ``save_config`` of it into the
    experiment dir is therefore complete provenance regardless of composition.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    frozen = False
    if path and os.path.exists(path):
        raw = _load_yaml(path)
        frozen = bool(raw.pop("frozen", False))
        includes = raw.pop("include", [])
        if frozen and includes:
            raise ValueError(f"{path}: frozen configs may not use include:")
        base_dir = os.path.dirname(os.path.abspath(path))
        for fragment in includes:
            fragment_path = fragment if os.path.isabs(fragment) else os.path.join(base_dir, fragment)
            if not os.path.exists(fragment_path):
                raise FileNotFoundError(f"{path}: included fragment not found: {fragment_path}")
            fragment_raw = _load_yaml(fragment_path)
            if "include" in fragment_raw:
                raise ValueError(f"{fragment_path}: fragments may not include further fragments")
            cfg = _deep_update(cfg, fragment_raw)
        cfg = _deep_update(cfg, raw)
    # Retained rather than dropped: reproduction records are exempt from the
    # model-identity rule, since their names are the archive's, and nothing
    # downstream can tell frozen from live once the flag is gone.
    cfg["frozen"] = frozen
    apply_overrides(cfg, overrides)
    return Config(cfg)


def save_config(config: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
