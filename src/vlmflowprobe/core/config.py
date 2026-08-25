"""Configuration helpers for SAE experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict
import copy
import os

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "llava-hf/llava-1.5-7b-hf",
        "target_layer": 12,
        "d_model": 4096,
        "conv_mode": "vicuna_v1",
        "model_base": None,
        "activation_site": "residual",
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
        "position_type": "attribute",
    },
    "reproducibility": {
        "seed": 42,
        "deterministic": True,
        "benchmark": False,
    },
    "experiment": {
        "output_dir": "output/sae_experiments/exp_default",
        "output_base": "output/sae_experiments",
        "name": "experiment",
        "use_timestamp": False,
    },
    "dataset": {
        "task_types": ["ChooseAttr"],
        "split": "validation",
        "refined_dataset": "datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv",
        "image_folder": "datasets/images",
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
        "position_type": "attribute",
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
        "position_type": "attribute",
        "mode": "residual",
        "delta_scale": 1.0,
        "operation": "zero",
        "operation_scale": 1.0,
    },
    "random_control": {
        "n_random_sets": 1,
        "sampling": "uniform",
        "seed": 42,
        # NOTE: "correct_mean" is a v1 statistic. v2 stats files carry only causal_score,
        # activation_mean and gradient_mean, so matched sampling against this key silently
        # degrades to uniform -- which is what every published v2 run actually did. New
        # configs should set matched_metric: "activation_mean" and strict_matching: true.
        # The default is left as-is so existing configs still reproduce their numbers.
        "matched_metric": "correct_mean",
        # Raise instead of falling back to uniform when the metric is missing.
        "strict_matching": False,
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
        "search_paths": ["output/sae_experiments"],
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


def load_config(path: str) -> Config:
    """Load YAML config and merge with defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            updates = yaml.safe_load(handle) or {}
        cfg = _deep_update(cfg, updates)
    return Config(cfg)


def save_config(config: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
