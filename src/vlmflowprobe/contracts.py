"""The output-artifact contract and the condition-id grammar.

These names are load-bearing across tool boundaries: the multilayer analyzer,
the distiller, and the figure notebooks in the archive all key on them. Treat
any change as a breaking schema change.
"""

import os
import re
from typing import Dict, List

# ---------------------------------------------------------------- artifacts
#: Files a completed single-layer experiment directory contains.
EXPERIMENT_ARTIFACTS = [
    "config.yaml",           # fully-resolved config (save_config)
    "provenance.json",       # git SHA, versions, GPU, argv, seed
    "sae_checkpoint.pt",     # stage 01 (gitignored; ~1 GB)
    "reconstruction_eval.json",
]
#: Files in the sibling ``{experiment_dir}_causal`` directory.
CAUSAL_ARTIFACTS = [
    "causal_feature_catalog.json",   # stage 02; descending causal-score order
    "causal_feature_stats.json",     # stage 02 (gitignored; distilled sibling committed)
    "causal_summary.json",
    os.path.join("results", "ablation_v2_results.json"),  # stage 03 (gitignored)
]
#: Files a multilayer run directory contains.
MULTILAYER_ARTIFACTS = [
    "sample_cache.json",             # per-sample baselines+positions; ORDER AUTHORITY
    "checkpoint.jsonl",              # resume log (gitignored)
    os.path.join("conditions", "<id>", "results.json"),   # gitignored
    os.path.join("conditions", "<id>", "summary.json"),   # committed; per_sample_distilled + question_ids_sha1
    os.path.join("analysis", "multilayer_summary.json"),
    os.path.join("analysis", "multilayer_summary.md"),
]

# ---------------------------------------------------------------- condition ids
#: The condition-id grammar. ``analyze_multilayer`` pairs ablation and
#: knockout conditions by these shapes; the archive's figure notebook keys on
#: them verbatim. ``L<range>`` renders contiguous spans as ``10-14`` and
#: non-contiguous sets as ``10,12,14`` (see ``_compact_layer_range``).
CONDITION_ID_PATTERNS: Dict[str, str] = {
    "a0_regression": r"^A0_regression_L\d+$",
    "gate_none": r"^gate_none$",
    "gate_passthrough": r"^gate_passthrough(_delta)?_L[\d,-]+$",
    "joint": r"^joint_L[\d,-]+$",
    "span_knockout": r"^span_knockout_L[\d,-]+$",
    "single_knockout": r"^knockout_L\d+$",
    "nested": r"^nested(_knockout)?_L[\d,-]+$",
    "non_nested": r"^nonnested(_knockout)?_L[\d,-]+$",
    "leave_one_out": r"^loo_drop\d+$",
    "budget_spread": r"^budget_spread\d+x\d+$",
    "budget_concentrated": r"^budget_concentrated_L\d+_k\d+$",
    "downstream": r"^downstream_(ablate|combined)_L\d+$|^downstream_knockout_L[\d,-]+$",
    "sensitivity": r"^sensitivity_(joint|knockout)_L[\d,-]+$|^sensitivity_passthrough_tail_L\d+$",
    # Ablation with Image->Question severed at every layer: what the features do
    # when no image information can reach the text positions at all.
    "isolation": r"^full_knockout_L[\d,-]+$|^isolated_(ablate_L\d+|joint_L[\d,-]+)$",
}

_COMPILED = {name: re.compile(pattern) for name, pattern in CONDITION_ID_PATTERNS.items()}


def classify_condition_id(condition_id: str) -> str:
    """Family name for a condition id; raises ValueError for unknown shapes."""
    for name, pattern in _COMPILED.items():
        if pattern.match(condition_id):
            return name
    raise ValueError(f"condition id {condition_id!r} matches no known family")


# ---------------------------------------------------------------- checks
def check_experiment_dir(experiment_dir: str) -> Dict[str, List[str]]:
    """Which contract artifacts exist for an experiment; used by smoke tests.

    Returns {"present": [...], "missing": [...]} over the artifacts that are
    expected to exist on disk after a full single-layer run (gitignored ones
    included — this checks a live run directory, not a git checkout).
    """
    present, missing = [], []
    for rel in EXPERIMENT_ARTIFACTS:
        (present if os.path.exists(os.path.join(experiment_dir, rel)) else missing).append(rel)
    causal_dir = experiment_dir + "_causal"
    for rel in CAUSAL_ARTIFACTS:
        path = os.path.join(causal_dir, rel)
        label = os.path.join("_causal", rel)
        (present if os.path.exists(path) else missing).append(label)
    return {"present": present, "missing": missing}
