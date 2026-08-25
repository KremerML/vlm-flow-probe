"""Evaluation metrics for SAE experiments."""

from typing import Iterable

import numpy as np


def accuracy_at_k(predictions: Iterable[str], targets: Iterable[str], k: int = 1) -> float:
    preds = list(predictions)
    targs = list(targets)
    if not preds:
        return 0.0
    correct = [p == t for p, t in zip(preds, targs)]
    return sum(correct) / len(correct)


def mean_probability_drop(baseline_probs: Iterable[float], ablated_probs: Iterable[float]) -> float:
    baseline = np.array(list(baseline_probs))
    ablated = np.array(list(ablated_probs))
    if baseline.size == 0:
        return 0.0
    return float(np.mean(baseline - ablated))


def forced_choice_margin(true_logprobs: Iterable[float], false_logprobs: Iterable[float]) -> float:
    true_vals = np.array(list(true_logprobs))
    false_vals = np.array(list(false_logprobs))
    if true_vals.size == 0 or false_vals.size == 0:
        return 0.0
    return float(np.mean(true_vals - false_vals))


def mean_margin_drop(baseline_margins: Iterable[float], ablated_margins: Iterable[float]) -> float:
    base = np.array(list(baseline_margins))
    abl = np.array(list(ablated_margins))
    if base.size == 0:
        return 0.0
    return float(np.mean(base - abl))


