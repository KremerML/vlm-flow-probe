"""Statistical tests and plots for ablation results."""

from typing import Callable, Dict, Iterable, Tuple

import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

from vlmflowprobe.ablation.metrics import mean_probability_drop as compute_probability_drop


def paired_t_test(baseline_probs: Iterable[float], ablated_probs: Iterable[float]) -> Tuple[float, float]:
    base = np.array(list(baseline_probs))
    abl = np.array(list(ablated_probs))
    if stats is None or base.size == 0:
        return 0.0, 1.0
    diff = base - abl
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    if np.std(diff) == 0.0:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(base, abl)
    return float(t_stat), float(p_val)


def effect_size_cohens_d(baseline: Iterable[float], ablated: Iterable[float]) -> float:
    base = np.array(list(baseline))
    abl = np.array(list(ablated))
    if base.size == 0:
        return 0.0
    diff = base - abl
    return float(np.mean(diff) / (np.std(diff) + 1e-8))


def bootstrap_confidence_interval(data: Iterable[float], n_bootstrap: int = 1000) -> Tuple[float, float]:
    values = np.array(list(data))
    if values.size == 0:
        return 0.0, 0.0
    samples = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(values, size=len(values), replace=True)
        samples.append(np.mean(resample))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_ci(
    per_sample: Dict[str, Iterable[float]],
    statistic: Callable[[Dict[str, np.ndarray]], float],
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[float, float, float]:
    """Bootstrap a statistic computed across several conditions measured on the same samples.

    The essential property is that each iteration resamples the *question indices once* and
    applies the same indices to every condition. Conditions here are evaluated on an
    identical, identically-ordered sample set, so their per-question values are strongly
    correlated; resampling each condition independently would discard that pairing and
    inflate the interval on a ratio like ``A(S)/K(S)`` enormously.

    Args:
        per_sample: condition id -> per-question values, all the same length and order.
        statistic: takes ``{condition_id: np.ndarray}`` and returns a scalar.
        n_bootstrap: resampling iterations.
        seed: makes the interval reproducible.
        ci: percentile bounds.

    Returns:
        ``(point_estimate, ci_low, ci_high)``. The bounds are NaN if no iteration produced
        a finite value, which happens when a denominator straddles zero.
    """
    arrays = {key: np.asarray(list(values), dtype=float) for key, values in per_sample.items()}
    if not arrays:
        return float("nan"), float("nan"), float("nan")

    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"paired bootstrap needs one value per question per condition; got lengths "
            f"{ {k: len(v) for k, v in arrays.items()} }"
        )
    n_samples = lengths.pop()
    if n_samples == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(statistic(arrays))

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        value = statistic({key: arr[idx] for key, arr in arrays.items()})
        if np.isfinite(value):
            draws.append(value)

    if not draws:
        return point, float("nan"), float("nan")
    low, high = np.percentile(draws, list(ci))
    return point, float(low), float(high)


def ratio_statistic(numerator: str, denominator: str, min_denominator: float = 1e-9):
    """A statistic for ``paired_bootstrap_ci``: mean(numerator) / mean(denominator).

    Returns NaN when the denominator is at or below ``min_denominator`` so those draws are
    dropped rather than exploding the interval. Layer 13's knockout drop is negative, so its
    ratio is genuinely undefined and must not be reported.
    """

    def statistic(arrays: Dict[str, np.ndarray]) -> float:
        denom = float(np.mean(arrays[denominator]))
        if denom <= min_denominator:
            return float("nan")
        return float(np.mean(arrays[numerator])) / denom

    return statistic


def wilcoxon_vs_controls(
    binding: Iterable[float], control_mean: Iterable[float]
) -> Tuple[float, float]:
    """Per-question signed-rank test of binding against the mean control effect.

    With 15 control sets the empirical p-value is floored at 1/16 = 0.0625, so it cannot
    express the strength of these effects. This test runs over the questions instead, where
    n is 256, and has no such floor.
    """
    diff = np.asarray(list(binding), dtype=float) - np.asarray(list(control_mean), dtype=float)
    if stats is None or diff.size == 0 or np.allclose(diff, 0.0):
        return 0.0, 1.0
    statistic, p_value = stats.wilcoxon(diff)
    return float(statistic), float(p_value)


def z_score_standard_error(z: float, n_sets: int) -> float:
    """Standard error of a z estimated from ``n_sets`` control draws.

    The control standard deviation is itself estimated from a handful of draws, so its
    relative error is about ``1/sqrt(2(n-1))`` and the z inherits it. At 15 sets a reported
    z of 81.6 carries an SE around 15 -- three significant figures are not meaningful.
    """
    if n_sets < 2:
        return float("nan")
    return float(abs(z) / np.sqrt(2.0 * (n_sets - 1)))


def plot_ablation_comparison(results_dict: Dict[str, Dict], save_path: str) -> None:
    from vlmflowprobe.utils.visualization_utils import plot_ablation_comparison as _impl
    _impl(results_dict, save_path)


def generate_statistical_report(results: Dict, metric: str = "pred_token_prob") -> Dict[str, float]:
    binding = results.get("binding_results", [])
    baseline_probs, ablated_probs = _select_metric_arrays(binding, metric)
    t_stat, p_val = paired_t_test(baseline_probs, ablated_probs)
    effect = effect_size_cohens_d(baseline_probs, ablated_probs)
    ci_low, ci_high = bootstrap_confidence_interval(baseline_probs)
    report = {
        "primary_metric": metric,
        "t_stat": t_stat,
        "p_value": p_val,
        "effect_size": effect,
        "baseline_ci_low": ci_low,
        "baseline_ci_high": ci_high,
        "mean_drop": compute_probability_drop(baseline_probs, ablated_probs),
    }

    pred_baseline, pred_ablated = _select_metric_arrays(binding, "pred_token_prob")
    _pred_t, _pred_p = paired_t_test(pred_baseline, pred_ablated)
    report.update(
        {
            "pred_t_stat": _pred_t,
            "pred_p_value": _pred_p,
            "pred_effect_size": effect_size_cohens_d(pred_baseline, pred_ablated),
            "pred_mean_drop": compute_probability_drop(pred_baseline, pred_ablated),
        }
    )

    gt_baseline, gt_ablated = _select_metric_arrays(binding, "gt_token_prob")
    if gt_baseline and gt_ablated:
        report.update(
            {
                "gt_t_stat": paired_t_test(gt_baseline, gt_ablated)[0],
                "gt_p_value": paired_t_test(gt_baseline, gt_ablated)[1],
                "gt_effect_size": effect_size_cohens_d(gt_baseline, gt_ablated),
                "gt_mean_drop": compute_probability_drop(gt_baseline, gt_ablated),
            }
        )
    return report


def _select_metric_arrays(binding_results: list, metric: str):
    if metric == "forced_choice_margin":
        pairs = [
            (r.get("baseline_margin"), r.get("ablated_margin"))
            for r in binding_results
            if r.get("baseline_margin") is not None and r.get("ablated_margin") is not None
        ]
        baseline = [b for b, _ in pairs]
        ablated = [a for _, a in pairs]
        return baseline, ablated

    if metric == "gt_token_prob":
        pairs = [
            (r.get("baseline_gt_prob"), r.get("ablated_gt_prob"))
            for r in binding_results
            if r.get("baseline_gt_prob") is not None and r.get("ablated_gt_prob") is not None
        ]
        baseline = [b for b, _ in pairs]
        ablated = [a for _, a in pairs]
        return baseline, ablated

    baseline = [r.get("baseline_prob", 0.0) for r in binding_results]
    ablated = [r.get("ablated_prob", 0.0) for r in binding_results]
    return baseline, ablated
