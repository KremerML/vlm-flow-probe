"""Opt-in instrumentation for the matched random-control sampler.

Off by default (`random_control.log_matching: false`): full runs should not pay
for this, and the per-draw records are large. Turn it on for smoke tests and
audits, where the question is not "what was the margin drop" but "did matching
actually happen, and how well".

It exists because the failure it detects was invisible for months: configs
asked for matched controls, the metric key was absent, every draw silently fell
back to uniform, and the controls ended up as near-dead features that made every
z-score look enormous. A summary that reports the metric *distributions* of the
binding and control sets side by side makes that failure impossible to miss --
matched controls have a control/binding median ratio near 1, uniform ones are
off by six orders of magnitude.
"""

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional


@dataclass
class DrawRecord:
    """One control feature drawn against one binding feature."""

    binding_feature: int
    binding_value: Optional[float]
    control_feature: int
    control_value: Optional[float]
    #: "matched" (nearest-neighbour window), "uniform_no_target" (the binding
    #: feature had no metric), or "uniform_pool_exhausted".
    path: str
    #: True when the metric came from a fallback key rather than the requested
    #: one -- silent metric-mixing, which strict mode now refuses.
    used_fallback_key: bool = False

    @property
    def abs_error(self) -> Optional[float]:
        if self.binding_value is None or self.control_value is None:
            return None
        return abs(self.control_value - self.binding_value)


@dataclass
class MatchingDiagnostics:
    """Collects per-draw records across every control set of a run."""

    metric: str
    strict: bool = False
    sets: List[List[DrawRecord]] = field(default_factory=list)
    _current: Optional[List[DrawRecord]] = None

    def start_set(self) -> None:
        self._current = []
        self.sets.append(self._current)

    def record(self, **kwargs) -> None:
        if self._current is None:
            self.start_set()
        self._current.append(DrawRecord(**kwargs))

    # ---------------------------------------------------------------- summary
    def summarize(
        self,
        binding_values: Optional[List[float]] = None,
        causal_top_k: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Aggregate every set into one report.

        ``causal_top_k`` (a set of feature ids) enables the archive's smoking-gun
        check: how many control features land in the causal top-K. Uniform
        sampling of 200 features from 32768 puts ~0.7% there by chance; matched
        sampling on activation should put far more, because high-activation
        features are exactly what the causal score selects.
        """
        all_draws = [d for s in self.sets for d in s]
        if not all_draws:
            return {"metric": self.metric, "strict": self.strict, "n_draws": 0}

        paths: Dict[str, int] = {}
        for draw in all_draws:
            paths[draw.path] = paths.get(draw.path, 0) + 1

        control_values = [d.control_value for d in all_draws if d.control_value is not None]
        errors = [d.abs_error for d in all_draws if d.abs_error is not None]
        binding_values = binding_values or [
            d.binding_value for d in all_draws if d.binding_value is not None
        ]

        report: Dict[str, Any] = {
            "metric": self.metric,
            "strict": self.strict,
            "n_sets": len(self.sets),
            "n_draws": len(all_draws),
            "draw_paths": paths,
            "matched_fraction": paths.get("matched", 0) / len(all_draws),
            "n_fallback_key_uses": sum(1 for d in all_draws if d.used_fallback_key),
            "binding": _describe(binding_values),
            "control": _describe(control_values),
            "abs_error": _describe(errors),
        }

        b_med = report["binding"].get("median")
        c_med = report["control"].get("median")
        if b_med and c_med is not None:
            report["control_over_binding_median_ratio"] = c_med / b_med

        if causal_top_k is not None:
            controls = [d.control_feature for d in all_draws]
            in_top = sum(1 for f in controls if f in causal_top_k)
            report["controls_in_causal_top_k"] = {
                "k": len(causal_top_k),
                "n_in": in_top,
                "fraction": in_top / len(controls) if controls else None,
            }
        return report


def _describe(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "min": ordered[0],
        "p25": ordered[max(0, int(0.25 * (n - 1)))],
        "median": median(ordered),
        "p75": ordered[max(0, int(0.75 * (n - 1)))],
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable block for smoke-test logs."""
    if not report.get("n_draws"):
        return "  (no matching draws recorded)"
    lines = [
        f"  metric={report['metric']!r} strict={report['strict']} "
        f"sets={report['n_sets']} draws={report['n_draws']}",
        f"  draw paths: {report['draw_paths']} "
        f"(matched {100 * report['matched_fraction']:.1f}%)",
        f"  fallback-key uses: {report['n_fallback_key_uses']}",
    ]
    for name in ("binding", "control", "abs_error"):
        d = report.get(name, {})
        if d.get("n"):
            lines.append(
                f"  {name:<10} n={d['n']:<5} median={d['median']:.6g} "
                f"mean={d['mean']:.6g} min={d['min']:.6g} max={d['max']:.6g}"
            )
    ratio = report.get("control_over_binding_median_ratio")
    if ratio is not None:
        verdict = "MATCHED" if 0.2 <= ratio <= 5 else "NOT MATCHED"
        lines.append(f"  control/binding median ratio: {ratio:.4g}  -> {verdict}")
    top = report.get("controls_in_causal_top_k")
    if top:
        lines.append(
            f"  controls in causal top-{top['k']}: {top['n_in']} "
            f"({100 * (top['fraction'] or 0):.1f}%)"
        )
    return "\n".join(lines)


def matched_pool_depth(
    feature_stats: Dict[int, dict],
    binding_features: List[int],
    metric: str = "activation_mean",
) -> Dict[str, Any]:
    """How many independent matched control sets the dictionary can actually supply.

    Matching draws controls that resemble the binding set on ``metric``. If few
    non-binding features fall in the binding set's range, the "independent"
    control sets are forced to reuse the same handful of features -- their
    between-set variance collapses and any z computed across them is inflated,
    by a different mechanism than the ghost-feature bug but with the same
    effect. This measures the ceiling before a run rather than after.
    """
    binding_set = {int(f) for f in binding_features}
    values = {
        int(f): s.get(metric)
        for f, s in feature_stats.items()
        if isinstance(s, dict) and s.get(metric) is not None
    }
    binding_values = sorted(v for f, v in values.items() if f in binding_set)
    if not binding_values:
        return {"metric": metric, "error": f"no binding feature carries {metric!r}"}

    lo, hi = binding_values[0], binding_values[-1]
    n = len(binding_values)
    q1, q3 = binding_values[n // 4], binding_values[(3 * n) // 4]
    in_range = [f for f, v in values.items() if f not in binding_set and lo <= v <= hi]
    in_iqr = [f for f in in_range if q1 <= values[f] <= q3]

    n_binding = len(binding_set)
    max_disjoint = len(in_range) / n_binding if n_binding else 0.0
    return {
        "metric": metric,
        "n_binding": n_binding,
        "binding_range": [lo, hi],
        "binding_iqr": [q1, q3],
        "n_candidates_in_range": len(in_range),
        "n_candidates_in_iqr": len(in_iqr),
        "dictionary_size": len(values),
        "max_disjoint_matched_sets": max_disjoint,
        "sufficient_for_one_set": len(in_range) >= n_binding,
    }
