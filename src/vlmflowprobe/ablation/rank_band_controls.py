"""Controls that are disjoint from their binding set by construction.

Why not sampled matched controls
--------------------------------
The selection rule is a top-k cut on a causal score, so no non-selected
feature has a score inside the binding range -- matching on the selection
variable is impossible by definition. Matching on activation instead fails
because the causal score is close to a monotone function of activation
(measured rank correlation 0.949 at layer 11 of LLaVA-1.5): only 228
non-selected features at layer 11, and 185 at layer 14, lie inside the binding
activation range, against 200 needed. Fifteen "independent" matched sets then
overlap 97% and carry no between-set variance, so a z over them divides by a
number that is not the standard deviation of any population.

What replaces it
----------------
Controls defined by construction rather than drawn from a pool:

``rank_band``      the next k features down the same ranking. Disjoint from
                   the binding set by definition, identical in count, similar
                   in activation. Perturbation matching is an empirical
                   property to be checked per run, not an assumption -- at
                   layer 11 it came out at 0.01844 against the binding set's
                   0.01855.

``activation_top`` the top k by activation alone, ignoring the causal score.
                   Carries more activation mass and more perturbation than the
                   binding set, so it is biased in favour of the null: a
                   binding set that still wins does not depend on matching.

``ranking_difference`` the features on which the causal and activation
                   rankings disagree, which isolates what the gradient factor
                   contributes over an activation filter.

Inference moves from "over feature sets", which is unidentified here, to "over
questions", which is a real superpopulation.
"""

from typing import Dict, List, Optional, Sequence


def rank_features(feature_stats: Dict[int, dict], key: str) -> List[int]:
    """Feature ids ordered by ``key``, descending, ties broken by id.

    Deterministic ordering matters: the bands below are defined by position in
    this list, so an unstable sort would silently change which features a
    control contains.
    """
    return [
        int(f) for f, _ in sorted(
            feature_stats.items(),
            key=lambda kv: (-float(kv[1].get(key, 0.0) or 0.0), int(kv[0])),
        )
    ]


def rank_band(
    feature_stats: Dict[int, dict], lo: int, hi: int, key: str = "causal_score"
) -> List[int]:
    """Features at ranks ``[lo, hi)`` of the ordering induced by ``key``."""
    if lo < 0 or hi < lo:
        raise ValueError(f"invalid band [{lo}, {hi})")
    return rank_features(feature_stats, key)[lo:hi]


def adjacent_band_control(
    feature_stats: Dict[int, dict], k: int, key: str = "causal_score"
) -> List[int]:
    """The control for a top-k binding set: ranks ``[k, 2k)``."""
    band = rank_band(feature_stats, k, 2 * k, key=key)
    if len(band) < k:
        raise ValueError(
            f"ranking holds {len(band)} features at ranks {k}..{2 * k}, needs {k}; "
            f"the dictionary is too small for an adjacent-band control at k={k}"
        )
    return band


def activation_top_control(feature_stats: Dict[int, dict], k: int) -> List[int]:
    """Top k by activation alone -- the deliberately conservative control."""
    return rank_band(feature_stats, 0, k, key="activation_mean")


def ranking_difference(
    feature_stats: Dict[int, dict], k: int
) -> Dict[str, List[int]]:
    """Split the top-k of two rankings into shared and exclusive parts.

    ``causal_only`` are selected by the causal score but not by activation;
    ``activation_only`` the reverse. If the two behave alike under ablation the
    gradient factor adds nothing beyond an activation filter.
    """
    causal = set(rank_band(feature_stats, 0, k, key="causal_score"))
    activation = set(rank_band(feature_stats, 0, k, key="activation_mean"))
    return {
        "shared": sorted(causal & activation),
        "causal_only": sorted(causal - activation),
        "activation_only": sorted(activation - causal),
    }


def activation_matched_control(
    feature_stats: Dict[int, dict],
    binding: Sequence[int],
    metric: str = "activation_mean",
) -> List[int]:
    """Nearest-neighbour activation matches, excluding the binding set.

    Retained to document what the sampled matched control actually produces.
    Where the candidate pool is exhausted -- which is the normal case at k=200
    -- this reaches down in activation and returns a set that is not matched;
    ``matched_pool_depth`` in ``matching_diagnostics`` quantifies that before a
    run rather than after.
    """
    import bisect

    binding_set = {int(b) for b in binding}
    pool = sorted(
        (int(f) for f in feature_stats if int(f) not in binding_set),
        key=lambda f: float(feature_stats[f].get(metric, 0.0) or 0.0),
    )
    values = [float(feature_stats[f].get(metric, 0.0) or 0.0) for f in pool]
    chosen: List[int] = []
    used = set()
    for b in binding:
        target = float(feature_stats[int(b)].get(metric, 0.0) or 0.0)
        i = bisect.bisect_left(values, target)
        picked = None
        for offset in range(len(pool)):
            for j in (i + offset, i - offset):
                if 0 <= j < len(pool) and pool[j] not in used:
                    picked = pool[j]
                    break
            if picked is not None:
                break
        if picked is None:
            break
        chosen.append(picked)
        used.add(picked)
    return chosen


def dose_response_bands(
    feature_stats: Dict[int, dict], width: int = 40, n_bands: int = 10,
    key: str = "causal_score",
) -> List[Dict[str, object]]:
    """Disjoint equal-size bands down the ranking.

    Equal size removes the budget confound: a decline across bands cannot be a
    feature-count effect, because every band ablates the same number.
    """
    ordering = rank_features(feature_stats, key)
    bands = []
    for i in range(n_bands):
        lo, hi = i * width, (i + 1) * width
        features = ordering[lo:hi]
        if len(features) < width:
            break
        bands.append({"lo": lo, "hi": hi, "features": features})
    return bands


def random_subsets(
    features: Sequence[int], size: int, n_sets: int, seed: int = 42
) -> List[List[int]]:
    """Random equal-size subsets, for the one null that IS estimable.

    Subsets of a 200-feature pool overlap by about ``size / 200``, so unlike
    the matched-control sets these carry real between-set variance. They also
    show whether an effect is spread across a set or concentrated in a few of
    its members.
    """
    import random

    pool = [int(f) for f in features]
    if size > len(pool):
        raise ValueError(f"cannot draw {size} features from a pool of {len(pool)}")
    rng = random.Random(seed)
    return [sorted(rng.sample(pool, size)) for _ in range(n_sets)]


def perturbation_match_quality(
    binding_perturbation: Optional[float], control_perturbation: Optional[float]
) -> Optional[Dict[str, float]]:
    """How closely a control's realized perturbation matches its binding set.

    Reported per condition so that "the control perturbs comparably" is a
    measurement rather than a claim. A control perturbing MORE while producing
    LESS effect is the strongest form of the result, since no magnitude-only
    account of the damage can explain it.
    """
    if not binding_perturbation or control_perturbation is None:
        return None
    ratio = control_perturbation / binding_perturbation
    return {
        "binding": binding_perturbation,
        "control": control_perturbation,
        "ratio": ratio,
        "relative_difference": ratio - 1.0,
        "control_perturbs_more": ratio > 1.0,
    }
