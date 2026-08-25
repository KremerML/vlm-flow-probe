"""Turn multi-layer condition outputs into the numbers the writeup needs.

Three quantities, in descending order of how much weight they can carry:

**R(S) = A(S) / K(S)** -- the redundancy index. Joint ablation drop over a span divided by
the multi-layer knockout drop over the *same span, same samples, same filter policy*. This
generalises "% of knockout ceiling" and, unlike the published figures, has a numerator and a
denominator from the same sample set. The hypothesis predicts R(S) well above the 0.50-0.68
single layers reach. It assumes no additivity model at all, which is why it leads.

**Comparative saturation** -- fit A(S_k) and K(S_k) over span size k. If ablation and
knockout grow at the same rate the shortfall is a fixed proportion of the flow rather than
redundancy; if ablation saturates faster, features become redundant faster than the flow
does. The null here is fitted to the knockout data itself, so no arbitrary curve is assumed.
Note this is an *aggregate* statement about the two slopes. It does not license the stronger
claim that R is constant span by span -- `redundancy_by_span` measures that directly, and on
this run it is not: the per-span ratios disperse well beyond their own intervals.

**Interaction index** -- A(S) against the sum of single-layer drops. Sub-additivity alone
proves nothing, because the metric saturates even under true independence; so the ratio is
reported *calibrated by the knockout's own sub-additivity*, which the same run measures.

    PY=LLaVA-NeXT/.venv/bin/python
    $PY sae_experiments/tools/analyze_multilayer_ablation.py \
        --experiment_dir output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vlmflowprobe.ablation.statistical_analysis import (  # noqa: E402
    paired_bootstrap_ci,
    ratio_statistic,
    wilcoxon_vs_controls,
    z_score_standard_error,
)

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


def load_condition(directory):
    """Read one condition, preferring the committed summary over the gitignored raw file.

    `summary.json` carries the per-sample margin drops under `per_sample_distilled`, so a
    fresh clone can reproduce every number here. `results.json` is only consulted for runs
    that predate that fold, or when a summary somehow lacks the block.
    """
    summary_path = os.path.join(directory, "summary.json")
    results_path = os.path.join(directory, "results.json")

    payload = None
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            payload = json.load(handle)
        folded = payload.get("per_sample_distilled") or {}
        if folded.get("margin_drops") is not None:
            payload["margin_drops"] = [float(v) for v in folded["margin_drops"]]
            # Carried so paired statistics can verify the two arms are the same questions in
            # the same order. distill_results writes it; until now nothing read it back.
            payload["question_ids_sha1"] = folded.get("question_ids_sha1")
            return payload

    if os.path.exists(results_path):
        with open(results_path) as handle:
            payload = json.load(handle)
        payload["margin_drops"] = margin_drops(payload.get("per_sample", []))
        payload["question_ids_sha1"] = None  # raw records predate the folded hash
        return payload

    if payload is not None:
        payload["margin_drops"] = []
    return payload


def load_conditions(experiment_dir):
    """Load every completed condition, keyed by id, with per-sample margin drops."""
    root = os.path.join(experiment_dir, "conditions")
    if not os.path.isdir(root):
        raise SystemExit(f"no conditions directory under {experiment_dir}")

    conditions = {}
    for condition_id in sorted(os.listdir(root)):
        payload = load_condition(os.path.join(root, condition_id))
        if payload is None:
            continue
        conditions[condition_id] = {
            "summary": payload.get("summary", {}),
            "significance": payload.get("significance", {}),
            "meta": payload.get("meta", {}),
            "control_summaries": payload.get("control_summaries", []),
            "margin_drops": payload["margin_drops"],
            "question_ids_sha1": payload.get("question_ids_sha1"),
        }
    return conditions


def margin_drops(rows):
    """Per-sample baseline_margin - ablated_margin, the quantity everything is built on."""
    drops = []
    for row in rows:
        baseline = row.get("baseline_margin")
        ablated = row.get("ablated_margin")
        if baseline is None or ablated is None:
            continue
        drops.append(float(baseline) - float(ablated))
    return drops


def redundancy_index(conditions, ablation_id, knockout_id, seed=42):
    """R = A/K with a paired bootstrap CI, or a reason it cannot be reported."""
    if ablation_id not in conditions or knockout_id not in conditions:
        return {"status": "missing_condition"}

    ablation = conditions[ablation_id]["margin_drops"]
    knockout = conditions[knockout_id]["margin_drops"]
    if not ablation or not knockout:
        return {"status": "no_per_sample_data"}
    if len(ablation) != len(knockout):
        return {"status": "sample_count_mismatch"}

    # The bootstrap resamples one index vector and applies it to both arms, which is only a
    # paired test if both arms are the same questions in the same order. Equal length is not
    # enough. Both hashes are present for every condition in the folded format.
    sha_ablation = conditions[ablation_id].get("question_ids_sha1")
    sha_knockout = conditions[knockout_id].get("question_ids_sha1")
    if sha_ablation and sha_knockout and sha_ablation != sha_knockout:
        return {
            "status": "sha1_mismatch",
            "ablation_condition": ablation_id,
            "knockout_condition": knockout_id,
            "ablation_question_ids_sha1": sha_ablation,
            "knockout_question_ids_sha1": sha_knockout,
        }

    mean_knockout = float(np.mean(knockout))
    result = {
        "ablation_condition": ablation_id,
        "knockout_condition": knockout_id,
        "ablation_mean": float(np.mean(ablation)),
        "knockout_mean": mean_knockout,
        "n_samples": len(ablation),
        "question_ids_sha1": sha_ablation or sha_knockout,
    }

    # A negative ceiling makes the ratio meaningless, not merely noisy. Layer 13's knockout
    # is inhibitory, so this is a real case and must be reported as undefined.
    if mean_knockout <= 0:
        result["status"] = "undefined_negative_ceiling"
        result["ratio"] = None
        return result

    point, low, high = paired_bootstrap_ci(
        {"ablation": ablation, "knockout": knockout},
        ratio_statistic("ablation", "knockout"),
        seed=seed,
    )
    result.update(
        {"status": "ok", "ratio": point, "ci_low": low, "ci_high": high}
    )
    return result



# Ablation/knockout pairs measured over the same span, on the same 256 questions. The runner's
# ids are irregular (the full span is joint_/span_knockout_, not nested_/nested_knockout_), so
# they are named here rather than derived. "{14}" is simultaneously the size-1 nested span and
# the layer-14 single. Layers 10, 12 and 13 have no single-layer entry: their ablations predate
# this harness and carry no per-sample record that can be paired against a knockout arm.
SPAN_PAIRS = (
    ("nested", "{14}", "nested_L14", "nested_knockout_L14", (14,)),
    ("nested", "{13,14}", "nested_L13-14", "nested_knockout_L13-14", (13, 14)),
    ("nested", "{12,13,14}", "nested_L12-14", "nested_knockout_L12-14", (12, 13, 14)),
    ("nested", "{11-14}", "nested_L11-14", "nested_knockout_L11-14", (11, 12, 13, 14)),
    ("nested", "{10-14}", "joint_L10-14", "span_knockout_L10-14", (10, 11, 12, 13, 14)),
    ("single", "{11}", "A0_regression_L11", "knockout_L11", (11,)),
    ("non-nested", "{10,11,12}", "nonnested_L10-12", "nonnested_knockout_L10-12", (10, 11, 12)),
    ("non-nested", "{10,12,14}", "nonnested_L10,12,14", "nonnested_knockout_L10,12,14", (10, 12, 14)),
    ("sensitivity", "{10,11,12,14}", "sensitivity_joint_L10,11,12,14",
     "sensitivity_knockout_L10,11,12,14", (10, 11, 12, 14)),
)


def redundancy_by_span(conditions, seed=42, pairs=SPAN_PAIRS):
    """R = A/K with a paired bootstrap CI for every span the matrix measures both arms of.

    The joint-span R alone cannot distinguish "ablation recovers a fixed share of the flow"
    from "ablation recovers a share that grows as the span covers more of it" -- the second is
    what a redundancy account predicts. Reporting R per span separates them.
    """
    rows = []
    for kind, label, ablation_id, knockout_id, layers in pairs:
        entry = {
            "kind": kind,
            "label": label,
            "span_size": len(layers),
            "layers": list(layers),
        }
        entry.update(redundancy_index(conditions, ablation_id, knockout_id, seed))
        rows.append(entry)
    return rows


def redundancy_trend(conditions, seed=42, n_bootstrap=10000, pairs=SPAN_PAIRS):
    """Does R change with span size? One resample of questions, applied to every span.

    Each bootstrap draw resamples the question set once and recomputes R for all nested spans
    from that same draw, so the spans stay paired with each other as well as internally. That
    is what makes the differences and the trend slope testable: the spans share their sampling
    noise, and only what differs between them survives.

    Reports the slope of R on span size (a redundancy account predicts a positive slope), the
    pooled ratio, the observed spread, and whether any single value lies inside every span's
    interval -- the direct test of a "constant fraction" reading.
    """
    nested = [p for p in pairs if p[0] == "nested"]
    arms = []
    for _, label, ablation_id, knockout_id, layers in nested:
        if ablation_id not in conditions or knockout_id not in conditions:
            return {"status": "missing_condition", "condition": ablation_id}
        a = conditions[ablation_id]["margin_drops"]
        k = conditions[knockout_id]["margin_drops"]
        if not a or not k or len(a) != len(k):
            return {"status": "no_per_sample_data", "condition": ablation_id}
        arms.append((label, len(layers), np.asarray(a, float), np.asarray(k, float)))

    hashes = {
        conditions[cid].get("question_ids_sha1")
        for _, _, ab, ko, _ in nested
        for cid in (ab, ko)
    }
    hashes.discard(None)
    if len(hashes) > 1:
        return {"status": "sha1_mismatch", "hashes": sorted(hashes)}

    n_samples = len(arms[0][2])
    if any(len(a) != n_samples for _, _, a, _ in arms):
        return {"status": "sample_count_mismatch"}

    rng = np.random.default_rng(seed)
    index = rng.integers(0, n_samples, size=(n_bootstrap, n_samples))

    sizes = np.array([size for _, size, _, _ in arms], dtype=float)
    draws = np.stack([a[index].mean(1) / k[index].mean(1) for _, _, a, k in arms])
    point = np.array([a.mean() / k.mean() for _, _, a, k in arms])

    centred = sizes - sizes.mean()
    denominator = float((centred ** 2).sum())
    slope_draws = (centred[:, None] * (draws - draws.mean(0))).sum(0) / denominator
    slope_point = float((centred * (point - point.mean())).sum() / denominator)

    pooled_draws = np.stack([a[index].mean(1) for _, _, a, _ in arms]).sum(0) / np.stack(
        [k[index].mean(1) for _, _, _, k in arms]
    ).sum(0)
    pooled_point = float(sum(a.mean() for _, _, a, _ in arms) / sum(k.mean() for _, _, _, k in arms))

    spread_draws = draws.max(0) - draws.min(0)

    def interval(values):
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    per_span = []
    for (label, size, _, _), value, row in zip(arms, point, draws):
        low, high = interval(row)
        per_span.append(
            {"label": label, "span_size": size, "ratio": float(value), "ci_low": low, "ci_high": high}
        )

    highest_low = max(row["ci_low"] for row in per_span)
    lowest_high = min(row["ci_high"] for row in per_span)

    slope_low, slope_high = interval(slope_draws)
    pooled_low, pooled_high = interval(pooled_draws)
    spread_low, spread_high = interval(spread_draws)

    # A redundancy account predicts R rises as the span covers more of the compensating layers.
    trending = not (slope_low <= 0 <= slope_high)
    constant = highest_low <= lowest_high

    return {
        "status": "ok",
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "n_samples": n_samples,
        "question_ids_sha1": next(iter(hashes)) if hashes else None,
        "per_span": per_span,
        "slope_per_layer": slope_point,
        "slope_ci_low": slope_low,
        "slope_ci_high": slope_high,
        "slope_excludes_zero": bool(trending),
        "pooled_ratio": pooled_point,
        "pooled_ci_low": pooled_low,
        "pooled_ci_high": pooled_high,
        "spread": float(point.max() - point.min()),
        "spread_ci_low": spread_low,
        "spread_ci_high": spread_high,
        "common_value": bool(constant),
        "highest_ci_low": float(highest_low),
        "lowest_ci_high": float(lowest_high),
        "interpretation": _trend_interpretation(
            slope_point, slope_low, slope_high, pooled_point, pooled_low, pooled_high,
            point.min(), point.max(), constant, trending,
        ),
    }


def _trend_interpretation(slope, slope_low, slope_high, pooled, pooled_low, pooled_high,
                          lowest, highest, constant, trending):
    parts = [
        f"Pooled recovery over the nested spans is {pooled:.1%} "
        f"(95% CI {pooled_low:.1%} - {pooled_high:.1%})."
    ]
    if constant:
        parts.append(
            f"Every span's interval contains a common value, so a single fixed share "
            f"({lowest:.1%} - {highest:.1%}) is consistent with all of them."
        )
    else:
        parts.append(
            f"The per-span ratios run {lowest:.1%} to {highest:.1%} with NO value inside every "
            "interval, so R is not constant across spans and must not be reported as such."
        )
    if trending:
        parts.append(
            f"R trends with span size at {slope:+.4f} per layer "
            f"(95% CI {slope_low:+.4f} - {slope_high:+.4f}), which excludes zero."
        )
    else:
        parts.append(
            f"R shows no trend with span size ({slope:+.4f} per layer, 95% CI "
            f"{slope_low:+.4f} - {slope_high:+.4f}, containing zero). A redundancy account "
            "predicts R rises as the span covers more of the layers said to compensate; it "
            "does not rise."
        )
    return " ".join(parts)


def saturation_fit(span_sizes, values):
    """Fit y = a(1 - exp(-b k)); returns (a, b) or None when scipy is unavailable."""
    if curve_fit is None or len(span_sizes) < 3:
        return None

    def model(k, a, b):
        return a * (1.0 - np.exp(-b * k))

    try:
        params, _ = curve_fit(
            model,
            np.asarray(span_sizes, dtype=float),
            np.asarray(values, dtype=float),
            p0=[max(values) if values else 1.0, 0.5],
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None

    a, b = float(params[0]), float(params[1])
    # As b -> 0 the model degenerates to the straight line y = (a*b)k: a and b individually
    # run away while only their product is identified. Quoting b then means nothing, so flag
    # it and report the slope instead.
    degenerate = b * max(span_sizes) < 0.1
    return {
        "a": a,
        "b": b,
        "degenerate": degenerate,
        "initial_slope": a * b,
    }


def nested_curves(conditions):
    """Ablation and knockout saturation as a function of span size."""
    ablation_points, knockout_points = [], []
    for condition_id, payload in conditions.items():
        summary = payload["summary"]
        size = len(summary.get("layers") or summary.get("knockout_layers") or [])
        if not size:
            continue
        drop = summary.get("mean_margin_drop")
        if drop is None:
            continue
        if condition_id.startswith(("nested_L", "joint_L")):
            ablation_points.append((size, float(drop)))
        elif condition_id.startswith(("nested_knockout_L", "span_knockout_L")):
            knockout_points.append((size, float(drop)))

    ablation_points.sort()
    knockout_points.sort()
    out = {
        "ablation": [{"span_size": s, "margin_drop": d} for s, d in ablation_points],
        "knockout": [{"span_size": s, "margin_drop": d} for s, d in knockout_points],
        "ablation_fit": saturation_fit(*zip(*ablation_points)) if len(ablation_points) >= 3 else None,
        "knockout_fit": saturation_fit(*zip(*knockout_points)) if len(knockout_points) >= 3 else None,
    }
    if out["ablation_fit"] and out["knockout_fit"]:
        out["b_minus_d"] = out["ablation_fit"]["b"] - out["knockout_fit"]["b"]
        if out["ablation_fit"]["degenerate"] or out["knockout_fit"]["degenerate"]:
            slope_a = out["ablation_fit"]["initial_slope"]
            slope_k = out["knockout_fit"]["initial_slope"]
            out["degenerate"] = True
            out["slope_ratio"] = slope_a / slope_k if slope_k else None
            out["interpretation"] = (
                "NEITHER curve saturates over this range -- both fits collapsed to straight "
                f"lines (ablation {slope_a:.3f}/layer, knockout {slope_k:.3f}/layer), so b and "
                "d are not identified and b - d carries no information. What is identified is "
                f"the slope ratio, {out['slope_ratio']:.3f}: over spans of 1-5 layers the two "
                "curves grow in a fixed proportion, with no sign of the ablation curve turning "
                "over sooner. Wider spans would be needed to see saturation at all. This is a "
                "statement about the two slopes and NOT about R span by span -- see "
                "redundancy_by_span, where the ratios disperse beyond their own intervals."
            )
        else:
            out["degenerate"] = False
            out["interpretation"] = (
                "ablation saturates faster than the flow does (redundancy)"
                if out["b_minus_d"] > 0
                else "ablation and knockout saturate alike; the shortfall is a fixed proportion "
                "of the flow in aggregate (see redundancy_by_span for the span-by-span picture)"
            )
    return out


def single_layer_ablation(conditions, layer, features_per_layer=None):
    """Find a standalone SAE ablation of exactly `layer`, by shape rather than by condition-id.

    Several ids can describe the same intervention (A0_regression_L11, downstream_ablate_L11,
    budget_concentrated_L11_k200), so match structurally and require the survivors to agree.
    """
    candidates = []
    for condition_id, payload in sorted(conditions.items()):
        summary = payload.get("summary", {})
        if summary.get("kind") != "sae" or summary.get("knockout_layers"):
            continue
        if list(summary.get("layers") or []) != [layer]:
            continue
        if features_per_layer is not None and summary.get("total_features") != features_per_layer:
            continue
        candidates.append((condition_id, summary.get("mean_margin_drop")))

    if not candidates:
        return None, None
    values = [v for _, v in candidates if v is not None]
    if values and max(values) - min(values) > 1e-6:
        raise ValueError(
            f"single-layer ablations of L{layer} disagree: "
            + ", ".join(f"{cid}={v:.6f}" for cid, v in candidates)
        )
    return candidates[0]


def interaction_index(conditions, span, joint_id, span_knockout_id):
    """A(S) vs the sum of single-layer effects, calibrated by the knockout's own sub-additivity."""
    joint_summary = conditions.get(joint_id, {}).get("summary", {})
    per_layer = joint_summary.get("features_per_layer") or {}
    canonical_k = next(iter(per_layer.values()), None)

    single_ablation, single_knockout, missing = {}, {}, []
    for layer in span:
        condition_id, value = single_layer_ablation(conditions, layer, canonical_k)
        if value is None:
            missing.append(layer)
        else:
            single_ablation[layer] = {"condition_id": condition_id, "margin_drop": value}
        knockout = conditions.get(f"knockout_L{layer}")
        if knockout:
            single_knockout[layer] = knockout["summary"].get("mean_margin_drop")

    result = {"span": list(span)}
    joint = joint_summary.get("mean_margin_drop")
    span_knockout = conditions.get(span_knockout_id, {}).get("summary", {}).get("mean_margin_drop")

    # A partial sum would silently understate the denominator and inflate rho, so refuse it.
    if missing:
        result["ablation_status"] = "incomplete"
        result["missing_single_layer_ablations"] = missing
    elif joint is not None:
        total = float(sum(v["margin_drop"] for v in single_ablation.values()))
        result["ablation_status"] = "ok"
        result["single_ablation_sources"] = {
            str(k): v["condition_id"] for k, v in single_ablation.items()
        }
        result["joint_ablation"] = float(joint)
        result["sum_single_ablation"] = total
        result["rho_ablation"] = float(joint) / total if total else None

    missing_knockout = [l for l in span if l not in single_knockout]
    if missing_knockout:
        result["knockout_status"] = "incomplete"
        result["missing_single_layer_knockouts"] = missing_knockout
    elif span_knockout is not None:
        total = float(sum(single_knockout.values()))
        result["knockout_status"] = "ok"
        result["span_knockout"] = float(span_knockout)
        result["sum_single_knockout"] = total
        result["rho_knockout"] = float(span_knockout) / total if total else None

    rho_a, rho_k = result.get("rho_ablation"), result.get("rho_knockout")
    if rho_a is not None and rho_k:
        # The calibrated quantity: how sub-additive the ablation is *relative to* how
        # sub-additive the intervention that removes everything is. Above 1 means the
        # features combine less redundantly than the flow itself does.
        result["rho_ratio"] = rho_a / rho_k
    return result


def leave_one_out(conditions, span, joint_id):
    """In-context marginal contribution of each layer, against its standalone effect."""
    joint_summary = conditions.get(joint_id, {}).get("summary", {})
    joint = joint_summary.get("mean_margin_drop")
    if joint is None:
        return []
    canonical_k = next(iter((joint_summary.get("features_per_layer") or {}).values()), None)

    rows = []
    for layer in span:
        without = conditions.get(f"loo_drop{layer}", {}).get("summary", {}).get("mean_margin_drop")
        _, standalone = single_layer_ablation(conditions, layer, canonical_k)
        if without is None:
            continue
        marginal = float(joint) - float(without)
        rows.append(
            {
                "layer": layer,
                "joint": float(joint),
                "without_layer": float(without),
                "marginal": marginal,
                "standalone": float(standalone) if standalone is not None else None,
                # Much smaller in context than alone is the redundancy signature: the other
                # layers already carry what this one contributes.
                "marginal_over_standalone": (
                    marginal / float(standalone)
                    if standalone not in (None, 0)
                    else None
                ),
            }
        )
    return rows


def significance_report(conditions):
    """Restate significance with the p-floor and the z precision made explicit."""
    rows = []
    for condition_id, payload in conditions.items():
        significance = payload.get("significance", {}).get("mean_margin_drop")
        if not significance:
            continue
        n_sets = len(payload.get("control_summaries", []))
        z = significance.get("z_score")
        controls = [
            c.get("mean_margin_drop") for c in payload.get("control_summaries", [])
        ]
        binding = payload["margin_drops"]

        entry = {
            "condition_id": condition_id,
            "binding": significance.get("binding"),
            "random_mean": significance.get("random_mean"),
            "random_std": significance.get("random_std"),
            "z_score": z,
            "z_standard_error": z_score_standard_error(z, n_sets) if z else None,
            "n_control_sets": n_sets,
            "empirical_p": significance.get("empirical_p_value"),
            "empirical_p_floor": 1.0 / (n_sets + 1) if n_sets else None,
            "control_regime": payload.get("meta", {}).get("control_effective"),
        }
        if binding and controls and all(c is not None for c in controls):
            control_mean = float(np.mean(controls))
            _, p_value = wilcoxon_vs_controls(binding, [control_mean] * len(binding))
            entry["wilcoxon_p"] = p_value
        rows.append(entry)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--span", default="10,11,12,13,14")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    span = [int(x) for x in args.span.split(",")]
    span_tag = f"{span[0]}-{span[-1]}" if span == list(range(span[0], span[-1] + 1)) else ",".join(map(str, span))
    joint_id = f"joint_L{span_tag}"
    span_knockout_id = f"span_knockout_L{span_tag}"

    conditions = load_conditions(args.experiment_dir)
    print(f"Loaded {len(conditions)} conditions from {args.experiment_dir}\n")

    report = {
        "experiment_dir": args.experiment_dir,
        "span": span,
        "n_conditions": len(conditions),
        "redundancy_index": redundancy_index(conditions, joint_id, span_knockout_id, args.seed),
        "single_layer_ratios": {
            str(layer): redundancy_index(
                conditions, f"nested_L{layer}", f"knockout_L{layer}", args.seed
            )
            for layer in span
        },
        "redundancy_by_span": redundancy_by_span(conditions, args.seed),
        "redundancy_trend": redundancy_trend(conditions, args.seed),
        "saturation": nested_curves(conditions),
        "interaction": interaction_index(conditions, span, joint_id, span_knockout_id),
        "leave_one_out": leave_one_out(conditions, span, joint_id),
        "significance": significance_report(conditions),
    }

    out_dir = os.path.join(args.experiment_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "multilayer_summary.json")
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2)

    markdown = render_markdown(report)
    md_path = os.path.join(out_dir, "multilayer_summary.md")
    with open(md_path, "w") as handle:
        handle.write(markdown)

    print(markdown)
    print(f"\nWrote {json_path}\n      {md_path}")



def render_redundancy_by_span(report):
    """The span-by-span table plus the trend test, which is what the constancy claim rests on."""
    rows = report.get("redundancy_by_span") or []
    trend = report.get("redundancy_trend") or {}
    if not rows:
        return []

    lines = ["## Redundancy by span", ""]
    lines.append("| kind | span | size | A | K | R | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        if row.get("status") == "ok":
            lines.append(
                f"| {row['kind']} | `{row['label']}` | {row['span_size']} | "
                f"{row['ablation_mean']:.4f} | {row['knockout_mean']:.4f} | "
                f"{row['ratio']:.1%} | {row['ci_low']:.1%} - {row['ci_high']:.1%} |"
            )
        else:
            lines.append(
                f"| {row['kind']} | `{row['label']}` | {row['span_size']} | — | — | "
                f"undefined | {row.get('status')} |"
            )
    lines.append("")

    if trend.get("status") == "ok":
        lines.append(trend["interpretation"])
        lines.append("")
        lines.append(
            f"Observed spread across the nested spans is {trend['spread']:.1%} "
            f"(95% CI {trend['spread_ci_low']:.1%} - {trend['spread_ci_high']:.1%}); "
            f"highest lower bound {trend['highest_ci_low']:.1%} against lowest upper bound "
            f"{trend['lowest_ci_high']:.1%}. "
            f"{trend['n_bootstrap']} draws, seed {trend['seed']}, n = {trend['n_samples']}."
        )
    else:
        lines.append(f"Trend test not available: `{trend.get('status')}`.")
    lines.append("")
    return lines


def render_markdown(report):
    lines = ["# Multi-layer ablation summary", ""]
    span = report["span"]
    lines.append(f"Span: {span}. {report['n_conditions']} conditions.")
    lines.append("")

    lines.append("## Redundancy index R(S) = A(S) / K(S)")
    lines.append("")
    index = report["redundancy_index"]
    if index.get("status") == "ok":
        lines.append(
            f"**R(S) = {index['ratio']:.3f}** "
            f"(95% CI {index['ci_low']:.3f} - {index['ci_high']:.3f}), "
            f"from A = {index['ablation_mean']:.4f} and K = {index['knockout_mean']:.4f} "
            f"over n = {index['n_samples']}."
        )
    else:
        lines.append(f"Not available: `{index.get('status')}`.")
    lines.append("")

    lines.append("Single layers, for comparison (same samples for numerator and denominator):")
    lines.append("")
    lines.append("| layer | ablation | knockout | ratio | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for layer, entry in report["single_layer_ratios"].items():
        if entry.get("status") == "ok":
            lines.append(
                f"| {layer} | {entry['ablation_mean']:.4f} | {entry['knockout_mean']:.4f} | "
                f"{entry['ratio']:.3f} | {entry['ci_low']:.3f} - {entry['ci_high']:.3f} |"
            )
        else:
            lines.append(f"| {layer} | — | — | undefined | {entry.get('status')} |")
    lines.append("")

    lines.extend(render_redundancy_by_span(report))

    saturation = report["saturation"]
    if saturation.get("b_minus_d") is not None:
        lines.append("## Comparative saturation")
        lines.append("")
        if not saturation.get("degenerate"):
            lines.append(
                f"Ablation b = {saturation['ablation_fit']['b']:.4f}, "
                f"knockout d = {saturation['knockout_fit']['b']:.4f}, "
                f"b - d = {saturation['b_minus_d']:+.4f}."
            )
            lines.append("")
        lines.append(saturation["interpretation"])
        lines.append("")

    interaction = report["interaction"]
    if interaction.get("rho_ablation") is not None:
        lines.append("## Interaction index")
        lines.append("")
        lines.append(
            f"Joint ablation {interaction['joint_ablation']:.4f} against a sum of single "
            f"layers of {interaction['sum_single_ablation']:.4f}: rho_A = "
            f"{interaction['rho_ablation']:.3f}."
        )
        if interaction.get("rho_knockout"):
            lines.append(
                f"The knockout's own sub-additivity over the same span is rho_K = "
                f"{interaction['rho_knockout']:.3f}, so the calibrated ratio is "
                f"**{interaction['rho_ratio']:.3f}**. Raw sub-additivity is expected from "
                f"metric saturation even under independence; the calibrated ratio is the "
                f"part that carries information."
            )
        lines.append("")
    elif interaction.get("ablation_status") == "incomplete":
        lines.append("## Interaction index")
        lines.append("")
        lines.append(
            "Not computed: this run contains no standalone ablation for layer(s) "
            + ", ".join(str(l) for l in interaction["missing_single_layer_ablations"])
            + ". Summing only the layers present would understate the denominator and "
            "inflate rho_A, so it is withheld. Supply the missing single-layer conditions, "
            "or compute rho_A against the published single-layer drops by hand -- the "
            "binding arm does not depend on the control regime, so those remain comparable."
        )
        lines.append("")

    if report["leave_one_out"]:
        lines.append("## Leave-one-out")
        lines.append("")
        lines.append("| layer | joint | without | marginal | standalone | marginal/standalone |")
        lines.append("|---|---|---|---|---|---|")
        for row in report["leave_one_out"]:
            ratio = row["marginal_over_standalone"]
            lines.append(
                f"| {row['layer']} | {row['joint']:.4f} | {row['without_layer']:.4f} | "
                f"{row['marginal']:+.4f} | "
                f"{row['standalone']:.4f} | {ratio:.3f} |"
                if row["standalone"] is not None and ratio is not None
                else f"| {row['layer']} | {row['joint']:.4f} | {row['without_layer']:.4f} | "
                     f"{row['marginal']:+.4f} | — | — |"
            )
        lines.append("")
        lines.append(
            "A marginal contribution far below the standalone effect at every layer is the "
            "strong redundancy signature: the rest of the span already carries what that "
            "layer contributes."
        )
        lines.append("")

    if report["significance"]:
        lines.append("## Significance")
        lines.append("")
        lines.append("| condition | binding | z | z SE | control sets | p floor | Wilcoxon p | control regime |")
        lines.append("|---|---|---|---|---|---|---|---|")
        def cell(value, spec):
            # A single control set gives no spread, so z and its SE are undefined there.
            return "--" if value is None else format(value, spec)

        for row in report["significance"]:
            lines.append(
                f"| {row['condition_id']} | {cell(row['binding'], '.4f')} | "
                f"{cell(row['z_score'], '.0f')} | {cell(row['z_standard_error'], '.0f')} | "
                f"{row['n_control_sets']} | {cell(row['empirical_p_floor'], '.4f')} | "
                f"{cell(row.get('wilcoxon_p'), '.2e')} | {row['control_regime']} |"
            )
        lines.append("")
        lines.append(
            "z is reported to one significant figure: it is estimated from a handful of "
            "control draws and its own standard error is large. The empirical p cannot go "
            "below its floor, so the Wilcoxon test over questions is the one with power."
        )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
