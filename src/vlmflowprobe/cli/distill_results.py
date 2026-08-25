"""Distill large result JSONs into small committed summaries.

The pipeline writes several files that are far too large to keep in git or to load
into an LLM context: per-feature dictionaries over all 32768 SAE features, and
ablation results carrying one record per evaluated sample. This tool reduces each
to a `*.summary.json` sibling that preserves everything the writeup needs — the
top-scoring features, the distribution the scores came from, and the aggregate
statistics — while dropping the long tail.

The originals stay on disk (gitignored); the summaries are what gets committed.

    $PY sae_experiments/tools/distill_results.py --root output --dry_run
    $PY sae_experiments/tools/distill_results.py --root output

Stdlib only, so it runs without the LLaVA venv.
"""

import argparse
import hashlib
import json
import math
import os
import sys

# Per-sample lists longer than this are replaced by summary statistics.
SAMPLE_LIST_THRESHOLD = 20

# Rounding applied to the per-sample margin drops kept verbatim in condition summaries.
# The margins themselves are log-probabilities of order 1, so six decimals is far below
# any difference the analysis can resolve.
CONDITION_DROP_DECIMALS = 6

# How many top features each distilled catalog retains.
DEFAULT_TOP_K = 500

# Files below this stay tracked as-is — small enough that the repo can carry them
# at full fidelity. Sources above it are distilled, and any summary that would not
# actually be smaller than its source is skipped.
DEFAULT_MIN_BYTES = 128 * 1024


def _percentiles(values, points=(50, 90, 99, 99.9)):
    """Nearest-rank percentiles of an already-sorted ascending list."""
    if not values:
        return {}
    out = {}
    n = len(values)
    for p in points:
        idx = min(n - 1, max(0, int(math.ceil(p / 100.0 * n)) - 1))
        out[f"p{p:g}"] = values[idx]
    return out


def _describe(values):
    """Mean/std/min/max/percentiles for a list of numbers."""
    vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    ordered = sorted(vals)
    desc = {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": ordered[0],
        "max": ordered[-1],
        "frac_positive": sum(1 for v in vals if v > 0) / n,
    }
    desc.update(_percentiles(ordered))
    return desc


def distill_feature_dict(data, score_key, top_k, extra_keys=()):
    """Reduce a {feature_index: {metric: value}} dict to top-k plus distribution.

    Also records the activation magnitude of the retained features against the
    global distribution — the statistic that distinguishes v2's causal features
    from the v1 "ghost features" that drove the null results.
    """
    items = []
    for fid, rec in data.items():
        if not isinstance(rec, dict):
            continue
        score = rec.get(score_key)
        if isinstance(score, (int, float)):
            items.append((abs(score), fid, rec))
    items.sort(key=lambda t: t[0], reverse=True)

    scores = sorted(abs(s) for s, _, _ in items)
    total_mass = sum(scores)
    top = items[:top_k]

    summary = {
        "n_features_total": len(data),
        "n_features_scored": len(items),
        "score_key": score_key,
        "top_k_retained": len(top),
        "score_distribution": _describe(scores),
        "score_mass_total": total_mass,
        "n_nonzero": sum(1 for s in scores if s > 0),
    }

    for k in (50, 200, 500):
        if k <= len(items):
            captured = sum(s for s, _, _ in items[:k])
            summary[f"score_mass_frac_top{k}"] = (
                captured / total_mass if total_mass else 0.0
            )

    for key in extra_keys:
        all_vals = [
            r.get(key) for _, _, r in items if isinstance(r.get(key), (int, float))
        ]
        top_vals = [
            r.get(key) for _, _, r in top if isinstance(r.get(key), (int, float))
        ]
        if all_vals:
            summary[f"{key}_all"] = _describe([abs(v) for v in all_vals])
        if top_vals:
            summary[f"{key}_top_k"] = _describe([abs(v) for v in top_vals])

    summary["top_features"] = [
        dict(feature=int(fid) if str(fid).isdigit() else fid, **rec) for _, fid, rec in top
    ]
    return summary


def distill_sample_lists(node):
    """Recursively replace long per-sample lists with per-field summary statistics."""
    if isinstance(node, dict):
        return {k: distill_sample_lists(v) for k, v in node.items()}

    if isinstance(node, list):
        if len(node) > SAMPLE_LIST_THRESHOLD and all(
            isinstance(x, dict) for x in node
        ):
            fields = {}
            # Sorted, not set order: string hashing is salted per process, so an
            # unsorted set makes every re-run rewrite these files with the same
            # numbers in a different order and show up as a diff.
            for key in sorted({k for rec in node for k in rec}):
                desc = _describe([rec.get(key) for rec in node])
                if desc is not None:
                    fields[key] = desc
            return {
                "_distilled": True,
                "_original_length": len(node),
                "field_summaries": fields,
            }
        if len(node) > SAMPLE_LIST_THRESHOLD and all(
            isinstance(x, (int, float)) for x in node
        ):
            return {
                "_distilled": True,
                "_original_length": len(node),
                "summary": _describe(node),
            }
        return [distill_sample_lists(x) for x in node]

    return node


def derive_ablation_fields(data):
    """Add the per-sample quantities the writeup needs but the runner never stored.

    `margin_drop` is the headline metric yet only its two operands are recorded per
    sample, and prediction flips are not counted anywhere. Both are recoverable, but
    only from the raw lists — so they must be computed before those lists are
    collapsed into summary statistics.
    """
    for key in ("binding_results", "random_results"):
        records = data.get(key)
        if not isinstance(records, list):
            continue
        flips = heals = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            base, abl = rec.get("baseline_margin"), rec.get("ablated_margin")
            if isinstance(base, (int, float)) and isinstance(abl, (int, float)):
                rec["margin_drop"] = base - abl
            answer = rec.get("answer")
            bp, ap = rec.get("baseline_pred"), rec.get("ablated_pred")
            if answer is not None and bp is not None and ap is not None:
                if bp == answer and ap != answer:
                    flips += 1
                elif bp != answer and ap == answer:
                    heals += 1
        data[f"{key}_prediction_changes"] = {
            "n": len(records),
            "correct_to_wrong": flips,
            "wrong_to_correct": heals,
        }
    return data


def distill_condition_samples(rows):
    """Reduce one multi-layer condition's per-sample records to what survives in git.

    `analyze_multilayer_ablation.load_conditions` reads exactly one quantity out of
    `per_sample` — `baseline_margin - ablated_margin` per question — because the control
    arm is already aggregated into `control_summaries`. So the vector of drops, plus the
    prediction flips that nothing else records, is the whole of what the raw file is for.

    The drops are kept verbatim rather than described, because the analysis *pairs* them
    across conditions (the redundancy index R = A/K takes a paired bootstrap CI over
    ablation and knockout measured on the same questions), and a pairing cannot be
    rebuilt from means. Rows keep their order, which is `sample_cache.json` order and
    identical across every condition; `question_ids_sha1` is the guard on that assumption.
    At 256 samples the whole 47-condition matrix costs ~150 KB against 16 MB of raw records.
    """
    drops = []
    ids = []
    flips = heals = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        ids.append(str(row.get("question_id")))
        base, abl = row.get("baseline_margin"), row.get("ablated_margin")
        if isinstance(base, (int, float)) and isinstance(abl, (int, float)):
            drops.append(round(base - abl, CONDITION_DROP_DECIMALS))
        answer = row.get("answer")
        bp, ap = row.get("baseline_pred"), row.get("ablated_pred")
        if answer is not None and bp is not None and ap is not None:
            if bp == answer and ap != answer:
                flips += 1
            elif bp != answer and ap == answer:
                heals += 1

    return {
        "n_samples": len(rows),
        "sample_order": "sample_cache.json",
        "question_ids_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest(),
        "margin_drops": drops,
        "margin_drop_distribution": _describe(drops),
        "prediction_changes": {
            "n": len(rows),
            "correct_to_wrong": flips,
            "wrong_to_correct": heals,
        },
    }


def is_condition_result(path):
    """True for `<experiment>/conditions/<condition_id>/results.json`.

    Matched by shape rather than by name: `results.json` is generic enough that a
    filename test alone would sweep up unrelated files elsewhere in the tree.
    """
    parts = os.path.normpath(path).split(os.sep)
    return (
        len(parts) >= 3
        and parts[-1] == "results.json"
        and parts[-3] == "conditions"
    )


def distill_condition_result(path, data):
    """Fold the per-sample block into the condition's existing `summary.json`.

    Unlike every other source here, this one already has a committed sibling — the runner
    writes `summary.json` as `results.json` minus `per_sample`. So the distillation merges
    into that file instead of minting a `results.summary.json` nobody would read.
    """
    out_path = os.path.join(os.path.dirname(path), "summary.json")
    if os.path.exists(out_path):
        with open(out_path) as fh:
            summary = json.load(fh)
    else:
        summary = {k: v for k, v in data.items() if k != "per_sample"}
    summary["per_sample_distilled"] = distill_condition_samples(data.get("per_sample", []))
    return summary, out_path


def distill_file(path, top_k):
    """Dispatch on filename. Returns (summary_dict, output_path) or None to skip."""
    base = os.path.basename(path)
    with open(path) as fh:
        data = json.load(fh)

    stem = path[: -len(".json")] if path.endswith(".json") else path
    out_path = stem + ".summary.json"

    if is_condition_result(path):
        summary, out_path = distill_condition_result(path, data)
    elif base == "causal_feature_stats.json":
        summary = distill_feature_dict(
            data,
            score_key="causal_score",
            top_k=top_k,
            extra_keys=("activation_mean", "gradient_mean"),
        )
    elif base == "feature_stats.json":
        summary = distill_feature_dict(
            data,
            score_key="ratio",
            top_k=top_k,
            extra_keys=("correct_mean", "incorrect_mean", "diff"),
        )
        by_diff = distill_feature_dict(data, score_key="diff", top_k=top_k)
        summary["top_features_by_diff"] = by_diff["top_features"]
        summary["diff_distribution"] = by_diff["score_distribution"]
    elif "ablation" in base and base.endswith(".json"):
        summary = distill_sample_lists(derive_ablation_fields(data))
    else:
        return None

    summary["_source"] = os.path.relpath(path)
    summary["_source_bytes"] = os.path.getsize(path)
    return summary, out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output", help="Directory tree to scan")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min_bytes", type=int, default=DEFAULT_MIN_BYTES,
                        help="Skip sources smaller than this (they stay tracked as-is)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be written without writing")
    args = parser.parse_args()

    targets = []
    for dirpath, _, filenames in os.walk(args.root):
        for name in filenames:
            if not name.endswith(".json") or name.endswith(".summary.json"):
                continue
            path = os.path.join(dirpath, name)
            if name in ("causal_feature_stats.json", "feature_stats.json") or (
                "ablation" in name and "results" in name
            ) or is_condition_result(path):
                if os.path.getsize(path) >= args.min_bytes:
                    targets.append(path)

    targets.sort()
    total_in = total_out = 0
    written = skipped = 0

    for path in targets:
        try:
            result = distill_file(path, args.top_k)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  SKIP {path}: {exc}", file=sys.stderr)
            continue
        if result is None:
            continue
        summary, out_path = result
        blob = json.dumps(summary, indent=1)
        in_sz = os.path.getsize(path)
        out_sz = len(blob.encode())

        if out_sz >= in_sz:
            print(f"  SKIP (no gain) {path}")
            skipped += 1
            continue

        total_in += in_sz
        total_out += out_sz
        pct = 100.0 * out_sz / in_sz
        print(f"  {in_sz/1048576:7.2f} MB -> {out_sz/1048576:6.3f} MB ({pct:5.1f}%)  {out_path}")
        if not args.dry_run:
            with open(out_path, "w") as fh:
                fh.write(blob)
            written += 1

    print(
        f"\n{len(targets) - skipped} distilled: {total_in/1048576:.1f} MB -> "
        f"{total_out/1048576:.1f} MB ({100.0*total_out/total_in if total_in else 0:.1f}%), "
        f"{written} written, {skipped} skipped"
    )


if __name__ == "__main__":
    main()
