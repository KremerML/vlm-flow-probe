"""Run controls that are disjoint from their binding set by construction.

Replaces the sampled matched-control arm, which is not identified for this
experiment: the selection rule is a top-k cut on a causal score, so no
non-selected feature has a score inside the binding range, and matching on
activation instead fails because the two are nearly monotone in each other.
See ``ablation/rank_band_controls.py`` for the measurements behind that.

Condition families (``--phases``):

  single        per layer: pass-through, the binding set, the adjacent causal
                rank bands, top-k by activation alone, and the activation-matched
                set retained for comparison
  gradient      the causal/activation ranking difference, isolating what the
                gradient factor adds over an activation filter
  doseresponse  disjoint equal-size bands down the ranking, so a decline across
                bands cannot be a feature-count effect
  subsets       random equal-size subsets of two deep pools -- the one place a
                set-level null is estimable
  multi         rank-band controls mirroring multi-layer binding conditions,
                so both arms of a comparison are adjusted the same way

Pass ``--pair_with`` to evaluate on another run's sample cache, which makes
every new condition paired question-for-question with that run's conditions.

Resumable: one fsync'd JSONL line per completed condition.
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

PHASES = ("single", "gradient", "doseresponse", "subsets", "multi")


def log(message="", indent=0):
    print(f"[{datetime.now():%H:%M:%S}] {'  ' * indent}{message}", flush=True)


def duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def load_stats(path):
    with open(path) as handle:
        return {int(k): v for k, v in json.load(handle).items()}


def build_conditions(stats_by_layer, phases, k=200, layers=None):
    """Assemble the requested condition families.

    ``stats_by_layer`` maps layer index to that layer's feature statistics.
    """
    from vlmflowprobe.ablation.multilayer_experiments import (
        KIND_PASSTHROUGH, KIND_SAE, Condition,
    )
    from vlmflowprobe.ablation.rank_band_controls import (
        activation_matched_control, activation_top_control, adjacent_band_control,
        dose_response_bands, random_subsets, rank_band, ranking_difference,
    )

    layers = sorted(layers or stats_by_layer)
    conditions = []

    if "single" in phases:
        for layer in layers:
            stats = stats_by_layer[layer]
            binding = rank_band(stats, 0, k)
            conditions += [
                Condition(f"ctl_passthrough_L{layer}", KIND_PASSTHROUGH,
                          features={layer: []},
                          label=f"L{layer} pass-through: SAE reconstruction floor"),
                Condition(f"bind_causal_1_{k}_L{layer}", KIND_SAE,
                          features={layer: binding},
                          label=f"L{layer} binding: causal ranks 1-{k}"),
                Condition(f"ctl_band_{k}_{2*k}_L{layer}", KIND_SAE,
                          features={layer: adjacent_band_control(stats, k)},
                          label=f"L{layer} control: causal ranks {k+1}-{2*k}"),
                Condition(f"ctl_band_{2*k}_{3*k}_L{layer}", KIND_SAE,
                          features={layer: rank_band(stats, 2 * k, 3 * k)},
                          label=f"L{layer} control: causal ranks {2*k+1}-{3*k}"),
                Condition(f"ctl_acttop_{k}_L{layer}", KIND_SAE,
                          features={layer: activation_top_control(stats, k)},
                          label=f"L{layer} control: top-{k} by activation alone"),
                Condition(f"ctl_actmatched_{k}_L{layer}", KIND_SAE,
                          features={layer: activation_matched_control(stats, binding)},
                          label=f"L{layer} control: activation-matched to binding"),
            ]

    if "gradient" in phases:
        for layer in layers:
            parts = ranking_difference(stats_by_layer[layer], k)
            for part, features in parts.items():
                conditions.append(
                    Condition(f"grad_{part}_L{layer}", KIND_SAE,
                              features={layer: features},
                              label=f"L{layer}: {len(features)} features, {part}")
                )

    if "doseresponse" in phases:
        for layer in layers:
            for b in dose_response_bands(stats_by_layer[layer], width=40, n_bands=10):
                conditions.append(
                    Condition(f"dose_L{layer}_r{b['lo']}_{b['hi']}", KIND_SAE,
                              features={layer: b["features"]},
                              label=f"L{layer} causal ranks {b['lo']+1}-{b['hi']}")
                )

    if "subsets" in phases:
        for layer in layers:
            stats = stats_by_layer[layer]
            for name, pool in (("top200", rank_band(stats, 0, 200)),
                               ("tail", rank_band(stats, 200, 1000))):
                for i, subset in enumerate(random_subsets(pool, 40, 12, seed=42)):
                    conditions.append(
                        Condition(f"subset_{name}_L{layer}_{i:02d}", KIND_SAE,
                                  features={layer: subset},
                                  label=f"L{layer}: random 40 of {name}")
                    )

    if "multi" in phases:
        # Mirror each multi-layer binding condition one band down the ranking.
        spans = {
            "spread40x5": {l: rank_band(stats_by_layer[l], 40, 80) for l in layers},
        }
        for l in layers:
            spans[f"single_L{l}"] = {l: adjacent_band_control(stats_by_layer[l], k)}
        spans["joint"] = {l: rank_band(stats_by_layer[l], k, 2 * k) for l in layers}
        for name, features in spans.items():
            conditions.append(
                Condition(f"ctl_{name}", KIND_SAE, features=features,
                          label=f"rank-band control mirroring {name}")
            )

    return conditions


def completed(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["condition_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phases", default="single",
                        help=f"comma-separated from {list(PHASES)}, or 'all'")
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--k", type=int, default=200, help="binding set size")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--experiment_name", default="controls")
    parser.add_argument("--pair_with", default=None,
                        help="sample_cache.json to evaluate against, for exact pairing")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    from vlmflowprobe.ablation.multilayer_ablator import MultiLayerFeatureAblator
    from vlmflowprobe.ablation.multilayer_experiments import MultiLayerAblationExperiment
    from vlmflowprobe.ablation.sample_cache import (
        build_sample_cache, load_sample_cache, save_sample_cache,
    )
    from vlmflowprobe.core.config import load_config
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.utils.provenance import write_provenance
    from vlmflowprobe.utils.runtime import load_adapter, load_sae, setup_experiment

    config = load_config(args.config, overrides=args.override)
    phases = list(PHASES) if args.phases == "all" else [
        p.strip() for p in args.phases.split(",") if p.strip()
    ]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        parser.error(f"unknown phase(s) {unknown}; choose from {list(PHASES)}")

    ml_cfg = config.get("multilayer", {})
    stats_by_layer = {
        int(layer): load_stats(path)
        for layer, path in (ml_cfg.get("stats_paths") or {}).items()
        if os.path.exists(path)
    }
    if not stats_by_layer:
        sys.exit("no feature statistics found; set multilayer.stats_paths")

    conditions = build_conditions(stats_by_layer, phases, k=args.k)
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in conditions if c.condition_id in wanted]

    if args.dry_run:
        log(f"{len(conditions)} conditions in phases {phases}")
        for condition in conditions:
            counts = ", ".join(f"L{l}:{len(condition.features[l])}"
                               for l in sorted(condition.features))
            log(f"{condition.condition_id:<34} [{counts}]  {condition.label}", indent=1)
        log(f"estimated {duration(len(conditions) * 85)} at 85s per condition")
        return

    experiment_dir, seed = setup_experiment(args, config)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_path = os.path.join(experiment_dir, "checkpoint.jsonl")
    done = set() if args.force else completed(checkpoint_path)
    pending = [c for c in conditions if c.condition_id not in done]
    log(f"{len(conditions)} conditions, {len(pending)} to run -> {experiment_dir}")
    if not pending:
        log("nothing to do")
        return

    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer, split="val")
    saes = {
        int(layer): load_sae(config, adapter, path)
        for layer, path in (ml_cfg.get("sae_paths") or {}).items()
    }
    experiment = MultiLayerAblationExperiment(adapter, saes, {}, stats_by_layer, config)
    write_provenance(experiment_dir, config=config.to_dict(), seed=seed, argv=sys.argv)

    cache_path = args.pair_with or os.path.join(experiment_dir, "sample_cache.json")
    if os.path.exists(cache_path):
        records = load_sample_cache(cache_path)
        log(f"paired against {len(records)} samples from {cache_path}")
    else:
        log("building sample cache")
        records = build_sample_cache(
            experiment.ablator, dataset, position_type=experiment.position_type,
            logprob_normalize=experiment.logprob_normalize,
            max_samples=args.max_samples, show_progress=not args.no_progress,
        )
        save_sample_cache(records, cache_path)

    started = time.time()
    for idx, condition in enumerate(pending, 1):
        elapsed = time.time() - started
        eta = (elapsed / (idx - 1) * (len(pending) - idx + 1)) if idx > 1 else None
        log(f"[{idx}/{len(pending)}] {condition.condition_id}"
            + (f"  (ETA {duration(eta)})" if eta else ""))
        rows, summary = experiment.run_condition(
            dataset, condition, records,
            max_samples=args.max_samples, show_progress=not args.no_progress,
        )
        target = os.path.join(experiment_dir, "conditions", condition.condition_id)
        os.makedirs(target, exist_ok=True)
        from vlmflowprobe.cli.distill_results import distill_condition_samples

        payload = {
            "condition_id": condition.condition_id,
            "label": condition.label,
            "summary": summary,
            "control_summaries": [],
            "meta": {
                "control_design": "disjoint_rank_band",
                "layers": condition.layers,
                "mode": condition.mode,
                "n_samples": len(rows),
                "seed": seed,
                "paired_sample_cache": cache_path,
            },
            "feature_ids": {str(l): condition.features[l] for l in sorted(condition.features)},
        }
        with open(os.path.join(target, "results.json"), "w") as handle:
            json.dump({"per_sample": rows, **payload}, handle, indent=2)
        with open(os.path.join(target, "summary.json"), "w") as handle:
            json.dump({**payload, "per_sample_distilled": distill_condition_samples(rows)},
                      handle, indent=1)
        with open(checkpoint_path, "a") as handle:
            handle.write(json.dumps({
                "condition_id": condition.condition_id,
                "mean_margin_drop": summary.get("mean_margin_drop"),
                "mean_relative_perturbation": summary.get("mean_relative_perturbation"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        log(f"drop {summary.get('mean_margin_drop'):+.4f}  "
            f"perturbation {summary.get('mean_relative_perturbation'):.5f}", indent=1)

    log(f"done: {len(pending)} conditions in {duration(time.time() - started)}")


if __name__ == "__main__":
    main()
