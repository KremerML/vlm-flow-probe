"""Run the multi-layer ablation condition matrix.

Tests whether the 30-50% shortfall between single-layer SAE ablation and the attention-
knockout ceiling is explained by cross-layer redundancy -- whether ablating one layer leaves
the model free to re-read the image at layers L+1..31.

Lives in tools/ rather than pipeline/NN_ because it reads many experiment directories at
once (like tools/knockout_sae_pipeline.py) and because being importable makes the condition
builders testable.

    PY=LLaVA-NeXT/.venv/bin/python
    CFG=configs/clevr_lite/multilayer_l10-14_attn_out_question.yaml

    # Review the matrix before spending any GPU time
    $PY sae_experiments/tools/run_multilayer_ablation.py --config $CFG --dry_run

    # The gate first. A0 must reproduce the published single-layer layer-11 number
    # (0.2131) and gate_none must be exactly 0.0, or the harness is not trustworthy.
    $PY sae_experiments/tools/run_multilayer_ablation.py --config $CFG --phases gate

    # Then the rest
    $PY sae_experiments/tools/run_multilayer_ablation.py \
        --config $CFG --phases primary,nested,leave_one_out

Resumable: one fsync'd JSONL line per completed condition, and completed ids are skipped on
restart. Conditions take ~90s, so per-condition granularity is enough.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta

import vlmflowprobe.cli._bootstrap  # noqa: F401

from dataclasses import replace as _dc_replace

from vlmflowprobe.ablation.multilayer_experiments import (  # noqa: E402
    KIND_COMBINED,
    KIND_KNOCKOUT,
    KIND_NONE,
    KIND_PASSTHROUGH,
    KIND_SAE,
    Condition,
    MultiLayerAblationExperiment,
    _compact_layer_range,
)
from vlmflowprobe.ablation.sample_cache import (  # noqa: E402
    build_sample_cache,
    load_sample_cache,
    save_sample_cache,
)
from vlmflowprobe.core.config import load_config  # noqa: E402
from vlmflowprobe.core.sparse_autoencoder import (  # noqa: E402
    build_sae_from_state,
    sae_state_from_checkpoint,
)
from vlmflowprobe.cli.distill_results import distill_condition_samples  # noqa: E402
from vlmflowprobe.data.datasets import build_dataset  # noqa: E402
from vlmflowprobe.utils.config_utils import resolve_dtype  # noqa: E402
from vlmflowprobe.utils.provenance import write_provenance  # noqa: E402
from vlmflowprobe.utils.runtime import load_adapter, setup_experiment  # noqa: E402

PHASES = (
    "gate",
    "primary",
    "nested",
    "non_nested",
    "leave_one_out",
    "budget",
    "downstream",
    "sensitivity",
)


def log(message="", indent=0):
    """Timestamped, explicitly flushed line.

    The shell script pipes stdout through tee, so Python block-buffers it and nothing
    appears for minutes at a time. tqdm writes to stderr and is unaffected, which is why
    the bars showed up while the messages did not. Every log line flushes.
    """
    stamp = datetime.now().strftime("%H:%M:%S")
    prefix = " " * indent
    print(f"[{stamp}] {prefix}{message}", flush=True)


def human_duration(seconds):
    if seconds is None or seconds != seconds:  # None or NaN
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def gpu_memory_note():
    """Peak allocated / reserved, so a run creeping toward the 24 GB ceiling is visible."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        return f" | gpu peak {allocated:.1f}G alloc / {reserved:.1f}G reserved"
    except Exception:
        return ""


# --------------------------------------------------------------------------- conditions


def build_conditions(experiment, config, phases):
    """Enumerate every condition the requested phases call for.

    Returns them in dependency order: the gate first, because nothing downstream means
    anything until A0 reproduces the published single-layer number.
    """
    ml_cfg = config.get("multilayer", {})
    cond_cfg = config.get("conditions", {})
    span = [int(l) for l in ml_cfg.get("layers", [])]
    k = ml_cfg.get("features_per_layer", 200)
    mode = config.get("ablation", {}).get("mode", "replace")
    flow = (config.get("knockout", {}).get("flows") or ["Image->Question"])[0]

    conditions = []

    if "gate" in phases:
        # A0 is the regression test for the whole feature_ablator refactor: one layer,
        # top-200, through the multi-layer harness, must reproduce 0.2131.
        conditions.append(
            Condition(
                condition_id="A0_regression_L11",
                kind=KIND_SAE,
                features=experiment.features_for([11], 200),
                mode=mode,
                label="harness regression vs published layer-11 result",
            )
        )
        conditions.append(
            Condition(
                condition_id="gate_none",
                kind=KIND_NONE,
                features={},
                mode=mode,
                label="no intervention; must be exactly 0.0",
            )
        )
        # Pass-through over each nested span: replace mode swaps the hidden state for the
        # SAE reconstruction, so stacking N of them stacks N layers of reconstruction error.
        # A layer mapped to [] is hooked but ablates nothing, which is what measures it.
        for nested in cond_cfg.get("nested", []):
            layers = [int(l) for l in nested]
            conditions.append(
                Condition(
                    condition_id=f"gate_passthrough_L{_compact_layer_range(layers)}",
                    kind=KIND_PASSTHROUGH,
                    features={layer: [] for layer in layers},
                    mode=mode,
                    label="reconstruction error only, no features ablated",
                )
            )
        conditions.append(
            Condition(
                condition_id=f"gate_passthrough_delta_L{_compact_layer_range(span)}",
                kind=KIND_PASSTHROUGH,
                features={layer: [] for layer in span},
                mode="residual",
                label="delta mode passthrough; ~0 by construction, a harness check",
            )
        )

    if "primary" in phases:
        conditions.append(
            Condition(
                condition_id=f"joint_L{_compact_layer_range(span)}",
                kind=KIND_SAE,
                features=experiment.features_for(span, k),
                mode=mode,
                label="the headline joint ablation",
            )
        )
        conditions.append(
            Condition(
                condition_id=f"span_knockout_L{_compact_layer_range(span)}",
                kind=KIND_KNOCKOUT,
                features={},
                knockout_layers=tuple(span),
                flow=flow,
                label="the ceiling, measured on the ablation's own samples",
            )
        )
        # Single-layer knockouts on the same samples, so every "% of ceiling" has a
        # denominator from the same sample set as its numerator. The published ratios
        # divided an n=256 unfiltered ablation by an n=7084 filtered knockout.
        for layer in span:
            conditions.append(
                Condition(
                    condition_id=f"knockout_L{layer}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=(layer,),
                    flow=flow,
                    label="single-layer ceiling on the shared sample set",
                )
            )

    if "nested" in phases:
        for nested in cond_cfg.get("nested", []):
            layers = [int(l) for l in nested]
            if layers == span:
                continue  # already covered by the primary joint condition
            tag = _compact_layer_range(layers)
            conditions.append(
                Condition(
                    condition_id=f"nested_L{tag}",
                    kind=KIND_SAE,
                    features=experiment.features_for(layers, k),
                    mode=mode,
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"nested_knockout_L{tag}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=tuple(layers),
                    flow=flow,
                    label="knockout saturation curve, the null the ablation curve is read against",
                )
            )

    if "non_nested" in phases:
        for group in cond_cfg.get("non_nested", []):
            layers = [int(l) for l in group]
            tag = _compact_layer_range(layers)
            conditions.append(
                Condition(
                    condition_id=f"nonnested_L{tag}",
                    kind=KIND_SAE,
                    features=experiment.features_for(layers, k),
                    mode=mode,
                    label="separates span size from span depth",
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"nonnested_knockout_L{tag}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=tuple(layers),
                    flow=flow,
                )
            )

    if "leave_one_out" in phases and cond_cfg.get("leave_one_out", True):
        for dropped in span:
            remaining = [l for l in span if l != dropped]
            conditions.append(
                Condition(
                    condition_id=f"loo_drop{dropped}",
                    kind=KIND_SAE,
                    features=experiment.features_for(remaining, k),
                    mode=mode,
                    label=f"in-context marginal contribution of layer {dropped}",
                )
            )

    if "budget" in phases:
        budget = cond_cfg.get("budget_matched", {})
        spread = int(budget.get("spread_per_layer", 40))
        conditions.append(
            Condition(
                condition_id=f"budget_spread{spread}x{len(span)}",
                kind=KIND_SAE,
                features=experiment.features_for(span, spread),
                mode=mode,
                label=f"{spread * len(span)} features spread across {len(span)} layers",
            )
        )
        # The concentrated arm is a curve, not a point: a single k=200 comparison would
        # confound feature rank and count saturation with the layer distribution.
        concentrated_layer = int(budget.get("concentrated_layer", 11))
        for count in budget.get("concentrated_k", []):
            conditions.append(
                Condition(
                    condition_id=f"budget_concentrated_L{concentrated_layer}_k{count}",
                    kind=KIND_SAE,
                    features=experiment.features_for([concentrated_layer], int(count)),
                    mode=mode,
                    label="concentrated count curve at equal and unequal budgets",
                )
            )

    if "downstream" in phases:
        downstream = cond_cfg.get("downstream_knockout", {})
        last_layer = int(downstream.get("downstream_to", 31))
        for anchor in downstream.get("anchor_layers", []):
            anchor = int(anchor)
            tail = tuple(range(anchor + 1, last_layer + 1))
            features = experiment.features_for([anchor], k)
            conditions.append(
                Condition(
                    condition_id=f"downstream_ablate_L{anchor}",
                    kind=KIND_SAE,
                    features=features,
                    mode=mode,
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"downstream_knockout_L{anchor + 1}-{last_layer}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=tail,
                    flow=flow,
                    label="blocks re-reading without ablating anything",
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"downstream_combined_L{anchor}",
                    kind=KIND_COMBINED,
                    features=features,
                    knockout_layers=tail,
                    flow=flow,
                    mode=mode,
                    label="re-reading blocked AND features removed",
                )
            )

    if "sensitivity" in phases:
        sens = [int(l) for l in cond_cfg.get("sensitivity_span", [])]
        if sens:
            tag = _compact_layer_range(sens)
            conditions.append(
                Condition(
                    condition_id=f"sensitivity_joint_L{tag}",
                    kind=KIND_SAE,
                    features=experiment.features_for(sens, k),
                    mode=mode,
                    label="span without the sign-inconsistent layer 13",
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"sensitivity_knockout_L{tag}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=tuple(sens),
                    flow=flow,
                )
            )
        # Isolates the last layer's ablation contribution from its reconstruction error:
        # ablate 10..13 while layer 14 carries a pass-through hook.
        if len(span) > 1:
            head, tail_layer = span[:-1], span[-1]
            features = experiment.features_for(head, k)
            features[tail_layer] = []
            conditions.append(
                Condition(
                    condition_id=f"sensitivity_passthrough_tail_L{tail_layer}",
                    kind=KIND_SAE,
                    features=features,
                    mode=mode,
                    label=f"layers {_compact_layer_range(head)} ablated, layer {tail_layer} pass-through",
                )
            )

    return conditions


def assign_control_kinds(conditions):
    """Stamp each condition's joint-control policy onto Condition.control_kind.

    Only conditions that carry a significance claim need controls, and only the
    primary ones need the full n_random_sets. This is the single place the
    condition-id families map to a control policy; counting itself lives on
    ``Condition.n_controls``.
    """
    stamped = []
    for condition in conditions:
        if condition.condition_id.startswith(("joint_", "A0_")):
            control_kind = "full"
        elif condition.condition_id.startswith(("nested_L", "nonnested_L", "budget_spread")):
            control_kind = "single"
        else:
            control_kind = "none"
        stamped.append(_dc_replace(condition, control_kind=control_kind))
    return stamped


def controls_for(condition, config):
    return condition.n_controls(int(config.get("random_control", {}).get("n_random_sets", 15)))


# --------------------------------------------------------------------------- loading


def load_multilayer_saes(config, adapter):
    ml_cfg = config.get("multilayer", {})
    dtype = resolve_dtype(ml_cfg.get("sae_dtype", "float32"))
    device = adapter.device if adapter is not None else "cpu"

    site = config.get("model", {}).get("activation_site", "attn_out")
    saes = {}
    for layer in ml_cfg.get("layers", []):
        layer = int(layer)
        path = ml_cfg["sae_paths"][layer]
        sae_state = sae_state_from_checkpoint(torch_load(path))
        # Take dimensions (and the architecture) from the checkpoint rather than the
        # config: with one config covering several layers there is no single
        # n_features to read.
        n_features, d_in = sae_state["encoder.weight"].shape
        if adapter is not None and int(d_in) != int(adapter.site_dim(layer, site)):
            raise ValueError(
                f"layer {layer}: SAE {path} has d_in={d_in} but site {site!r} is "
                f"{adapter.site_dim(layer, site)} wide"
            )
        sae = build_sae_from_state(sae_state, l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3))
        sae.to(device=device, dtype=dtype)
        sae.eval()
        saes[layer] = sae
        print(f"[multilayer] layer {layer}: {type(sae).__name__} {n_features}x{d_in} from {path}")
    return saes


def torch_load(path):
    import torch

    return torch.load(path, map_location="cpu")


def load_catalogs_and_stats(config):
    ml_cfg = config.get("multilayer", {})
    catalogs, stats = {}, {}
    for layer in ml_cfg.get("layers", []):
        layer = int(layer)
        with open(ml_cfg["catalog_paths"][layer]) as handle:
            catalog = json.load(handle)
        # Catalogs are written in descending causal-score order; preserve it.
        catalogs[layer] = [int(k) for k in catalog.keys()]

        stats_path = ml_cfg.get("stats_paths", {}).get(layer)
        if stats_path and os.path.exists(stats_path):
            with open(stats_path) as handle:
                stats[layer] = {int(k): v for k, v in json.load(handle).items()}
        else:
            stats[layer] = {}
        print(
            f"[multilayer] layer {layer}: {len(catalogs[layer])} catalog features, "
            f"{len(stats[layer])} scored features"
        )
    return catalogs, stats


# --------------------------------------------------------------------------- checkpoint


def completed_conditions(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["condition_id"])
    return done


def append_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phases",
        default="gate",
        help=f"comma-separated subset of {','.join(PHASES)}, or 'all'",
    )
    parser.add_argument("--conditions", default=None, help="comma-separated condition ids")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument(
        "--mode",
        default=None,
        help="override ablation.mode, e.g. 'residual' if the pass-through gate trips",
    )
    parser.add_argument("--force", action="store_true", help="re-run completed conditions")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--override", action="append", default=[],
                        help="config override, section.key=value (repeatable)")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print the condition matrix and exit without loading the model",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    if args.mode:
        config.data["ablation"] = dict(config.get("ablation", {}), mode=args.mode)

    phases = list(PHASES) if args.phases == "all" else [
        p.strip() for p in args.phases.split(",") if p.strip()
    ]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        parser.error(f"unknown phase(s) {unknown}; choose from {list(PHASES)}")

    if args.dry_run:
        return dry_run(config, phases, args)

    experiment_dir, seed = setup_experiment(args, config)
    show_progress = not args.no_progress

    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer)

    saes = load_multilayer_saes(config, adapter)
    catalogs, stats = load_catalogs_and_stats(config)
    experiment = MultiLayerAblationExperiment(adapter, saes, catalogs, stats, config)

    write_provenance(experiment_dir, config=config.to_dict(), seed=seed, argv=sys.argv)
    conditions = assign_control_kinds(build_conditions(experiment, config, phases))
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in conditions if c.condition_id in wanted]

    checkpoint_path = os.path.join(experiment_dir, "checkpoint.jsonl")
    done = set() if args.force else completed_conditions(checkpoint_path)

    pending = [c for c in conditions if c.condition_id not in done]
    total_runs = sum(1 + controls_for(c, config) for c in pending)
    log(f"{len(conditions)} conditions in this phase set, "
        f"{len(done & {c.condition_id for c in conditions})} already done, "
        f"{len(pending)} to run ({total_runs} evaluation passes)")

    # One cache for the whole run: the baseline and the resolved positions do not depend
    # on the condition, so paying for them once instead of once per condition is roughly
    # half the wall clock across the matrix.
    cache_path = os.path.join(experiment_dir, "sample_cache.json")
    if os.path.exists(cache_path) and not args.force:
        sample_records = load_sample_cache(cache_path)
        log(f"reusing sample cache: {len(sample_records)} samples ({cache_path})")
    else:
        log("building sample cache (one baseline + position resolve per sample, "
            "reused by every condition)...")
        cache_started = time.time()
        sample_records = build_sample_cache(
            experiment.ablator,
            dataset,
            position_type=experiment.position_type,
            logprob_normalize=experiment.logprob_normalize,
            max_samples=args.max_samples,
            show_progress=show_progress,
        )
        save_sample_cache(sample_records, cache_path)
        log(f"sample cache built in {human_duration(time.time() - cache_started)}")
    log(f"{len(sample_records)} samples cached")

    rc_cfg = config.get("random_control", {})
    sampling = str(rc_cfg.get("sampling", "matched"))
    matched_metric = str(rc_cfg.get("matched_metric", "activation_mean"))
    strict = bool(rc_cfg.get("strict_matching", True))
    log(f"control regime per layer: {experiment.effective_sampling(sampling, matched_metric)}")
    log("")

    run_started = time.time()
    condition_times = []

    for index, condition in enumerate(conditions, start=1):
        position = f"{index}/{len(conditions)}"
        if condition.condition_id in done:
            log(f"[SKIP {position}] {condition.condition_id} (already in checkpoint.jsonl)")
            continue

        n_controls = controls_for(condition, config)
        log(f"[RUN {position}] {condition.condition_id}")
        log(condition.describe(), indent=13)
        if condition.label:
            log(f"-> {condition.label}", indent=13)
        log(f"1 binding pass + {n_controls} control passes, {args.max_samples} samples each",
            indent=13)

        started = time.time()
        rows, summary = experiment.run_condition(
            dataset,
            condition,
            sample_records,
            max_samples=args.max_samples,
            show_progress=show_progress,
            progress_desc=f"  binding {condition.condition_id}",
        )
        binding_seconds = time.time() - started
        log(
            f"binding margin_drop = {summary.get('mean_margin_drop'):+.4f}  "
            f"acc_drop = {summary.get('accuracy_drop'):+.4f}  "
            f"({human_duration(binding_seconds)}){gpu_memory_note()}",
            indent=13,
        )

        control_summaries, control_sets = [], []
        if n_controls:
            # One log line per control set rather than a progress bar. Each set takes about
            # as long as the binding pass, so 15 of them is the ~19 minutes that previously
            # went by in silence; and a bar redrawn through tee becomes thousands of lines
            # in the log file.
            log(f"drawing and running {n_controls} joint control sets "
                f"(~{human_duration(binding_seconds * n_controls)} estimated)", indent=13)
            rng = random.Random(int(rc_cfg.get("seed", seed)))
            controls_started = time.time()
            for set_index in range(n_controls):
                control_features = experiment.sample_joint_random(
                    condition.features,
                    rng=rng,
                    sampling=sampling,
                    matched_metric=matched_metric,
                    strict=strict,
                )
                control_condition = Condition(
                    condition_id=f"{condition.condition_id}__control{set_index}",
                    kind=condition.kind,
                    features=control_features,
                    knockout_layers=condition.knockout_layers,
                    flow=condition.flow,
                    mode=condition.mode,
                )
                _, control_summary = experiment.run_condition(
                    dataset,
                    control_condition,
                    sample_records,
                    max_samples=args.max_samples,
                    show_progress=False,
                )
                control_summary["set_index"] = set_index
                control_summaries.append(control_summary)
                control_sets.append(
                    {str(l): v for l, v in sorted(control_features.items())}
                )

                controls_elapsed = time.time() - controls_started
                per_set = controls_elapsed / (set_index + 1)
                controls_eta = per_set * (n_controls - set_index - 1)
                log(
                    f"control {set_index + 1}/{n_controls}: "
                    f"margin_drop {control_summary.get('mean_margin_drop'):+.5f}  "
                    f"({human_duration(per_set)}/set, "
                    f"{human_duration(controls_eta)} left in this condition)",
                    indent=15,
                )

            drops = [c.get("mean_margin_drop") for c in control_summaries]
            if all(d is not None for d in drops):
                mean_drop = sum(drops) / len(drops)
                spread = (
                    sum((d - mean_drop) ** 2 for d in drops) / len(drops)
                ) ** 0.5
                log(
                    f"controls mean = {mean_drop:+.5f}  sd = {spread:.5f}  "
                    f"(n = {len(drops)})",
                    indent=13,
                )

        payload = {
            "condition_id": condition.condition_id,
            "label": condition.label,
            "summary": summary,
            "control_summaries": control_summaries,
            "control_feature_sets": control_sets,
            "significance": (
                experiment.compare_to_controls(summary, control_summaries)
                if control_summaries
                else {}
            ),
            "meta": {
                "feature_selection": "causal_v2",
                "intervention": "multilayer_sae_ablation",
                "layers": condition.layers,
                "knockout_layers": list(condition.knockout_layers),
                "mode": condition.mode,
                "encode_mode": config.get("multilayer", {}).get("encode_mode", "live"),
                "control": f"{sampling}_{matched_metric}" + ("_strict" if strict else ""),
                "control_effective": experiment.effective_sampling(sampling, matched_metric),
                "n_random_sets": n_controls,
                "n_samples": summary.get("n_samples"),
                "seed": seed,
                # Kept for readers of the older single-layer result files.
                "ablation_version": "v2_causal_features",
            },
        }
        write_condition(experiment_dir, condition.condition_id, rows, payload)
        append_checkpoint(checkpoint_path, {
            "condition_id": condition.condition_id,
            "mean_margin_drop": summary.get("mean_margin_drop"),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        if payload["significance"]:
            z = payload["significance"].get("mean_margin_drop", {}).get("z_score")
            if z is not None:
                log(f"z = {z:.1f} vs controls", indent=13)

        elapsed = time.time() - started
        condition_times.append(elapsed)
        remaining = [
            c for c in conditions[index:]
            if c.condition_id not in done
        ]
        # ETA from the mean of what has actually run, scaled by each remaining
        # condition's pass count -- a 16-pass condition is not one 1-pass condition.
        passes_done = sum(
            1 + controls_for(c, config)
            for c in conditions[:index] if c.condition_id not in done
        )
        per_pass = (sum(condition_times) / passes_done) if passes_done else None
        passes_left = sum(1 + controls_for(c, config) for c in remaining)
        eta = per_pass * passes_left if per_pass else None

        log(
            f"[DONE {position}] {condition.condition_id} in {human_duration(elapsed)} "
            f"| total elapsed {human_duration(time.time() - run_started)} "
            f"| {len(remaining)} conditions / {passes_left} passes left "
            f"| ETA {human_duration(eta)}"
            + (f" (~{(datetime.now() + timedelta(seconds=eta)).strftime('%H:%M')})" if eta else "")
        )
        log("")

    log(f"phase set complete. Results under {experiment_dir}/conditions/")
    log(f"total wall clock: {human_duration(time.time() - run_started)}")


def write_condition(experiment_dir, condition_id, rows, payload):
    """Write the raw per-sample records and the committed summary beside them.

    `results.json` is gitignored bulk — 16 MB across a 47-condition matrix. `summary.json`
    is what the repo carries, so it also folds in the per-sample margin drops the analysis
    pairs across conditions; without them a fresh clone could not recompute a CI. Written
    at the same indent `distill_results.py` uses, so a later distil pass is a no-op.
    """
    target = os.path.join(experiment_dir, "conditions", condition_id)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "results.json"), "w") as handle:
        json.dump({"per_sample": rows, **payload}, handle, indent=2)
    summary = {**payload, "per_sample_distilled": distill_condition_samples(rows)}
    with open(os.path.join(target, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


def dry_run(config, phases, args):
    """Print the matrix without loading the model, so it can be reviewed first."""
    catalogs, stats = load_catalogs_and_stats(config)
    ml_cfg = config.get("multilayer", {})

    experiment = MultiLayerAblationExperiment.__new__(MultiLayerAblationExperiment)
    experiment.catalogs = catalogs
    experiment.feature_stats = stats
    experiment.saes = {}
    experiment.config = config

    conditions = assign_control_kinds(build_conditions(experiment, config, phases))
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in conditions if c.condition_id in wanted]

    total_runs = 0
    print(f"\nPhases: {', '.join(phases)}")
    print(f"Span: {ml_cfg.get('layers')}  excluded: {ml_cfg.get('excluded_layers')}")
    print(f"Samples per condition: {args.max_samples}\n")
    print(f"{'#':>3}  {'condition':<44} {'runs':>5}  detail")
    print("-" * 118)
    for index, condition in enumerate(conditions, start=1):
        n_controls = controls_for(condition, config)
        runs = 1 + n_controls
        total_runs += runs
        print(f"{index:>3}  {condition.condition_id:<44} {runs:>5}  {condition.describe()}")
        if condition.label:
            print(f"{'':>3}  {'':<44} {'':>5}  -> {condition.label}")

    rc_cfg = config.get("random_control", {})
    print("-" * 118)
    print(f"{len(conditions)} conditions, {total_runs} evaluation runs of "
          f"{args.max_samples} samples each")
    print(
        f"\nControl regime: sampling={rc_cfg.get('sampling')} "
        f"metric={rc_cfg.get('matched_metric')} strict={rc_cfg.get('strict_matching')}"
    )
    print(f"Effective per layer: {experiment.effective_sampling(str(rc_cfg.get('sampling', 'matched')), str(rc_cfg.get('matched_metric', 'activation_mean')))}")
    return 0


if __name__ == "__main__":
    main()
