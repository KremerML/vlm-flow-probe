"""Smoke test for the matched random-control re-run.

Answers, before committing an overnight run: are the configs plumbed through,
does the sampler actually match, and does strict mode refuse the broken metric?

Phase A (CPU, seconds) drives the sampler directly on the real archived
`causal_feature_stats.json` for each layer:

  1. config plumbing  -- what the loaded config resolves sampling settings to
  2. broken metric    -- the archived `correct_mean` must RAISE under strict
  3. legacy repro     -- non-strict `correct_mean` reproduces uniform controls
  3b. pool depth      -- how many INDEPENDENT matched sets the dictionary allows
  4. matched sampling -- `activation_mean` matches the binding distribution
  5. determinism      -- same seed, same set; different offset, different set

Phase B (GPU, ~2 min, --with-gpu) runs a tiny real ablation through
`run_three_condition_test` so the whole path -- config -> sampler -> ablator ->
results payload -- is exercised end to end.

Extended matching diagnostics are enabled here via `random_control.log_matching`
(default false everywhere else).
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import json
import os
import random
import sys
from pathlib import Path

ARCHIVE_DEFAULT = Path.home() / "Documents/Github/cross-modal-information-flow-in-MLLM"

# The archive's per-layer causal stats. Layer 10 sits one directory deeper --
# a real irregularity the multilayer config also has to spell out explicitly.
STATS_REL = {
    10: "output/sae_experiments/sae_clevr_lite_layer10_attn_out_question_causal/results/ablation_v2/causal_feature_stats.json",
    11: "output/sae_experiments/sae_clevr_lite_layer11_attn_out_question_causal/causal_feature_stats.json",
    12: "output/sae_experiments/sae_clevr_lite_layer12_attn_out_question_causal/causal_feature_stats.json",
    13: "output/sae_experiments/sae_clevr_lite_layer13_attn_out_question_causal/causal_feature_stats.json",
    14: "output/sae_experiments/sae_clevr_lite_layer14_attn_out_question_causal/causal_feature_stats.json",
}
CATALOG_REL = {
    10: "output/sae_experiments/sae_clevr_lite_layer10_attn_out_question_causal/results/ablation_v2/causal_feature_catalog.json",
    11: "output/sae_experiments/sae_clevr_lite_layer11_attn_out_question_causal/causal_feature_catalog.json",
    12: "output/sae_experiments/sae_clevr_lite_layer12_attn_out_question_causal/causal_feature_catalog.json",
    13: "output/sae_experiments/sae_clevr_lite_layer13_attn_out_question_causal/causal_feature_catalog.json",
    14: "output/sae_experiments/sae_clevr_lite_layer14_attn_out_question_causal/causal_feature_catalog.json",
}

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f": {detail}" if detail else ""))
    return ok


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


class _FakeSAE:
    """Stands in for the real dictionary: the sampler only reads n_features."""

    def __init__(self, n_features):
        self.n_features = n_features


def load_layer(archive, layer):
    stats_path = archive / STATS_REL[layer]
    catalog_path = archive / CATALOG_REL[layer]
    if not stats_path.exists() or not catalog_path.exists():
        return None, None
    with open(stats_path) as fh:
        stats = {int(k): v for k, v in json.load(fh).items()}
    with open(catalog_path) as fh:
        catalog = json.load(fh)
    binding = [int(k) for k in (catalog.get("features", catalog))]
    return stats, binding


def phase_a(archive, layers, n_sets, top_k):
    from vlmflowprobe.ablation.ablation_experiments import AblationExperiment
    from vlmflowprobe.ablation.matching_diagnostics import (
        MatchingDiagnostics, format_report, matched_pool_depth,
    )
    from vlmflowprobe.core.config import load_config

    # ---------------------------------------------------------------- 1. config plumbing
    section("1. CONFIG PLUMBING")
    repo = Path(__file__).resolve().parents[3]
    frozen = repo / "configs/frozen/sae_layer11_attn_out_question.yaml"
    live = repo / "configs/experiments/llava15/sae_layer11_attn_out_question.yaml"

    cfg_frozen = load_config(str(frozen))
    rc = cfg_frozen.get("random_control", {})
    print(f"  frozen config     : matched_metric={rc.get('matched_metric')!r} "
          f"strict={rc.get('strict_matching')} n_sets={rc.get('n_random_sets')}")
    check("frozen config preserves the archived (broken) settings",
          rc.get("matched_metric") == "correct_mean" and rc.get("strict_matching") is False)

    cfg_live = load_config(str(live))
    rc_live = cfg_live.get("random_control", {})
    print(f"  experiment config : matched_metric={rc_live.get('matched_metric')!r} "
          f"strict={rc_live.get('strict_matching')} n_sets={rc_live.get('n_random_sets')}")
    check("experiment config defaults to matched+strict",
          rc_live.get("matched_metric") == "activation_mean" and rc_live.get("strict_matching") is True)

    # This is the exact override form the overnight re-run will use on frozen configs.
    overrides = [
        "random_control.matched_metric=activation_mean",
        "random_control.strict_matching=true",
        "random_control.log_matching=true",
    ]
    cfg_over = load_config(str(frozen), overrides=overrides)
    rc_over = cfg_over.get("random_control", {})
    print(f"  frozen+overrides  : matched_metric={rc_over.get('matched_metric')!r} "
          f"strict={rc_over.get('strict_matching')} log={rc_over.get('log_matching')}")
    check("CLI overrides reach random_control (the re-run's mechanism)",
          rc_over.get("matched_metric") == "activation_mean"
          and rc_over.get("strict_matching") is True
          and rc_over.get("log_matching") is True,
          " ".join(overrides))

    # ---------------------------------------------------------------- per-layer sampler
    experiment = AblationExperiment.__new__(AblationExperiment)

    for layer in layers:
        stats, binding = load_layer(archive, layer)
        if stats is None:
            print(f"\n  (layer {layer}: archived stats not found -- skipped)")
            continue
        binding = binding[:top_k]
        experiment.sae = _FakeSAE(len(stats))

        section(f"LAYER {layer}  ({len(stats)} features, {len(binding)} binding)")

        keys = sorted({k for v in list(stats.values())[:50] for k in v})
        print(f"  stats keys present: {keys}")
        check(f"L{layer}: stats carry 'activation_mean'", "activation_mean" in keys)
        check(f"L{layer}: stats do NOT carry 'correct_mean' (the archived bug's cause)",
              "correct_mean" not in keys)

        # -------------------------------------------------- 2. strict refuses broken metric
        print("\n  -- 2. broken metric under strict mode --")
        raised = False
        try:
            experiment._sample_random_features(
                binding_features=binding, n_random_features=len(binding),
                sampling="matched", feature_stats=stats, matched_metric="correct_mean",
                rng=random.Random(42), strict_matching=True,
            )
        except ValueError as exc:
            raised = True
            msg = str(exc)[:90]
        check(f"L{layer}: strict + 'correct_mean' raises instead of silently going uniform",
              raised, msg if raised else "NO EXCEPTION -- the guard is not working")

        # -------------------------------------------------- 3. legacy path reproduces uniform
        print("\n  -- 3. archived behaviour (non-strict 'correct_mean') --")
        diag_legacy = MatchingDiagnostics(metric="correct_mean", strict=False)
        legacy = experiment._sample_random_features(
            binding_features=binding, n_random_features=len(binding),
            sampling="matched", feature_stats=stats, matched_metric="correct_mean",
            rng=random.Random(42), strict_matching=False, diagnostics=diag_legacy,
        )
        b_vals = [stats[f]["activation_mean"] for f in binding if f in stats]
        l_vals = [stats[f]["activation_mean"] for f in legacy if f in stats]
        rep_legacy = diag_legacy.summarize(binding_values=b_vals, causal_top_k=set(binding))
        # Legacy draws carry no metric, so describe them on activation_mean directly.
        from vlmflowprobe.ablation.matching_diagnostics import _describe
        print(f"     binding activation_mean median: {_describe(b_vals)['median']:.6g}")
        print(f"     control activation_mean median: {_describe(l_vals)['median']:.6g}")
        ratio_legacy = _describe(l_vals)["median"] / max(_describe(b_vals)["median"], 1e-300)
        print(f"     control/binding ratio: {ratio_legacy:.3g}")
        print(f"     draw paths: {rep_legacy['draw_paths']}")
        check(f"L{layer}: legacy path reproduces the ghost-feature controls (ratio < 0.01)",
              ratio_legacy < 0.01, f"ratio={ratio_legacy:.3g}")

        # -------------------------------------------------- 3b. pool depth
        print("\n  -- 3b. how deep is the matched candidate pool? --")
        depth = matched_pool_depth(stats, binding, "activation_mean")
        print(f"     binding activation range : [{depth['binding_range'][0]:.4f}, "
              f"{depth['binding_range'][1]:.4f}] (n={depth['n_binding']})")
        print(f"     non-binding features in that range : {depth['n_candidates_in_range']} "
              f"of {depth['dictionary_size']}")
        print(f"     inside the binding IQR             : {depth['n_candidates_in_iqr']}")
        print(f"     max disjoint matched sets possible : "
              f"{depth['max_disjoint_matched_sets']:.2f}")
        check(f"L{layer}: pool supports at least one full matched set",
              depth["sufficient_for_one_set"],
              f"{depth['n_candidates_in_range']} candidates for {depth['n_binding']} slots")
        requested = n_sets
        feasible = depth["max_disjoint_matched_sets"] >= requested
        check(f"L{layer}: pool supports the {requested} INDEPENDENT sets a z-score needs",
              feasible,
              f"only {depth['max_disjoint_matched_sets']:.2f} disjoint sets available -- "
              f"control sets must reuse features, collapsing between-set variance")

        # -------------------------------------------------- 4. matched sampling works
        print("\n  -- 4. matched sampling on 'activation_mean' --")
        diag = MatchingDiagnostics(metric="activation_mean", strict=True)
        matched_sets = []
        for set_idx in range(n_sets):
            matched = experiment._sample_random_features(
                binding_features=binding, n_random_features=len(binding),
                sampling="matched", feature_stats=stats, matched_metric="activation_mean",
                rng=random.Random(42 + set_idx), strict_matching=True, diagnostics=diag,
            )
            matched_sets.append(matched)
        report = diag.summarize(binding_values=b_vals, causal_top_k=set(binding))
        print(format_report(report))

        ratio = report.get("control_over_binding_median_ratio")
        check(f"L{layer}: control median within 5x of binding median",
              ratio is not None and 0.2 <= ratio <= 5.0,
              f"ratio={ratio:.4g}" if ratio else "no ratio")
        check(f"L{layer}: every draw used the matched path",
              report["matched_fraction"] == 1.0,
              f"{100 * report['matched_fraction']:.1f}% matched, paths={report['draw_paths']}")
        check(f"L{layer}: no silent fallback-key substitutions",
              report["n_fallback_key_uses"] == 0,
              f"{report['n_fallback_key_uses']} uses")
        check(f"L{layer}: controls disjoint from the binding set",
              all(not (set(m) & set(binding)) for m in matched_sets))
        med_err = report["abs_error"].get("median")
        rel_err = med_err / report["binding"]["median"] if med_err is not None else None
        check(f"L{layer}: median matching error below 25% of binding median",
              rel_err is not None and rel_err < 0.25,
              f"median |err|={med_err:.4g} = {rel_err:.1%} of binding median "
              f"(uniform sampling would be ~100%)")

        # Independence: 15 control sets only give a meaningful z if they differ.
        import statistics as _st
        overlaps = [
            len(set(a) & set(b)) / len(binding)
            for i, a in enumerate(matched_sets) for b in matched_sets[i + 1:]
        ]
        distinct = len(set().union(*[set(m) for m in matched_sets]))
        set_medians = [
            _st.median(stats[f]["activation_mean"] for f in m) for m in matched_sets
        ]
        spread = _st.stdev(set_medians) if len(set_medians) > 1 else 0.0
        print(f"     pairwise set overlap : {_st.mean(overlaps):.1%} mean"
              if overlaps else "     (single set)")
        print(f"     distinct features across {len(matched_sets)} sets : {distinct} "
              f"(of {len(matched_sets) * len(binding)} draws)")
        print(f"     between-set median spread : {spread:.3g}")
        if overlaps:
            check(f"L{layer}: control sets are meaningfully independent (<50% overlap)",
                  _st.mean(overlaps) < 0.5,
                  f"{_st.mean(overlaps):.1%} overlap, {distinct} distinct features -- "
                  f"between-set variance {'collapsed' if spread == 0 else 'small'}, "
                  f"so a 15-set z is not estimable")

        # -------------------------------------------------- 5. determinism
        print("\n  -- 5. determinism --")
        again = experiment._sample_random_features(
            binding_features=binding, n_random_features=len(binding),
            sampling="matched", feature_stats=stats, matched_metric="activation_mean",
            rng=random.Random(42), strict_matching=True,
        )
        check(f"L{layer}: same seed reproduces the same control set", again == matched_sets[0])
        if len(matched_sets) > 1:
            check(f"L{layer}: different seed offset gives a different set",
                  matched_sets[0] != matched_sets[1],
                  f"overlap {len(set(matched_sets[0]) & set(matched_sets[1]))}/{len(binding)}")


def phase_b(archive, layer, n_samples, n_sets):
    """Tiny real ablation: config -> sampler -> ablator -> results payload."""
    section(f"PHASE B: end-to-end ablation, layer {layer}, n={n_samples}, {n_sets} control sets")
    from vlmflowprobe.ablation.ablation_experiments import AblationExperiment
    from vlmflowprobe.ablation.matching_diagnostics import format_report
    from vlmflowprobe.core.config import load_config
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.utils.runtime import load_adapter, load_sae

    repo = Path(__file__).resolve().parents[3]
    overrides = [
        "random_control.matched_metric=activation_mean",
        "random_control.strict_matching=true",
        "random_control.log_matching=true",
        f"random_control.n_random_sets={n_sets}",
        f"ablation.n_random_sets={n_sets}",
        f"dataset.data_dir={archive}/datasets/clevr_lite",
        "dataset.split=val",
    ]
    config = load_config(
        str(repo / f"configs/frozen/sae_layer{layer}_attn_out_question.yaml"),
        overrides=overrides,
    )
    print(f"  resolved: {config.get('random_control')}")

    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer, split="val")
    ckpt = archive / f"output/sae_experiments/sae_clevr_lite_layer{layer}_attn_out_question/sae_checkpoint.pt"
    sae = load_sae(config, adapter, str(ckpt))

    stats, binding = load_layer(archive, layer)
    experiment = AblationExperiment(adapter, sae, config)
    results = experiment.run_three_condition_test(
        dataset, binding, feature_stats=stats, show_progress=False, max_samples=n_samples,
    )

    settings = results.get("random_control_settings", {})
    print(f"\n  sampling_effective: {settings.get('sampling_effective')!r}")
    check("run reports matched sampling (not a uniform fallback)",
          settings.get("sampling_effective") == "matched_activation_mean",
          str(settings.get("sampling_effective")))
    check("log_matching recorded in the results payload",
          settings.get("log_matching") is True)

    report = results.get("matching_diagnostics")
    check("matching diagnostics present in results", report is not None)
    if report:
        print(format_report(report))
        ratio = report.get("control_over_binding_median_ratio")
        check("end-to-end control/binding median ratio within 5x",
              ratio is not None and 0.2 <= ratio <= 5.0, f"ratio={ratio:.4g}" if ratio else "-")

    binding_drop = results.get("binding", {}).get("mean_margin_drop")
    random_drop = results.get("random", {}).get("mean_margin_drop")
    sig = results.get("significance", {}).get("mean_margin_drop", {})
    print(f"\n  binding margin_drop : {binding_drop:+.4f}")
    print(f"  control margin_drop : {random_drop:+.4f}   (n={n_samples}, indicative only)")
    print(f"  z                   : {sig.get('z_score')}")
    check("ablation produced numeric results for both arms",
          binding_drop is not None and random_drop is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-root", default=str(ARCHIVE_DEFAULT))
    parser.add_argument("--layers", default="11,14", help="comma-separated (default 11,14)")
    parser.add_argument("--sets", type=int, default=3, help="control sets in phase A")
    parser.add_argument("--top-k", type=int, default=200, help="binding features to use")
    parser.add_argument("--with-gpu", action="store_true", help="also run phase B")
    parser.add_argument("--gpu-layer", type=int, default=11)
    parser.add_argument("--gpu-samples", type=int, default=8)
    parser.add_argument("--gpu-sets", type=int, default=2)
    parser.add_argument("--out", default=None, help="write the check list as JSON here")
    args = parser.parse_args()

    archive = Path(os.path.expanduser(args.archive_root))
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    print("MATCHED-CONTROL SMOKE TEST")
    print(f"archive: {archive}")

    phase_a(archive, layers, args.sets, args.top_k)
    if args.with_gpu:
        phase_b(archive, args.gpu_layer, args.gpu_samples, args.gpu_sets)

    section("SUMMARY")
    passed = sum(1 for _, ok in _results if ok)
    for name, ok in _results:
        if not ok:
            print(f"  FAILED: {name}")
    print(f"\n  {passed}/{len(_results)} checks passed")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"passed": passed, "total": len(_results),
             "checks": [{"name": n, "passed": ok} for n, ok in _results]}, indent=1))
    print("  SMOKE TEST PASSED" if passed == len(_results) else "  SMOKE TEST FAILED")
    sys.exit(0 if passed == len(_results) else 1)


if __name__ == "__main__":
    main()
