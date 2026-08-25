"""Three-condition ablation: binding features vs matched random controls (stage 03).

Ablation mode comes from ``ablation.mode`` in the config — ``replace`` for
every active config. The optional pass-through baseline (zero features
zeroed) runs in error-preserving delta mode by construction and quantifies
SAE reconstruction error.

Reads ``{experiment_dir}_causal/causal_feature_catalog.json`` (falling back
to ``{experiment_dir}/feature_catalog.json``); writes
``{experiment_dir}_causal/results/ablation_v2_results.json``.
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import json
import os
import sys


def run_passthrough_baseline(ablator, dataset, position_type, max_samples=None, show_progress=False):
    """SAE pass-through (zero features zeroed, delta mode): reconstruction-error floor."""
    results = ablator.batch_ablation_experiment(
        dataset,
        feature_indices=[],
        position_type=position_type,
        mode="residual",
        show_progress=show_progress,
        max_samples=max_samples,
        score_options=True,
    )
    return ablator.compute_ablation_effect(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--skip_passthrough", action="store_true",
                        help="Skip the SAE pass-through baseline")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    from vlmflowprobe.ablation.ablation_experiments import AblationExperiment
    from vlmflowprobe.ablation.feature_ablator import FeatureAblator
    from vlmflowprobe.core.config import load_config
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.features.feature_catalog import FeatureCatalog
    from vlmflowprobe.utils.provenance import write_provenance
    from vlmflowprobe.utils.runtime import load_adapter, load_sae, setup_experiment

    config = load_config(args.config, overrides=args.override)
    model_cfg = config.get("model", {})
    ablation_cfg = config.get("ablation", {})
    experiment_dir, seed = setup_experiment(args, config)

    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer, split="val")

    checkpoint_path = args.sae_checkpoint or os.path.join(experiment_dir, "sae_checkpoint.pt")
    sae = load_sae(config, adapter, checkpoint_path)

    target_layer = model_cfg.get("target_layer", 0)
    activation_site = model_cfg.get("activation_site", "residual")
    position_type = ablation_cfg.get("position_type", "question")
    show_progress = not args.no_progress

    print(f"[03] layer={target_layer}, site={activation_site}, "
          f"position_type={position_type}, n_items={len(dataset.questions)}")

    ablator = FeatureAblator(adapter, sae, target_layer, activation_site=activation_site)

    passthrough_summary = None
    if not args.skip_passthrough:
        print("[03] Phase 1: SAE pass-through baseline (zero features zeroed)...")
        passthrough_summary = run_passthrough_baseline(
            ablator, dataset, position_type,
            max_samples=args.max_samples, show_progress=show_progress,
        )
        pt_md = passthrough_summary.get("mean_margin_drop", 0) or 0
        pt_rp = passthrough_summary.get("mean_relative_perturbation", 0) or 0
        print(f"[03] Pass-through: margin_drop={pt_md:+.4f}, rel_perturb={pt_rp:.4f}")
        if abs(pt_md) > 0.05:
            print(f"[03] WARNING: Pass-through margin_drop is {pt_md:+.4f} — "
                  f"reconstruction error may confound results")

    causal_dir = experiment_dir + "_causal"
    features_path = args.features or os.path.join(causal_dir, "causal_feature_catalog.json")
    if not os.path.exists(features_path):
        features_path = os.path.join(experiment_dir, "feature_catalog.json")
    catalog = FeatureCatalog()
    catalog.load_from_json(features_path)
    binding_features = list(catalog.features.keys())
    print(f"[03] Loaded {len(binding_features)} features from {features_path}")

    feature_stats_path = os.path.join(causal_dir, "causal_feature_stats.json")
    feature_stats = None
    if os.path.exists(feature_stats_path):
        with open(feature_stats_path) as f:
            feature_stats = {int(k): v for k, v in json.load(f).items()}

    print("[03] Phase 2: Three-condition ablation test...")
    experiment = AblationExperiment(adapter, sae, config)
    results = experiment.run_three_condition_test(
        dataset,
        binding_features,
        feature_stats=feature_stats,
        show_progress=show_progress,
        max_samples=args.max_samples,
    )

    results["passthrough_baseline"] = passthrough_summary
    results["meta"] = {
        "seed": seed,
        "target_layer": target_layer,
        "activation_site": activation_site,
        "feature_selection": "causal_v2",
        "intervention": "single_layer_sae_ablation",
        "mode": ablation_cfg.get("mode", "residual"),
        "control": results.get("random_control_settings", {}).get("sampling_effective"),
        "ablation_version": "v2_causal_features",
        "features_source": features_path,
    }

    output_path = args.output or os.path.join(causal_dir, "results", "ablation_v2_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    write_provenance(os.path.dirname(output_path), config=config.to_dict(), seed=seed, argv=sys.argv)

    b = results.get("binding", {})
    r = results.get("random", {})
    sig = results.get("significance", {})
    pt = passthrough_summary or {}

    print("[03] -- Results --------------------------------------")
    print(f"[03]   Baseline accuracy  : {results.get('baseline', {}).get('baseline_accuracy', 0):.3f}")
    if pt:
        print(f"[03]   Pass-through      | margin_drop={pt.get('mean_margin_drop', 0):+.4f}  "
              f"rel_perturb={pt.get('mean_relative_perturbation', 0):.4f}")
    print(f"[03]   Binding  (n={len(binding_features):>3})  | "
          f"margin_drop={b.get('mean_margin_drop', 0):+.4f}  "
          f"acc_drop={b.get('accuracy_drop', 0):+.4f}  "
          f"rel_perturb={b.get('mean_relative_perturbation', 0):.4f}")
    print(f"[03]   Random             | "
          f"margin_drop={r.get('mean_margin_drop', 0):+.4f}  "
          f"acc_drop={r.get('accuracy_drop', 0):+.4f}  "
          f"rel_perturb={r.get('mean_relative_perturbation', 0):.4f}")

    md_sig = sig.get("mean_margin_drop", {})
    if md_sig:
        print(f"[03]   Significance       | z={md_sig.get('z_score', 'N/A'):.1f}, "
              f"p={md_sig.get('empirical_p_value', 'N/A')}")
    if pt and pt.get("mean_margin_drop") is not None and b.get("mean_margin_drop") is not None:
        corrected = b["mean_margin_drop"] - pt["mean_margin_drop"]
        print(f"[03]   Corrected binding  | margin_drop={corrected:+.4f} "
              f"(after subtracting pass-through)")
    print(f"[03] Saved to {output_path}")


if __name__ == "__main__":
    main()
