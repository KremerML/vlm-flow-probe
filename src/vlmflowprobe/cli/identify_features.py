"""Causal SAE feature identification via gradient attribution (stage 02).

Scores every SAE feature by |gradient| * |activation| with the SAE spliced
into the forward pass, backpropagating the forced-choice margin (or the
correct logit). Writes into ``{experiment_dir}_{output_suffix}/``:
``causal_feature_stats.json``, ``causal_summary.json``,
``causal_feature_catalog.json``.
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--target", type=str, default="margin",
                        choices=["margin", "correct_logit"])
    parser.add_argument("--position_type", type=str, default=None,
                        help="Override position_type from config")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Override top_k from config")
    parser.add_argument("--output_suffix", type=str, default="causal",
                        help="Suffix for output directory")
    # The archive's stage 02 read these via setup_experiment but never declared
    # them, so passing --experiment_dir was an argparse error. Fixed here.
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    from vlmflowprobe.core.config import load_config
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.features.causal_feature_identifier import CausalFeatureIdentifier
    from vlmflowprobe.utils.provenance import write_provenance
    from vlmflowprobe.utils.runtime import load_adapter, load_sae, setup_experiment

    config = load_config(args.config, overrides=args.override)
    model_cfg = config.get("model", {})
    feat_cfg = config.get("feature_identification", {})

    experiment_dir, seed = setup_experiment(args, config)
    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer, split="val")

    checkpoint_path = args.sae_checkpoint or os.path.join(experiment_dir, "sae_checkpoint.pt")
    sae = load_sae(config, adapter, checkpoint_path)

    target_layer = model_cfg.get("target_layer", 12)
    activation_site = model_cfg.get("activation_site", "residual")
    position_type = args.position_type or feat_cfg.get("position_type", "question")
    top_k = args.top_k or feat_cfg.get("top_k", 200)

    n_items = len(dataset.questions)
    effective_n = min(n_items, args.max_samples) if args.max_samples else n_items
    print("[02] Causal feature identification")
    print(f"  layer={target_layer}, site={activation_site}, "
          f"position_type={position_type}, target={args.target}")
    print(f"  n_items={n_items}, processing={effective_n}")

    identifier = CausalFeatureIdentifier(
        sae, adapter, dataset, target_layer, activation_site=activation_site,
    )
    feature_stats = identifier.compute_causal_scores(
        position_type=position_type,
        max_samples=args.max_samples,
        target=args.target,
        show_progress=not args.no_progress,
    )
    if not feature_stats:
        print("[02] ERROR: No samples processed successfully.")
        sys.exit(1)

    summary = identifier.summary
    print(f"\n[02] Done: {summary['n_processed']} samples processed, "
          f"{summary['n_skipped']} skipped")

    top_features = identifier.get_top_k_features(top_k)
    if top_features:
        scores = [feature_stats[f]["causal_score"] for f in top_features]
        print(f"\n[02] Top-{len(top_features)} features by causal_score: "
              f"max={max(scores):.6f}, min={min(scores):.6f}, "
              f"mean={sum(scores) / len(scores):.6f}")
        print(f"\n  {'Rank':>4} {'Feature':>8} {'Causal':>12} {'Activation':>12} {'Gradient':>12}")
        for rank, feat_idx in enumerate(top_features[:20], 1):
            s = feature_stats[feat_idx]
            print(f"  {rank:4d} {feat_idx:8d} {s['causal_score']:12.6f} "
                  f"{s['activation_mean']:12.6f} {s['gradient_mean']:12.6f}")

    output_dir = experiment_dir + (f"_{args.output_suffix}" if args.output_suffix else "")
    os.makedirs(output_dir, exist_ok=True)
    identifier.save_results(output_dir)
    catalog_path = identifier.export_catalog(output_dir, top_k=top_k)
    write_provenance(output_dir, config=config.to_dict(), seed=seed, argv=sys.argv)

    print(f"\nSaved causal feature stats to {output_dir}/")
    print(f"Saved feature catalog to {catalog_path}")

    for key in ("causal_score", "activation_mean", "gradient_mean"):
        dist = summary.get(key, {})
        print(f"  {key}: p50={dist.get('p50', 0):.2e}, p90={dist.get('p90', 0):.2e}, "
              f"p99={dist.get('p99', 0):.2e}, max={dist.get('max', 0):.2e}")


if __name__ == "__main__":
    main()
