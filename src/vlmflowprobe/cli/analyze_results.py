"""Analyze ablation results and generate reports."""

import argparse
import json
import os
import sys

from vlmflowprobe.ablation import statistical_analysis
from vlmflowprobe.core.config import load_config
from vlmflowprobe.ablation.hypothesis_tester import HypothesisTester
from vlmflowprobe.utils.provenance import write_provenance
from vlmflowprobe.utils.runtime import setup_experiment


def main() -> None:
    """Generate statistical and hypothesis-testing reports from ablation outputs.

    Args:
        None: CLI arguments are parsed inside this function.

    Returns:
        None: Writes analysis JSON and comparison plot artifacts to disk.

    Raises:
        FileNotFoundError: If the configured ablation results file is missing.
        ValueError: If report configuration is malformed.
        RuntimeError: If analysis routines fail.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    # The archive's stage 04 never seeded and defaulted --results to a file
    # stage 03 never writes; both fixed here.
    experiment_dir, seed = setup_experiment(args, config)
    results_path = args.results or os.path.join(
        experiment_dir + "_causal", "results", "ablation_v2_results.json"
    )
    with open(results_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)

    tester = HypothesisTester(config)
    hypothesis = tester.test_causal_necessity(results)
    eval_cfg = config.get("evaluation", {})
    report = statistical_analysis.generate_statistical_report(
        results,
        metric=eval_cfg.get("primary_metric", "pred_token_prob"),
    )
    report.update(hypothesis)

    output_dir = args.output or os.path.join(experiment_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    statistical_analysis.plot_ablation_comparison(
        results,
        os.path.join(output_dir, "ablation_comparison.png"),
    )

    with open(os.path.join(output_dir, "hypothesis_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_provenance(output_dir, config=config.to_dict(), seed=seed, argv=sys.argv)

    print(f"Saved analysis report to {output_dir}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()
