"""Analyze ablation results and generate reports."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vlmflowprobe.ablation import statistical_analysis
from vlmflowprobe.core.config import load_config
from vlmflowprobe.ablation.hypothesis_tester import HypothesisTester
from vlmflowprobe.utils.checkpoint_utils import resolve_experiment_dir


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
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_cfg = dict(config.get("experiment", {}))
    if args.experiment_name:
        experiment_cfg["name"] = args.experiment_name
        experiment_cfg.pop("output_dir", None)
    experiment_dir = resolve_experiment_dir(experiment_cfg, args.experiment_dir)

    results_path = args.results or os.path.join(experiment_dir, "results", "ablation_results.json")
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

    print(f"Saved analysis report to {output_dir}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()
