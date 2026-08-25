"""Per-layer attention knockout sweep (pipeline stage 00).

Writes, under ``{experiment_dir}/{knockout.output_subdir}/``:
``knockout_results.json`` (per-sample rows), ``knockout_summary.json``
(per-flow, per-layer aggregates), and a resumable ``checkpoint.jsonl``.
"""

import vlmflowprobe.cli._bootstrap  # noqa: F401  (env defaults before torch)

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--flows", type=str, default=None, help="Comma-separated flow names.")
    parser.add_argument("--override", action="append", default=[],
                        help="config override, section.key=value (repeatable)")
    args = parser.parse_args()

    from vlmflowprobe.core.config import load_config, save_config
    from vlmflowprobe.data.datasets import build_dataset
    from vlmflowprobe.knockout.runner import run_knockout_sweep
    from vlmflowprobe.utils.provenance import write_provenance
    from vlmflowprobe.utils.runtime import load_adapter, setup_experiment

    config = load_config(args.config, overrides=args.override)
    knockout_cfg = config.get("knockout", {})

    # The archive's stage 00 never seeded; every stage here does.
    experiment_dir, seed = setup_experiment(args, config)
    knockout_dir = os.path.join(experiment_dir, knockout_cfg.get("output_subdir", "knockout"))
    os.makedirs(knockout_dir, exist_ok=True)

    save_config(config, os.path.join(experiment_dir, "config.yaml"))
    write_provenance(experiment_dir, config=config.to_dict(), seed=seed, argv=sys.argv)

    adapter = load_adapter(config)
    dataset = build_dataset(config, tokenizer=adapter.tokenizer)

    flows = knockout_cfg.get("flows", ["Image->Question", "Image->Last"])
    if args.flows:
        flows = [flow.strip() for flow in args.flows.split(",") if flow.strip()]

    window = args.window if args.window is not None else knockout_cfg.get("window", 1)
    max_samples = args.max_samples if args.max_samples is not None else knockout_cfg.get("max_samples")

    checkpoint_path = os.path.join(knockout_dir, "checkpoint.jsonl")
    results, summaries = run_knockout_sweep(
        adapter,
        dataset,
        flows=flows,
        window=window,
        max_samples=max_samples,
        filter_correct=knockout_cfg.get("filter_correct", True),
        normalize_logprob=knockout_cfg.get("normalize_logprob", True),
        progress_desc="Knockout sweep",
        checkpoint_path=checkpoint_path,
    )

    results_path = os.path.join(knockout_dir, "knockout_results.json")
    summary_path = os.path.join(knockout_dir, "knockout_summary.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    print(f"Saved knockout results ({len(results)} rows) to {results_path}")
    print(f"Saved knockout summary ({len(summaries)} entries) to {summary_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()
