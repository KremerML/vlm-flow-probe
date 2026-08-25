"""Per-layer attention knockout runner for multimodal flows."""

from typing import Dict, Iterable, List, Optional, Tuple

import json
import os

from tqdm import tqdm

from vlmflowprobe.ablation import statistical_analysis
from vlmflowprobe.ablation import metrics as eval_metrics
from vlmflowprobe.adapters.base import ModelAdapter
from vlmflowprobe.data.loading import iter_batches
from vlmflowprobe.knockout.block_config import build_block_config, flow_block_pairs, flow_target
from vlmflowprobe.knockout.scoring import sequence_logprob


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _load_checkpoint(checkpoint_path: str) -> Tuple[set, List[Dict]]:
    """Load a JSONL checkpoint file written by run_knockout_sweep.

    Each line in the checkpoint is a JSON object with keys:
        "question_id": str
        "rows": list of per-layer/per-flow result dicts

    Returns:
        done_ids: set of question_ids that were fully completed
        existing_results: flat list of all row dicts from completed samples
    """
    done_ids: set = set()
    existing_results: List[Dict] = []
    if not os.path.exists(checkpoint_path):
        return done_ids, existing_results
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                q_id = entry["question_id"]
                done_ids.add(q_id)
                existing_results.extend(entry.get("rows", []))
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"[checkpoint] Warning: skipping malformed line {lineno}: {exc}")
    return done_ids, existing_results


def run_knockout_sweep(
    adapter: ModelAdapter,
    dataset,
    flows: List[str],
    window: int = 1,
    max_samples: Optional[int] = None,
    filter_correct: bool = True,
    normalize_logprob: bool = True,
    progress_desc: str = "Knockout sweep",
    checkpoint_path: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    num_layers = adapter.n_layers
    summaries: List[Dict] = []
    dataset_dict = dataset.dataset_dict

    # ── Checkpoint resume ───────────────────────────────────────────────────
    done_ids: set = set()
    results: List[Dict] = []
    checkpoint_file = None
    if checkpoint_path is not None:
        done_ids, results = _load_checkpoint(checkpoint_path)
        if done_ids:
            print(f"[checkpoint] Resuming: {len(done_ids)} samples already complete, "
                  f"{len(results)} rows loaded.")
        else:
            print(f"[checkpoint] Starting fresh. Checkpoint: {checkpoint_path}")
        checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")  # noqa: SIM115

    total = len(dataset.questions)
    if max_samples is not None:
        total = min(total, max_samples)

    total_steps = total * num_layers * max(1, len(flows))
    progress = tqdm(total=total_steps, desc=progress_desc, unit="step")

    try:
        for batch, line in iter_batches(dataset, adapter, max_samples=max_samples):
            question_id = line["q_id"]

            # ── Skip already-checkpointed samples ───────────────────────────
            if question_id in done_ids:
                progress.update(num_layers * max(1, len(flows)))
                continue

            detail = dataset_dict[question_id]
            true_option = detail.get("true option", "").strip()
            false_option = detail.get("false option", "").strip()
            if not true_option or not false_option:
                progress.update(num_layers * max(1, len(flows)))
                continue

            base_true_lp = sequence_logprob(adapter, batch, true_option, normalize=normalize_logprob)
            base_false_lp = sequence_logprob(adapter, batch, false_option, normalize=normalize_logprob)
            if base_true_lp is None or base_false_lp is None:
                progress.update(num_layers * max(1, len(flows)))
                continue
            base_margin = base_true_lp - base_false_lp
            if filter_correct and base_margin <= 0:
                progress.update(num_layers * max(1, len(flows)))
                continue

            sample_rows: List[Dict] = []
            for flow in flows:
                src_tgt_pairs = flow_block_pairs(flow, batch, adapter)
                if not src_tgt_pairs:
                    progress.update(num_layers)
                    continue
                target = flow_target(flow)

                for layer in range(num_layers):
                    block_config = build_block_config(layer, num_layers, window, src_tgt_pairs)
                    new_true_lp = sequence_logprob(
                        adapter, batch, true_option, normalize=normalize_logprob,
                        block_config=block_config, flow_target=target,
                    )
                    new_false_lp = sequence_logprob(
                        adapter, batch, false_option, normalize=normalize_logprob,
                        block_config=block_config, flow_target=target,
                    )
                    if new_true_lp is None or new_false_lp is None:
                        progress.update(1)
                        continue
                    new_margin = new_true_lp - new_false_lp
                    margin_drop = base_margin - new_margin
                    row = {
                        "question_id": question_id,
                        "flow": flow,
                        "layer": layer,
                        "base_true_logprob": base_true_lp,
                        "base_false_logprob": base_false_lp,
                        "base_margin": base_margin,
                        "new_true_logprob": new_true_lp,
                        "new_false_logprob": new_false_lp,
                        "new_margin": new_margin,
                        "margin_drop": margin_drop,
                    }
                    sample_rows.append(row)
                    results.append(row)
                    progress.update(1)

            # ── Write completed sample to checkpoint atomically ──────────────
            if sample_rows and checkpoint_file is not None:
                checkpoint_file.write(
                    json.dumps({"question_id": question_id, "rows": sample_rows}) + "\n"
                )
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())

    finally:
        progress.close()
        if checkpoint_file is not None:
            checkpoint_file.close()

    if not results:
        return results, summaries

    # ── Aggregate per-flow, per-layer summaries ──────────────────────────────
    by_key: Dict[Tuple[str, int], List[Dict]] = {}
    for row in results:
        key = (row["flow"], row["layer"])
        by_key.setdefault(key, []).append(row)

    for (flow, layer), rows in sorted(by_key.items()):
        base_margins = [r["base_margin"] for r in rows]
        new_margins = [r["new_margin"] for r in rows]
        t_stat, p_val = statistical_analysis.paired_t_test(base_margins, new_margins)
        effect = statistical_analysis.effect_size_cohens_d(base_margins, new_margins)
        summaries.append(
            {
                "flow": flow,
                "layer": layer,
                "samples": len(rows),
                "mean_base_margin": _mean(base_margins),
                "mean_new_margin": _mean(new_margins),
                "mean_margin_drop": eval_metrics.mean_margin_drop(base_margins, new_margins),
                "t_stat": t_stat,
                "p_value": p_val,
                "effect_size": effect,
            }
        )

    return results, summaries
