"""Per-sample work that is invariant across ablation conditions.

A condition sweep runs the same dataset many times over. Two things are recomputed on
every pass even though neither depends on the condition:

* the unablated baseline (one ``generate`` plus one ``sequence_logprob`` per option), and
* the resolved intervention positions, which run ``prepare_inputs_labels_for_multimodal``
  and therefore a full vision-tower pass.

``run_three_condition_test`` evaluates 1 binding + 15 random sets, so the baseline alone was
being recomputed 16 times per sample. Building this cache once and passing it to every
condition removes both.

The cache is a plain list ordered exactly as ``dataset.create_dataloader()`` yields, which is
what ``FeatureAblator._resolve_cached_baseline`` and ``_resolve_cached_positions`` expect.
Each record also carries its ``question_id`` so a mis-ordered cache is detectable rather than
silently wrong -- pass ``strict_cache=True`` to turn a miss into an error.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from vlmflowprobe.data.loading import iter_batches


@dataclass
class SampleRecord:
    """Everything about one sample that does not change between conditions."""

    question_id: str
    positions: Optional[List[int]]
    baseline: Dict[str, Any] = field(default_factory=dict)

    @property
    def position_count(self) -> Optional[int]:
        return len(self.positions) if self.positions else None


def build_sample_cache(
    ablator,
    dataset,
    position_type: str = "question",
    logprob_normalize: bool = True,
    max_samples: Optional[int] = None,
    score_options: bool = True,
    show_progress: bool = False,
    progress_desc: str = "Sample cache",
) -> List[SampleRecord]:
    """Resolve baselines and positions once for every sample.

    Reuses ``FeatureAblator._compute_baseline_record`` and ``_resolve_positions`` so the
    cached values are identical to what an uncached run would compute inline.
    """
    records: List[SampleRecord] = []

    for batch, line in iter_batches(
        dataset,
        ablator.adapter,
        max_samples=max_samples,
        show_progress=show_progress,
        progress_desc=progress_desc,
    ):
        positions = ablator._resolve_positions(batch, position_type, line)
        baseline = ablator._compute_baseline_record(
            batch=batch,
            dataset=dataset,
            line=line,
            logprob_normalize=logprob_normalize,
            score_options=score_options,
        )
        records.append(
            SampleRecord(
                question_id=str(line["q_id"]),
                positions=list(positions) if positions is not None else None,
                baseline=baseline,
            )
        )
    return records


def baseline_cache(records: List[SampleRecord]) -> List[Dict[str, Any]]:
    """The list ``batch_ablation_experiment(baseline_cache=...)`` expects."""
    return [record.baseline for record in records]


def positions_cache(records: List[SampleRecord]) -> List[Optional[List[int]]]:
    """The list ``batch_ablation_experiment(positions_cache=...)`` expects."""
    return [record.positions for record in records]


def save_sample_cache(records: List[SampleRecord], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {"n_samples": len(records), "records": [asdict(r) for r in records]}
    with open(path, "w") as handle:
        json.dump(payload, handle)


def load_sample_cache(path: str) -> List[SampleRecord]:
    with open(path) as handle:
        payload = json.load(handle)
    return [SampleRecord(**record) for record in payload["records"]]

