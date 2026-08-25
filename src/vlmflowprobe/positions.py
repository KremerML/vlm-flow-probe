"""The single position resolver.

The archive carried three near-identical copies of this logic
(``activation_collector._select_positions``,
``causal_feature_identifier._resolve_positions``,
``feature_ablator._resolve_positions``) with three deliberate semantic
divergences. Those divergences produced the published artifacts, so they are
kept — but as an explicit :class:`PositionPolicy` instead of drifted copies.

* ``COLLECTION_POLICY`` reproduces the collector/identifier semantics: it
  produced the SAE training data and the causal feature scores.
* ``ABLATION_POLICY`` reproduces the feature-ablator semantics: it produced
  the published ablation numbers. Its ``"all"`` returns ``None`` — the
  ablator's sentinel for "no position restriction" — and an empty question
  span short-circuits every position type to ``[]``.

All coordinates are post-expansion (see ``adapters.base``).
"""

from dataclasses import dataclass
from typing import List, Optional

from vlmflowprobe.adapters.base import ModelAdapter, ModelBatch


@dataclass(frozen=True)
class PositionPolicy:
    # "none_sentinel": position_type "all" -> None (no restriction);
    # "full_range": -> explicit [0..S) list.
    all_positions: str
    # If True, an empty question span returns [] before any branch — even for
    # "last"/"image", which do not otherwise depend on the question.
    early_empty_question: bool
    # "attribute" with no attribute tokens: "question_span" | "empty"
    attribute_fallback: str


# Produced the SAE training data and causal feature scores.
COLLECTION_POLICY = PositionPolicy(
    all_positions="full_range", early_empty_question=False, attribute_fallback="empty"
)
# Produced the published ablation numbers.
ABLATION_POLICY = PositionPolicy(
    all_positions="none_sentinel", early_empty_question=True, attribute_fallback="question_span"
)


def resolve_positions(
    position_type: Optional[str],
    batch: ModelBatch,
    adapter: ModelAdapter,
    *,
    policy: PositionPolicy,
    line: Optional[dict] = None,
) -> Optional[List[int]]:
    """Token positions an intervention touches, or ``None`` for "everywhere".

    ``line`` is the dataset's question record; only the ``"attribute"``
    position type reads it (``line["attribute_tokens"]`` offsets are relative
    to the question span).
    """
    if position_type in (None, "all"):
        if policy.all_positions == "none_sentinel":
            return None
        return list(range(batch.seq_len))

    question_range = adapter.question_token_span(batch)
    if policy.early_empty_question and not question_range:
        return []

    if position_type == "question":
        return question_range

    if position_type == "last":
        return [adapter.last_token_index(batch)]

    if position_type == "image":
        return list(adapter.image_token_span(batch))

    if position_type == "attribute":
        attr_positions: List[int] = []
        for attr in (line or {}).get("attribute_tokens", []):
            attr_positions.extend(attr.get("positions", []))
        if not attr_positions:
            if policy.attribute_fallback == "question_span":
                return question_range
            return []
        if not question_range:
            return []
        start, end = question_range[0], question_range[-1]
        positions = [start + pos for pos in attr_positions]
        return sorted({pos for pos in positions if start <= pos <= end})

    return question_range
