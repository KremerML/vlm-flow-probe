"""Flow grammar and block-config builders for attention knockout.

A *flow* is a string ``"Source->Target"`` naming an attention pathway to
sever; supported sources: ``Image``; targets: ``Question``, ``Last``. A
*block config* maps layer index to ``(target, source)`` position pairs, in
post-expansion coordinates, and is what
``ModelAdapter.install_attention_knockout`` consumes.
"""

from typing import Dict, Iterable, List, Tuple

from vlmflowprobe.adapters.base import ModelAdapter, ModelBatch

BlockConfig = Dict[int, List[Tuple[int, int]]]


def flow_target(flow: str) -> str:
    return flow.split("->")[-1].strip()


def resolve_flow_ranges(
    flow: str, batch: ModelBatch, adapter: ModelAdapter
) -> Tuple[List[int], List[int]]:
    """(source positions, target positions) for a flow on one sample."""
    source, target = [part.strip() for part in flow.split("->")]
    if source == "Image":
        source_range = list(adapter.image_token_span(batch))
    else:
        raise ValueError(f"Unsupported source flow: {source}")

    if target == "Question":
        target_range = adapter.question_token_span(batch)
    elif target == "Last":
        target_range = [adapter.last_token_index(batch)]
    else:
        raise ValueError(f"Unsupported target flow: {target}")

    if not source_range or not target_range:
        return [], []
    return source_range, target_range


def flow_block_pairs(flow: str, batch: ModelBatch, adapter: ModelAdapter) -> List[Tuple[int, int]]:
    """The (target, source) pairs blocking a flow on one sample."""
    source_range, target_range = resolve_flow_ranges(flow, batch, adapter)
    return [(tgt, src) for src in source_range for tgt in target_range]


def build_block_config(
    layer: int,
    num_layers: int,
    window: int,
    src_tgt_pairs: List[Tuple[int, int]],
) -> BlockConfig:
    if window <= 1:
        return {layer: src_tgt_pairs}
    half = window // 2
    layerlist = list(range(max(0, layer - half), min(num_layers, layer + half + 1)))
    return {l: list(src_tgt_pairs) for l in layerlist}


def build_block_config_for_layers(
    layers: Iterable[int],
    src_tgt_pairs: List[Tuple[int, int]],
) -> BlockConfig:
    """Block the same (target, source) pairs at an explicit set of layers.

    ``build_block_config`` can only express a symmetric window centred on one
    layer, so it cannot say "layers 12..31" (the downstream-knockout arm) or
    "{10, 12, 14}" (the non-nested spans that separate span size from span
    depth). This takes the layer set directly.
    """
    return {int(layer): list(src_tgt_pairs) for layer in sorted(set(int(l) for l in layers))}
