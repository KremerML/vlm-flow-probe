"""Utilities for attention knockout flow definitions and token ranges."""

from typing import Dict, Iterable, List, Optional, Tuple

import torch

# Legacy LLaVA-repo image placeholder sentinel; dies with the Phase-2 adapter rewiring.
IMAGE_TOKEN_INDEX = -200


def estimate_inputs_embeds_shape(model, input_ids, image_tensor, image_sizes) -> Optional[Tuple[int, int, int]]:
    if not hasattr(model, "prepare_inputs_labels_for_multimodal"):
        return None
    try:
        _, _, _, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor,
            ["image"],
            image_sizes=image_sizes,
        )
        return tuple(inputs_embeds.shape)
    except Exception:
        return None


def estimate_image_token_count(model, input_ids, image_tensor, image_sizes) -> int:
    shape = estimate_inputs_embeds_shape(model, input_ids, image_tensor, image_sizes)
    if shape is None:
        return 0
    return int(shape[1] - (input_ids.shape[-1] - 1))


def get_image_token_range(input_ids: torch.Tensor, inputs_embeds_shape: Tuple[int, int, int]) -> List[int]:
    image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
    image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
    return [x for x in range(image_index, image_index + image_dim)]


def get_question_token_range(
    input_ids: torch.Tensor,
    inputs_embeds_shape: Tuple[int, int, int],
    question_text: str,
    tokenizer,
) -> List[int]:
    image_token_count = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
    from vlmflowprobe.utils.token_utils import get_question_token_range as _impl
    return _impl(input_ids[0], image_token_count, question_text, tokenizer, IMAGE_TOKEN_INDEX)


def sequence_logprob(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    image_tensor,
    image_sizes,
    answer_text: str,
    normalize: bool = True,
    block_config: Optional[Dict[int, List[Tuple[int, int]]]] = None,
) -> Optional[float]:
    if not answer_text:
        return None
    answer_ids = tokenizer.encode(f" {answer_text.strip()}", add_special_tokens=False)
    if not answer_ids:
        return None
    device = input_ids.device
    answer_tensor = torch.tensor([answer_ids], device=device, dtype=input_ids.dtype)
    input_ids_full = torch.cat([input_ids, answer_tensor], dim=1)

    hooks = None
    if block_config:
        from vlmflowprobe.knockout.attention_hooks import set_block_attn_hooks_llava, remove_wrapper_llava
        hooks = set_block_attn_hooks_llava(model, block_config)

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids_full,
                images=image_tensor,
                image_sizes=image_sizes,
                use_cache=False,
            )
    finally:
        if hooks:
            remove_wrapper_llava(model, hooks)

    logits = outputs.logits
    log_probs = torch.log_softmax(logits[0], dim=-1)
    start = input_ids.shape[1]
    inputs_embeds_shape = estimate_inputs_embeds_shape(model, input_ids, image_tensor, image_sizes)
    if inputs_embeds_shape is not None:
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        start = input_ids.shape[1] + image_dim - 1
    token_logps = []
    for i, tok_id in enumerate(answer_ids):
        idx = start + i - 1
        if idx < 0 or idx >= log_probs.shape[0]:
            continue
        token_logps.append(log_probs[idx, tok_id].item())
    if not token_logps:
        return None
    if normalize:
        return float(sum(token_logps) / len(token_logps))
    return float(sum(token_logps))


def get_last_token_range(input_ids: torch.Tensor, inputs_embeds_shape: Tuple[int, int, int]) -> List[int]:
    image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
    ntoks = input_ids.shape[1] + image_dim - 1
    return [ntoks - 1]


def resolve_flow_ranges(
    flow: str,
    input_ids: torch.Tensor,
    inputs_embeds_shape: Tuple[int, int, int],
    question_text: str,
    tokenizer,
) -> Tuple[List[int], List[int]]:
    source, target = [part.strip() for part in flow.split("->")]
    if source == "Image":
        source_range = get_image_token_range(input_ids, inputs_embeds_shape)
    else:
        raise ValueError(f"Unsupported source flow: {source}")

    if target == "Question":
        target_range = get_question_token_range(
            input_ids, inputs_embeds_shape, question_text, tokenizer
        )
    elif target == "Last":
        target_range = get_last_token_range(input_ids, inputs_embeds_shape)
    else:
        raise ValueError(f"Unsupported target flow: {target}")

    if not source_range or not target_range:
        return [], []

    return source_range, target_range


def build_block_config(
    layer: int,
    num_layers: int,
    window: int,
    src_tgt_pairs: List[Tuple[int, int]],
) -> Dict[int, List[Tuple[int, int]]]:
    if window <= 1:
        return {layer: src_tgt_pairs}
    half = window // 2
    layerlist = list(range(max(0, layer - half), min(num_layers, layer + half + 1)))
    return {l: list(src_tgt_pairs) for l in layerlist}


def build_block_config_for_layers(
    layers: Iterable[int],
    src_tgt_pairs: List[Tuple[int, int]],
) -> Dict[int, List[Tuple[int, int]]]:
    """Block the same (target, source) pairs at an explicit set of layers.

    ``build_block_config`` can only express a symmetric window centred on one layer, so it
    cannot say "layers 12..31" (the downstream-knockout arm) or "{10, 12, 14}" (the
    non-nested spans that separate span size from span depth). This takes the layer set
    directly.
    """
    return {int(layer): list(src_tgt_pairs) for layer in sorted(set(int(l) for l in layers))}
