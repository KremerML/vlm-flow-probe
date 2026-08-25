"""Attention knockout via decoder-layer mask editing.

The mechanism (verified against transformers 4.57.x): with
``attn_implementation="eager"``, each decoder layer is called with a
materialized 4D additive attention mask as a keyword argument. A
``forward_pre_hook(with_kwargs=True)`` on the layer can therefore rewrite
that mask per layer — no monkeypatching, SDPA-safe by refusal (the hook
raises loudly if the expected mask shape is not there, which is what happens
under SDPA/FlashAttention where the mask may be ``None``).

Blocked ``(target, source)`` entries are **set** to the dtype minimum, not
added to it — adding to an already-minimal entry overflows fp16.
"""

from typing import List, Optional, Sequence, Tuple

import torch

from vlmflowprobe.adapters.base import AdapterContractError, BlockConfig


def install_mask_knockout(
    decoder_layers: Sequence[torch.nn.Module],
    block_config: BlockConfig,
    flow_target: Optional[str] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register mask-editing pre-hooks per layer in ``block_config``.

    Returns the hook handles; the caller removes them (or uses
    ``ModelAdapter.attention_knockout`` which guarantees removal).
    """
    handles = []
    for layer_idx, pairs in block_config.items():
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < len(decoder_layers):
            for h in handles:
                h.remove()
            raise AdapterContractError(
                f"block_config names layer {layer_idx} but the model has "
                f"{len(decoder_layers)} layers"
            )
        handles.append(
            decoder_layers[layer_idx].register_forward_pre_hook(
                _make_pre_hook(list(pairs), flow_target), with_kwargs=True
            )
        )
    return handles


def remove_mask_knockout(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def _make_pre_hook(pairs: List[Tuple[int, int]], flow_target: Optional[str]):
    def pre_hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.dim() != 4:
            raise AdapterContractError(
                "attention knockout expected a materialized 4D attention mask at the "
                "decoder layer; is the model loaded with attn_implementation='eager'?"
            )
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            if not args:
                raise AdapterContractError(
                    "decoder layer called without hidden_states; transformers "
                    "layer-call contract changed"
                )
            hidden = args[0]
        q_len = hidden.shape[1]

        active = pairs
        if q_len == 1:
            # Decode step past the prefill: the single query row is the newest
            # token. Knockout persists only for *->Last flows, where that row
            # is the flow's destination; other flows' targets are already in
            # the KV cache and out of reach.
            if flow_target == "Last":
                active = [(0, src) for _, src in pairs]
            else:
                active = []

        if active:
            mask = mask.clone()
            tgt = [t for t, _ in active]
            src = [s for _, s in active]
            mask[:, :, tgt, src] = torch.finfo(mask.dtype).min
            kwargs["attention_mask"] = mask
        return args, kwargs

    return pre_hook
