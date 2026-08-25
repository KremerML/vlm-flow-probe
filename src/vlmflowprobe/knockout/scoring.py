"""Teacher-forced sequence scoring, model-agnostic.

``sequence_logprob`` is the primitive behind every margin in the project:
``margin = sequence_logprob(true option) − sequence_logprob(false option)``.

Two conventions are load-bearing and deliberately preserved from the archive:

* the answer is encoded as ``" " + answer.strip()`` with
  ``add_special_tokens=False`` (the leading space keeps BPE tokenization
  consistent with the answer's in-context form);
* the logit at position ``p`` predicts token ``p+1``, so answer token ``i``
  is scored at index ``answer_start + i − 1``.
"""

from typing import Optional

import torch

from vlmflowprobe.adapters.base import BlockConfig, ModelAdapter, ModelBatch


def sequence_logprob(
    adapter: ModelAdapter,
    batch: ModelBatch,
    answer_text: str,
    normalize: bool = True,
    block_config: Optional[BlockConfig] = None,
    flow_target: Optional[str] = None,
) -> Optional[float]:
    """Mean (or summed) log-probability of ``answer_text`` appended to the batch.

    Handles multi-token answers via one teacher-forced forward pass. With
    ``block_config`` the pass runs under attention knockout, installed and
    removed around the single forward.
    """
    if not answer_text:
        return None
    answer_ids = adapter.tokenizer.encode(f" {answer_text.strip()}", add_special_tokens=False)
    if not answer_ids:
        return None
    answer_tensor = torch.tensor(
        [answer_ids], device=batch.input_ids.device, dtype=batch.input_ids.dtype
    )

    if block_config:
        with adapter.attention_knockout(block_config, flow_target=flow_target):
            with torch.inference_mode():
                logits = adapter.forward(batch, extra_input_ids=answer_tensor)
    else:
        with torch.inference_mode():
            logits = adapter.forward(batch, extra_input_ids=answer_tensor)

    log_probs = torch.log_softmax(logits[0], dim=-1)
    start = adapter.answer_start_index(batch)
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
