"""The model adapter interface.

Everything model-specific — prompt construction, image preprocessing, token
geometry, forward/generate calling conventions, module resolution, attention
knockout mechanics — lives behind ``ModelAdapter``. The rest of the package
sees only this interface plus ``ModelBatch``.

**Coordinate convention.** Every index an adapter exposes is a
**post-expansion language-model sequence index**: ``batch.input_ids`` indices,
hidden-state indices, and logits indices are all the same coordinate system.
Adapters whose processors materialize image placeholder tokens in
``input_ids`` (HF LLaVA, Qwen-VL, Gemma 3) satisfy this for free; an adapter
for a model that expands images internally must translate at its boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


class AdapterContractError(RuntimeError):
    """An adapter could not satisfy its contract.

    Raised instead of degrading silently: the archive code's habit of returning
    ``0`` image tokens on failure corrupted every downstream position
    calculation without a traceback, and this class exists so that can never
    happen again.
    """


@dataclass
class ModelBatch:
    """One sample, fully prepared for the adapter's model.

    ``input_ids`` is ``[1, S]`` in post-expansion coordinates. ``prompt`` is
    the exact string fed to the tokenizer; ``question`` is the raw question
    text before templating (the position resolver matches on it).
    ``extra`` is adapter-private (e.g. image sizes).
    """

    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None
    prompt: str = ""
    question: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


@dataclass
class GenResult:
    """Normalized greedy-generation output.

    ``sequences`` holds only the newly generated token ids (``[1, T]``);
    ``first_token_scores`` is the logits over the vocabulary at the first
    generated position (``[1, V]``).
    """

    sequences: torch.Tensor
    first_token_scores: torch.Tensor


# Whatever install_attention_knockout returns; remove_attention_knockout must
# accept it. Opaque to callers.
KnockoutHandle = Any

# {layer_index: [(target_idx, source_idx), ...]} in post-expansion coordinates.
BlockConfig = Dict[int, List[Tuple[int, int]]]


class ModelAdapter(ABC):
    """Abstract base for model adapters. See the module docstring."""

    #: registry key; subclasses override.
    name: str = "base"
    #: What ``sequence_logprob`` puts between the prompt and an answer. A space
    #: for prompts ending in ``ASSISTANT:``-style labels, empty for prompts
    #: ending in a newline (Gemma's ``model\n``).
    answer_prefix: str = " "

    def format_answer(self, answer: str) -> str:
        """The surface form an answer option takes in this model's output.

        The datasets carry lowercase options; LLaVA answers lowercase, Gemma
        capitalizes (``Blue``, and scores ``blue`` sixteen nats lower). Every
        place an option is tokenized -- ``sequence_logprob``, the causal
        identifier's target logits, the generation-accuracy token -- goes
        through this so the convention is set once, in the adapter.
        """
        return answer.strip()

    def __init__(self, model_cfg: Optional[Dict[str, Any]] = None):
        self.model_cfg: Dict[str, Any] = dict(model_cfg or {})

    # ------------------------------------------------------------------ lifecycle
    @abstractmethod
    def load(self) -> None:
        """Load model + processor, move to device, ``eval()``.

        Must leave the model in whatever attention implementation the
        knockout mechanism requires (eager for the HF adapters).
        """

    # ------------------------------------------------------------------ properties
    @property
    @abstractmethod
    def model(self) -> nn.Module: ...

    @property
    @abstractmethod
    def tokenizer(self) -> Any: ...

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # ------------------------------------------------------------------ inputs
    @abstractmethod
    def build_inputs(self, question: str, image: Any = None) -> ModelBatch:
        """Build a ready-to-run batch. Owns prompt templating and image
        preprocessing; ``image`` is a PIL image, a path, or ``None``."""

    # ------------------------------------------------------------------ execution
    @abstractmethod
    def forward(self, batch: ModelBatch, extra_input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Teacher-forced forward pass, ``use_cache=False``; returns logits
        ``[1, S(+A), V]``. ``extra_input_ids`` (``[1, A]``) are appended after
        ``batch.input_ids`` — the sequence-scoring path."""

    @abstractmethod
    def generate(self, batch: ModelBatch, max_new_tokens: int = 1) -> GenResult:
        """Greedy generation, normalized to :class:`GenResult`."""

    # ------------------------------------------------------------------ token geometry
    @abstractmethod
    def n_image_tokens(self, batch: ModelBatch) -> int:
        """Number of image tokens in the sequence.

        Raises :class:`AdapterContractError` when it cannot determine the
        answer — never returns a guessed 0.
        """

    @abstractmethod
    def image_token_span(self, batch: ModelBatch) -> range:
        """Contiguous range of image-token positions; raises on
        non-contiguous placeholders."""

    @abstractmethod
    def question_token_span(self, batch: ModelBatch) -> List[int]:
        """Positions of ``batch.question``'s tokens in the sequence."""

    def last_token_index(self, batch: ModelBatch) -> int:
        return batch.seq_len - 1

    def answer_start_index(self, batch: ModelBatch) -> int:
        """Position where appended answer tokens begin in a
        :meth:`forward` call with ``extra_input_ids``."""
        return batch.seq_len

    # ------------------------------------------------------------------ modules
    @abstractmethod
    def _decoder_layers(self) -> Sequence[nn.Module]:
        """The model's decoder layers, indexable by layer number."""

    def site_dim(self, layer: int, site: str) -> int:
        """Width of the tensor ``layer_module(layer, site)`` emits.

        ``d_model`` for every site the base class knows; adapters exposing a
        narrower site (Gemma's pre-``o_proj`` ``attn_z``) override it, and the
        SAE loaders check a dictionary's ``d_in`` against this rather than
        against ``d_model``.
        """
        return self.d_model

    def layer_module(self, layer: int, site: str) -> nn.Module:
        """Module to hook for an activation site at a layer.

        ``residual`` → the decoder layer itself (post-layer residual stream),
        ``attn_out`` → its self-attention module, ``mlp_out`` → its MLP.
        Unknown sites and out-of-range layers raise — the archive's silent
        fallback to the residual stream is deliberately gone.
        """
        layers = self._decoder_layers()
        if not 0 <= int(layer) < len(layers):
            raise AdapterContractError(
                f"layer {layer} out of range for a {len(layers)}-layer model"
            )
        decoder_layer = layers[int(layer)]
        if site == "residual":
            return decoder_layer
        if site == "attn_out":
            if not hasattr(decoder_layer, "self_attn"):
                raise AdapterContractError(
                    f"layer {layer} has no self_attn module for site 'attn_out'"
                )
            return decoder_layer.self_attn
        if site == "mlp_out":
            if not hasattr(decoder_layer, "mlp"):
                raise AdapterContractError(
                    f"layer {layer} has no mlp module for site 'mlp_out'"
                )
            return decoder_layer.mlp
        raise AdapterContractError(
            f"unknown activation_site {site!r}; expected residual | attn_out | mlp_out"
        )

    # ------------------------------------------------------------------ knockout
    @abstractmethod
    def install_attention_knockout(
        self, block_config: BlockConfig, flow_target: Optional[str] = None
    ) -> KnockoutHandle:
        """Block attention from source to target positions per layer.

        ``block_config`` maps layer index to ``(target, source)`` pairs —
        target attends-to source is what gets severed. ``flow_target`` names
        the flow's destination (``"Question"``/``"Last"``); adapters use it to
        decide how the block behaves on decode steps past the prefill.
        """

    @abstractmethod
    def remove_attention_knockout(self, handle: KnockoutHandle) -> None: ...

    @contextmanager
    def attention_knockout(self, block_config: BlockConfig, flow_target: Optional[str] = None):
        handle = self.install_attention_knockout(block_config, flow_target=flow_target)
        try:
            yield handle
        finally:
            self.remove_attention_knockout(handle)
