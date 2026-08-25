"""A CPU stub adapter for tests.

Wraps any tiny ``nn.Module`` exposing ``.layers`` (a ModuleList) and
optionally ``.generate``/``.forward`` with the archive test-suite calling
conventions. Token geometry is configured, not computed, so tests can pin
exact positions. Knockout installs are recorded, never applied — tests assert
install/remove pairing and payloads.
"""

from typing import Any, List, Optional, Sequence

import torch
from torch import nn

from vlmflowprobe.adapters.base import (
    AdapterContractError,
    GenResult,
    KnockoutHandle,
    ModelAdapter,
    ModelBatch,
)
from vlmflowprobe.adapters.registry import register


class StubTokenizer:
    eos_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [1]

    @staticmethod
    def batch_decode(sequences, skip_special_tokens=True):
        return ["yes"]


@register("stub")
class StubAdapter(ModelAdapter):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        *,
        seq_tokens: Sequence[int] = (1, 2, 3),
        question_span: Sequence[int] = (1,),
        image_span: range = range(0, 0),
        model_cfg: Optional[dict] = None,
    ):
        super().__init__(model_cfg)
        self._model = model
        self._tokenizer = tokenizer or StubTokenizer()
        self._seq_tokens = list(seq_tokens)
        self._question_span = list(question_span)
        self._image_span = image_span
        #: every install/remove, for assertions: ("install", block_config, flow_target)
        #: and ("remove", handle).
        self.knockout_log: List[tuple] = []
        self._handle_counter = 0

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:  # models are injected pre-built in tests
        if self._model is None:
            raise AdapterContractError("StubAdapter needs a model injected at construction")

    # ------------------------------------------------------------------ properties
    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def d_model(self) -> int:
        return int(self._model.layers[0].in_features)

    @property
    def n_layers(self) -> int:
        return len(self._model.layers)

    @property
    def device(self) -> torch.device:
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self._model.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------ inputs
    def build_inputs(self, question: str, image: Any = None) -> ModelBatch:
        input_ids = torch.tensor([self._seq_tokens], dtype=torch.long, device=self.device)
        return ModelBatch(input_ids=input_ids, prompt=question, question=question)

    # ------------------------------------------------------------------ execution
    def forward(self, batch: ModelBatch, extra_input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        ids = batch.input_ids
        if extra_input_ids is not None:
            ids = torch.cat([ids, extra_input_ids], dim=1)
        out = self._model(input_ids=ids, use_cache=False)
        return out.logits if hasattr(out, "logits") else out

    def generate(self, batch: ModelBatch, max_new_tokens: int = 1) -> GenResult:
        out = self._model.generate(
            inputs=batch.input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        return GenResult(sequences=out["sequences"], first_token_scores=out["scores"][0])

    # ------------------------------------------------------------------ geometry
    def n_image_tokens(self, batch: ModelBatch) -> int:
        return len(self._image_span)

    def image_token_span(self, batch: ModelBatch) -> range:
        return self._image_span

    def question_token_span(self, batch: ModelBatch) -> List[int]:
        return list(self._question_span)

    # ------------------------------------------------------------------ modules
    def _decoder_layers(self):
        return self._model.layers

    # ------------------------------------------------------------------ knockout
    def install_attention_knockout(self, block_config, flow_target=None) -> KnockoutHandle:
        self._handle_counter += 1
        handle = f"knockout-{self._handle_counter}"
        self.knockout_log.append(("install", handle, dict(block_config), flow_target))
        return handle

    def remove_attention_knockout(self, handle: KnockoutHandle) -> None:
        self.knockout_log.append(("remove", handle))

    # ------------------------------------------------------------------ assertions
    def assert_knockouts_balanced(self) -> None:
        installs = {e[1] for e in self.knockout_log if e[0] == "install"}
        removes = {e[1] for e in self.knockout_log if e[0] == "remove"}
        if installs != removes:
            raise AssertionError(
                f"unbalanced knockout install/remove: installed {sorted(installs)}, "
                f"removed {sorted(removes)}"
            )
