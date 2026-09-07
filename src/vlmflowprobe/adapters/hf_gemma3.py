"""HF-native Gemma 3 adapter (``google/gemma-3-4b-it`` and siblings).

Prompt construction uses the model's own chat template: one user turn holding
the image followed by the question and the single-word-answer suffix, then the
generation prompt. The Gemma 3 processor materializes the 256 image soft
tokens directly in ``input_ids`` (``\\n\\n<start_of_image>`` + 256 x
``<image_soft_token>`` + ``<end_of_image>\\n\\n``), so all geometry is
post-expansion for free.

Three things differ from the LLaVA adapter and are worth knowing:

* **Question span.** Every text position after the image block, through the
  final token. That is also what the published LLaVA runs used in effect
  (``tests/test_hf_llava_adapter.py`` documents the sublist match never hit),
  and it is the span that makes ``Image->Question`` knockout a complete cut of
  cross-modal transfer at a layer: no text position downstream of the image
  can read it there. Matching the question tokens alone would leave the
  suffix and the generation prompt free to read the image.
* **``attn_z`` site.** Gemma Scope 2's "attn_out" dictionaries read the
  *input* of ``o_proj`` -- the concatenated head outputs, width
  ``n_heads * head_dim`` (2048 for 4B), not ``d_model``. ``load`` wraps every
  ``o_proj`` in :class:`_OProjTap`, an identity module in front of it, and
  ``layer_module(layer, "attn_z")`` returns that tap so the package's
  output-rewriting forward hooks land on exactly the tensor the dictionaries
  were trained on. ``attn_out`` keeps its usual meaning (post-``o_proj``).
* **Answer form.** The generation prompt ends in ``model\\n``, so an answer
  continues without a leading space (``answer_prefix`` is ``""``; LLaVA's
  ``ASSISTANT:`` needs the space), and the IT model capitalizes its one-word
  answers: on the first CLEVR-Lite validation items it generated ``Blue``,
  ``Square``, ``Yellow`` without exception, scoring ``Blue`` at 0.00 nats and
  ``blue`` at -16.75. ``format_answer`` therefore capitalizes the first letter
  of every option before it is scored.

Two measurement conveniences, both on by default in ``adapter_options``:

* ``cache_image_features`` -- ``build_inputs`` runs the vision tower once and
  keeps the projected image features on the batch; ``forward``/``generate``
  then feed ``inputs_embeds`` (token embeddings with the features scattered
  over the image slots, exactly the model's own path) instead of
  ``pixel_values``. A knockout sweep runs ~140 forwards per sample and the
  image never changes, so this removes the SigLIP pass from all but one.
* ``fp32_logits`` -- the language model runs in bfloat16, whose 8-bit
  mantissa quantizes a logit of magnitude ~30 to steps of 0.125-0.25; every
  margin in the bring-up was such a multiple. ``lm_head`` is replaced by an
  untied float32 copy fed the float32-cast final hidden state, so log-probs
  (and hence margins) are resolved at float32 precision. The decoder itself
  is untouched.

The language model loads with ``attn_implementation="eager"`` (the vision
tower may use SDPA, ``adapter_options.vision_attn``), and ``token_type_ids``
are passed on every forward, so the image block keeps the bidirectional
attention it was trained with. Both the global and the sliding-window layers receive a
materialized 4D mask under eager attention in transformers 4.57, which is what
``knockout.mask_hooks`` edits; ``vfp-verify-adapter`` confirms the edit
reaches the attention computation on both layer kinds.
"""

from typing import Any, List, Optional

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
from vlmflowprobe.knockout.mask_hooks import install_mask_knockout, remove_mask_knockout

ANSWER_SUFFIX = " \nAnswer the question using a single word or phrase."

#: Activation site exposing the o_proj input (concatenated head outputs).
ATTN_Z_SITE = "attn_z"


class _OProjTap(nn.Module):
    """``o_proj`` with an identity module in front of it.

    Forward hooks in this package rewrite a module's *output*; the tensor
    Gemma Scope's attention SAEs read is ``o_proj``'s *input*, which no module
    emits. Wrapping ``o_proj`` gives that tensor a module of its own
    (``tap``) without touching the attention forward.
    """

    def __init__(self, o_proj: nn.Module):
        super().__init__()
        self.tap = nn.Identity()
        self.o_proj = o_proj

    @property
    def weight(self):
        return self.o_proj.weight

    def forward(self, x):
        return self.o_proj(self.tap(x))


class _Fp32Head(nn.Module):
    """An untied float32 ``lm_head``: float32 weight, input cast to float32."""

    def __init__(self, lm_head: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(lm_head.weight.detach().float().clone(), requires_grad=False)

    def forward(self, hidden):
        return nn.functional.linear(hidden.float(), self.weight)


def build_gemma_prompt(question: str, with_image: bool = True) -> str:
    """The chat-templated prompt, without ``<bos>`` (the processor adds it).

    Mirrors ``gemma-3-*-it``'s chat template for one user turn of
    ``[image, text]`` followed by the generation prompt; the template trims
    the text item, so the suffix's trailing punctuation survives intact.
    """
    text = (question + ANSWER_SUFFIX).strip()
    image = "<start_of_image>" if with_image else ""
    return f"<start_of_turn>user\n{image}{text}<end_of_turn>\n<start_of_turn>model\n"


@register("hf-gemma3")
class HFGemma3Adapter(ModelAdapter):
    answer_prefix = ""

    def format_answer(self, answer: str) -> str:
        answer = answer.strip()
        return answer[:1].upper() + answer[1:]

    def __init__(self, model_cfg: Optional[dict] = None):
        super().__init__(model_cfg)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        from vlmflowprobe.utils.torch_env import disable_native_triton_ops

        disable_native_triton_ops()

        name = self.model_cfg.get("name", "google/gemma-3-4b-it")
        dtype = self.model_cfg.get("dtype", "bfloat16")
        device = self.model_cfg.get(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        options = self.model_cfg.get("adapter_options", {}) or {}
        vision_attn = options.get("vision_attn", "sdpa")
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            name,
            torch_dtype=getattr(torch, dtype),
            attn_implementation={"text_config": "eager", "vision_config": vision_attn},
            device_map={"": device},
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(name)
        for layer in self._decoder_layers():
            attn = layer.self_attn
            if not isinstance(attn.o_proj, _OProjTap):
                attn.o_proj = _OProjTap(attn.o_proj)
        self._cache_image_features = bool(options.get("cache_image_features", True))
        if options.get("fp32_logits", True) and not isinstance(self._model.lm_head, _Fp32Head):
            self._model.lm_head = _Fp32Head(self._model.lm_head).to(device)

    def _require_loaded(self):
        if self._model is None:
            raise AdapterContractError("HFGemma3Adapter used before load()")

    # ------------------------------------------------------------------ properties
    @property
    def model(self):
        self._require_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._require_loaded()
        return self._processor.tokenizer

    @property
    def d_model(self) -> int:
        return int(self.model.config.get_text_config().hidden_size)

    @property
    def n_layers(self) -> int:
        return len(self._decoder_layers())

    @property
    def image_token_id(self) -> int:
        config = self.model.config
        token_id = getattr(config, "image_token_id", None)
        if token_id is None:
            token_id = getattr(config, "image_token_index", None)
        if token_id is None:
            raise AdapterContractError("model config has no image token id")
        return int(token_id)

    def site_dim(self, layer: int, site: str) -> int:
        if site == ATTN_Z_SITE:
            o_proj = self._decoder_layers()[int(layer)].self_attn.o_proj
            return int(o_proj.weight.shape[1])
        return self.d_model

    # ------------------------------------------------------------------ inputs
    def build_inputs(self, question: str, image: Any = None) -> ModelBatch:
        self._require_loaded()
        prompt = build_gemma_prompt(question, with_image=image is not None)
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        input_ids = inputs["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and (int(input_ids[0, 0]) != bos or int(input_ids[0, 1]) == bos):
            raise AdapterContractError(
                "expected exactly one <bos> at position 0; the processor's special-token "
                "handling changed"
            )
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.dtype)
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = (input_ids == self.image_token_id).long()
        extra = {"token_type_ids": token_type_ids}
        if pixel_values is not None and self._cache_image_features:
            with torch.no_grad():
                extra["image_features"] = self.model.get_image_features(pixel_values).detach()
        return ModelBatch(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            prompt=prompt,
            question=question,
            extra=extra,
        )

    def _model_inputs(self, batch: ModelBatch, input_ids: torch.Tensor) -> dict:
        """``pixel_values`` for the model, or the cached features scattered into ``inputs_embeds``.

        The scatter is the model's own: token embeddings from
        ``get_input_embeddings`` (Gemma's scaled embedding), image slots
        replaced by the projected features in order.
        """
        features = batch.extra.get("image_features")
        if features is None:
            return {"input_ids": input_ids, "pixel_values": batch.pixel_values}
        embeds = self.model.get_input_embeddings()(input_ids)
        mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(embeds)
        if int(mask[..., 0].sum()) != features.shape[0] * features.shape[1]:
            raise AdapterContractError(
                f"{int(mask[..., 0].sum())} image slots in input_ids but "
                f"{features.shape[0] * features.shape[1]} cached image features"
            )
        embeds = embeds.masked_scatter(mask, features.to(embeds.dtype).reshape(-1, embeds.shape[-1]))
        return {"input_ids": None, "inputs_embeds": embeds, "pixel_values": None}

    # ------------------------------------------------------------------ execution
    def forward(self, batch: ModelBatch, extra_input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        token_type_ids = batch.extra.get("token_type_ids")
        if extra_input_ids is not None:
            extra = extra_input_ids.to(input_ids.device)
            input_ids = torch.cat([input_ids, extra], dim=1)
            if attention_mask is not None:
                ones = torch.ones_like(extra, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, ones.to(attention_mask.device)], dim=1)
            if token_type_ids is not None:
                zeros = torch.zeros_like(extra, dtype=token_type_ids.dtype)
                token_type_ids = torch.cat([token_type_ids, zeros.to(token_type_ids.device)], dim=1)
        outputs = self.model(
            **self._model_inputs(batch, input_ids),
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=False,
        )
        return outputs.logits

    def generate(self, batch: ModelBatch, max_new_tokens: int = 1) -> GenResult:
        model_inputs = self._model_inputs(batch, batch.input_ids)
        if model_inputs.get("inputs_embeds") is not None:
            # input_ids alongside inputs_embeds: HF uses the embeds for the prefill and
            # the ids as the returned prefix, so the slice below stays valid.
            model_inputs["input_ids"] = batch.input_ids
        outputs = self.model.generate(
            **model_inputs,
            attention_mask=batch.attention_mask,
            token_type_ids=batch.extra.get("token_type_ids"),
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        sequences = outputs.sequences
        prompt_len = batch.input_ids.shape[1]
        if sequences.shape[1] > prompt_len and torch.equal(sequences[:, :prompt_len], batch.input_ids):
            sequences = sequences[:, prompt_len:]
        elif sequences.shape[1] > prompt_len:
            raise AdapterContractError("generate returned a sequence that neither is the new tokens "
                                       "nor starts with the prompt")
        return GenResult(sequences=sequences, first_token_scores=outputs.scores[0])

    # ------------------------------------------------------------------ geometry
    def n_image_tokens(self, batch: ModelBatch) -> int:
        count = int((batch.input_ids[0] == self.image_token_id).sum().item())
        if count == 0 and batch.pixel_values is not None:
            raise AdapterContractError(
                "batch has pixel_values but no image tokens in input_ids; "
                "the processor did not expand <start_of_image>"
            )
        return count

    def image_token_span(self, batch: ModelBatch) -> range:
        positions = torch.nonzero(
            batch.input_ids[0] == self.image_token_id, as_tuple=False
        ).flatten().tolist()
        if not positions:
            raise AdapterContractError("no image tokens in batch")
        start, stop = positions[0], positions[-1] + 1
        if positions != list(range(start, stop)):
            raise AdapterContractError("image tokens are not contiguous")
        return range(start, stop)

    def question_token_span(self, batch: ModelBatch) -> List[int]:
        # Every text position after the image block (see the module docstring);
        # the whole sequence when there is no image.
        try:
            image_span = self.image_token_span(batch)
        except AdapterContractError:
            return list(range(batch.seq_len))
        return list(range(image_span.stop, batch.seq_len))

    # ------------------------------------------------------------------ modules
    def _decoder_layers(self):
        model = self.model
        language_model = getattr(getattr(model, "model", model), "language_model", None)
        if language_model is None or not hasattr(language_model, "layers"):
            raise AdapterContractError(
                "cannot locate decoder layers at model.model.language_model.layers; "
                "transformers Gemma 3 layout changed"
            )
        return language_model.layers

    def layer_module(self, layer: int, site: str) -> nn.Module:
        if site == ATTN_Z_SITE:
            layers = self._decoder_layers()
            if not 0 <= int(layer) < len(layers):
                raise AdapterContractError(
                    f"layer {layer} out of range for a {len(layers)}-layer model"
                )
            o_proj = layers[int(layer)].self_attn.o_proj
            if not isinstance(o_proj, _OProjTap):
                raise AdapterContractError(
                    f"layer {layer}: o_proj is not tapped; load() did not run"
                )
            return o_proj.tap
        return super().layer_module(layer, site)

    # ------------------------------------------------------------------ knockout
    def install_attention_knockout(self, block_config, flow_target=None) -> KnockoutHandle:
        return install_mask_knockout(self._decoder_layers(), block_config, flow_target)

    def remove_attention_knockout(self, handle: KnockoutHandle) -> None:
        remove_mask_knockout(handle)
