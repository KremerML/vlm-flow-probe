"""HF-native LLaVA-1.6 / LLaVA-NeXT adapter (``llava-hf/llava-v1.6-vicuna-7b-hf``).

The same language model as :mod:`~vlmflowprobe.adapters.hf_llava`, asked the
same way, looking through different eyes. Everything the replication holds
fixed is inherited -- the vicuna_v1 prompt (``build_prompt``), the
``input_ids``-derived geometry, the question span, the answer form, the
knockout install -- and what changes is the vision front-end.

**AnyRes.** LLaVA-1.6 does not send one 336x336 crop through the vision tower.
It picks a grid from ``config.image_grid_pinpoints``, tiles the image at higher
resolution, and concatenates a base (global) view with the unpadded tiles,
separating tile rows with a learned ``image_newline`` vector. For a 224x224
CLEVR-Lite image that is 576 base + 576 high-res + 24 newline slots = 1176
image tokens, against 576 for LLaVA-1.5, so the prompt roughly doubles to
~1230 tokens. Nothing here hardcodes those numbers: the count is whatever the
processor materializes, and the checks below tie every other quantity to it.

**No ``expand2square``.** AnyRes owns the geometry now. Padding a square image
would still be a no-op, but padding a non-square one would change which grid
pinpoint the processor selects and hence the token count, so
``adapter_options.pad_to_square`` is refused outright rather than silently
ignored.

**``image_sizes``.** The unpadding step needs the original image size, so
``pixel_values`` alone no longer determines the image features; ``image_sizes``
travels on ``ModelBatch.extra`` and is passed on every pixel-path call.

**Image-feature cache** (``adapter_options.cache_image_features``, default on).
A knockout sweep runs tens of forwards per sample over an unchanging image, and
AnyRes makes the vision pass five tiles' worth of work instead of one.
``build_inputs`` runs it once and keeps the packed ``[N, d]`` features -- the
newline vectors already in place -- and later calls scatter them into
``inputs_embeds`` over the image slots, which is exactly the path the model's
own ``forward`` takes.

Loads with ``attn_implementation="eager"``: the knockout mechanism
(``knockout.mask_hooks``) needs the materialized 4D attention mask, and the
decoder is the same Llama stack that mechanism is pinned against.
"""

from typing import Any, Optional

import torch

from vlmflowprobe.adapters.base import AdapterContractError, GenResult, ModelBatch
from vlmflowprobe.adapters.hf_llava import HFLlavaAdapter
from vlmflowprobe.adapters.registry import register


@register("hf-llava-next")
class HFLlavaNextAdapter(HFLlavaAdapter):
    def __init__(self, model_cfg: Optional[dict] = None):
        super().__init__(model_cfg)
        self._cache_image_features = True
        self._vision_feature_layer = None
        self._vision_feature_select_strategy = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        from vlmflowprobe.utils.torch_env import disable_native_triton_ops

        disable_native_triton_ops()

        options = self.model_cfg.get("adapter_options", {}) or {}
        if options.get("pad_to_square"):
            raise AdapterContractError(
                "adapter_options.pad_to_square is not supported by hf-llava-next: "
                "AnyRes selects a tiling from the image's own size, so padding it "
                "would change the image token count"
            )

        name = self.model_cfg.get("name", "llava-hf/llava-v1.6-vicuna-7b-hf")
        dtype = self.model_cfg.get("dtype", "float16")
        device = self.model_cfg.get(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            name,
            torch_dtype=getattr(torch, dtype),
            attn_implementation="eager",
            device_map={"": device},
        )
        self._model.eval()
        self._processor = LlavaNextProcessor.from_pretrained(name)
        # Resolve the layout now: a changed one is a load-time failure, not a
        # mid-sweep one.
        self._decoder_layers()
        self._cache_image_features = bool(options.get("cache_image_features", True))
        config = self._model.config
        self._vision_feature_layer = config.vision_feature_layer
        self._vision_feature_select_strategy = config.vision_feature_select_strategy

    # ------------------------------------------------------------------ inputs
    def build_inputs(self, question: str, image: Any = None) -> ModelBatch:
        self._require_loaded()
        prompt = self.build_prompt(question, with_image=image is not None)
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.dtype)
        image_sizes = inputs.get("image_sizes")
        if pixel_values is not None and image_sizes is None:
            raise AdapterContractError(
                "processor returned pixel_values without image_sizes; AnyRes "
                "unpadding cannot be reproduced without them"
            )

        extra = {}
        if image_sizes is not None:
            extra["image_sizes"] = image_sizes
        if pixel_values is not None and self._cache_image_features:
            extra["image_features"] = self._image_features(pixel_values, image_sizes)

        return ModelBatch(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            prompt=prompt,
            question=question,
            extra=extra,
        )

    def _image_features(self, pixel_values: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        """The projected image features, packed as the model packs them: ``[N, d]``.

        ``get_image_features`` returns one tensor per image, already unpadded
        and already carrying the ``image_newline`` vectors, which the model's
        own forward concatenates before scattering. Same call, same arguments,
        same result -- only once per sample instead of once per forward.
        """
        with torch.no_grad():
            features = self.model.model.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=self._vision_feature_layer,
                vision_feature_select_strategy=self._vision_feature_select_strategy,
            )
        return torch.cat(features, dim=0).detach()

    def _model_inputs(self, batch: ModelBatch, input_ids: torch.Tensor) -> dict:
        """The pixel path, or the cached features scattered into ``inputs_embeds``.

        The scatter is the model's own: token embeddings, image slots replaced
        by the packed features in order. The count check is the contract --
        one cached feature row per image token, or the positions everything
        downstream is indexed by would be wrong.
        """
        features = batch.extra.get("image_features")
        if features is None:
            return {
                "input_ids": input_ids,
                "pixel_values": batch.pixel_values,
                "image_sizes": batch.extra.get("image_sizes"),
            }
        embeds = self.model.get_input_embeddings()(input_ids)
        mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(embeds)
        n_slots = int(mask[..., 0].sum())
        if n_slots != int(features.shape[0]):
            raise AdapterContractError(
                f"{n_slots} image slots in input_ids but {int(features.shape[0])} "
                "cached image features"
            )
        embeds = embeds.masked_scatter(mask, features.to(embeds.dtype))
        return {"input_ids": None, "inputs_embeds": embeds, "pixel_values": None}

    # ------------------------------------------------------------------ execution
    def forward(self, batch: ModelBatch, extra_input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        if extra_input_ids is not None:
            extra = extra_input_ids.to(input_ids.device)
            input_ids = torch.cat([input_ids, extra], dim=1)
            if attention_mask is not None:
                ones = torch.ones_like(extra, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, ones.to(attention_mask.device)], dim=1)
        outputs = self.model(
            **self._model_inputs(batch, input_ids),
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits

    def generate(self, batch: ModelBatch, max_new_tokens: int = 1) -> GenResult:
        model_inputs = self._model_inputs(batch, batch.input_ids)
        if model_inputs.get("inputs_embeds") is not None:
            # input_ids alongside inputs_embeds: HF uses the embeds for the prefill
            # and the ids as the returned prefix, so the slice below stays valid.
            model_inputs["input_ids"] = batch.input_ids
        outputs = self.model.generate(
            **model_inputs,
            attention_mask=batch.attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        sequences = outputs.sequences
        prompt_len = batch.input_ids.shape[1]
        if sequences.shape[1] > prompt_len and torch.equal(sequences[:, :prompt_len], batch.input_ids):
            sequences = sequences[:, prompt_len:]
        elif sequences.shape[1] > prompt_len:
            raise AdapterContractError(
                "generate returned a sequence that neither is the new tokens nor "
                "starts with the prompt"
            )
        return GenResult(sequences=sequences, first_token_scores=outputs.scores[0])
