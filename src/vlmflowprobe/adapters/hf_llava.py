"""HF-native LLaVA-1.5 adapter (`llava-hf/llava-1.5-7b-hf`).

Prompt construction reproduces the archive's ``llava_loader.py`` byte for
byte: the vicuna_v1 template applied by hand (no HF chat template), with the
single-word-answer suffix. The llava-hf processor materializes the 576 image
tokens directly in ``input_ids``, so all geometry is post-expansion for free —
the archive's dry ``prepare_inputs_labels_for_multimodal`` pass has no
equivalent here because nothing needs it.

Loads with ``attn_implementation="eager"`` — the knockout mechanism
(``knockout.mask_hooks``) requires the materialized 4D attention mask.
"""

from typing import Any, List, Optional

import torch

from vlmflowprobe.adapters.base import (
    AdapterContractError,
    GenResult,
    KnockoutHandle,
    ModelAdapter,
    ModelBatch,
)
from vlmflowprobe.adapters.registry import register
from vlmflowprobe.knockout.mask_hooks import install_mask_knockout, remove_mask_knockout
from vlmflowprobe.utils.token_utils import _find_sublist

VICUNA_V1_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
ANSWER_SUFFIX = " \nAnswer the question using a single word or phrase."


def expand2square(image, background_color):
    """Pad a PIL image to square with the given background (CLIP-mean) color.

    Mirrors LLaVA's ``image_aspect_ratio: pad`` preprocessing; a no-op for
    square inputs (all of CLEVR-Lite).
    """
    width, height = image.size
    if width == height:
        return image
    from PIL import Image

    side = max(width, height)
    result = Image.new(image.mode, (side, side), background_color)
    if width > height:
        result.paste(image, (0, (side - height) // 2))
    else:
        result.paste(image, ((side - width) // 2, 0))
    return result


@register("hf-llava")
class HFLlavaAdapter(ModelAdapter):
    def __init__(self, model_cfg: Optional[dict] = None):
        super().__init__(model_cfg)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        from vlmflowprobe.utils.torch_env import disable_native_triton_ops

        disable_native_triton_ops()

        name = self.model_cfg.get("name", "llava-hf/llava-1.5-7b-hf")
        dtype = self.model_cfg.get("dtype", "float16")
        device = self.model_cfg.get(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self._model = LlavaForConditionalGeneration.from_pretrained(
            name,
            torch_dtype=getattr(torch, dtype),
            attn_implementation="eager",
            device_map={"": device},
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(name)

    def _require_loaded(self):
        if self._model is None:
            raise AdapterContractError(f"{type(self).__name__} used before load()")

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
        return int(self.model.config.text_config.hidden_size)

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

    # ------------------------------------------------------------------ inputs
    def build_prompt(self, question: str, with_image: bool = True) -> str:
        """The vicuna_v1 prompt, byte-identical to the archive's ``llava_loader.py``.

        A method rather than an inline string because LLaVA-1.6 shares it: the
        two models differ in their vision front-end, not in how they are asked.
        """
        suffixed = question + ANSWER_SUFFIX
        if with_image:
            return f"{VICUNA_V1_SYSTEM} USER: <image>\n{suffixed} ASSISTANT:"
        return f"{VICUNA_V1_SYSTEM} USER: {suffixed} ASSISTANT:"

    def build_inputs(self, question: str, image: Any = None) -> ModelBatch:
        self._require_loaded()
        prompt = self.build_prompt(question, with_image=image is not None)

        adapter_options = self.model_cfg.get("adapter_options", {})
        if image is not None and adapter_options.get("pad_to_square", True):
            image_mean = self._processor.image_processor.image_mean
            image = expand2square(image, tuple(int(x * 255) for x in image_mean))

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.dtype)
        return ModelBatch(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            prompt=prompt,
            question=question,
        )

    # ------------------------------------------------------------------ execution
    def forward(self, batch: ModelBatch, extra_input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        if extra_input_ids is not None:
            input_ids = torch.cat([input_ids, extra_input_ids.to(input_ids.device)], dim=1)
            if attention_mask is not None:
                ones = torch.ones_like(extra_input_ids, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, ones.to(attention_mask.device)], dim=1)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=batch.pixel_values,
            use_cache=False,
        )
        return outputs.logits

    def generate(self, batch: ModelBatch, max_new_tokens: int = 1) -> GenResult:
        outputs = self.model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # HF sequences include the prompt; normalize to new tokens only.
        new_tokens = outputs.sequences[:, batch.input_ids.shape[1]:]
        return GenResult(sequences=new_tokens, first_token_scores=outputs.scores[0])

    # ------------------------------------------------------------------ geometry
    def n_image_tokens(self, batch: ModelBatch) -> int:
        count = int((batch.input_ids[0] == self.image_token_id).sum().item())
        if count == 0 and batch.pixel_values is not None:
            raise AdapterContractError(
                "batch has pixel_values but no image tokens in input_ids; "
                "the processor did not expand the <image> placeholder"
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
        ids = batch.input_ids[0].tolist()
        question_ids = self.tokenizer.encode(batch.question, add_special_tokens=False)
        if question_ids:
            match_start = _find_sublist(ids, question_ids)
            if match_start is not None:
                return list(range(match_start, match_start + len(question_ids)))
        # Fallback mirrors the archive: everything after the image block through
        # the final token (pre-expansion `end = len(ids) - 1 + count` is exactly
        # the post-expansion sequence length); the whole sequence with no image.
        try:
            image_span = self.image_token_span(batch)
        except AdapterContractError:
            return list(range(len(ids)))
        return list(range(image_span.stop, len(ids)))

    # ------------------------------------------------------------------ modules
    def _decoder_layers(self):
        model = self.model
        language_model = getattr(getattr(model, "model", model), "language_model", None)
        if language_model is None or not hasattr(language_model, "layers"):
            raise AdapterContractError(
                "cannot locate decoder layers at model.model.language_model.layers; "
                "transformers LLaVA layout changed"
            )
        return language_model.layers

    # ------------------------------------------------------------------ knockout
    def install_attention_knockout(self, block_config, flow_target=None) -> KnockoutHandle:
        return install_mask_knockout(self._decoder_layers(), block_config, flow_target)

    def remove_attention_knockout(self, handle: KnockoutHandle) -> None:
        remove_mask_knockout(handle)
