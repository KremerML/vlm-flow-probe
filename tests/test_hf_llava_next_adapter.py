"""LLaVA-1.6 adapter: the CPU-checkable parts, and the model-specific GPU checks.

The model-agnostic invariants run through ``test_adapter_contract`` (the
``hf-llava-next`` probe is marked gpu there). What is left here: that the
prompt is byte-identical to LLaVA-1.5's, that ``pad_to_square`` is refused
rather than ignored, and -- on the GPU -- the AnyRes image block (1176 tokens
on a 224x224 CLEVR-Lite image), the question span, the image-feature cache
against the pixel path, and knockout leaving no hooks behind.
"""

import pytest
import torch

from vlmflowprobe.adapters.base import AdapterContractError
from vlmflowprobe.adapters.hf_llava import HFLlavaAdapter
from vlmflowprobe.adapters.hf_llava_next import HFLlavaNextAdapter
from vlmflowprobe.adapters.registry import adapter_class


def test_prompt_is_byte_identical_to_llava_15():
    """The replication holds the prompt fixed; only the vision front-end differs."""
    question = "What color is the circle?"
    assert (
        HFLlavaNextAdapter({}).build_prompt(question)
        == HFLlavaAdapter({}).build_prompt(question)
    )
    prompt = HFLlavaNextAdapter({}).build_prompt(question)
    assert prompt.startswith("A chat between a curious user")
    assert "USER: <image>\nWhat color is the circle?" in prompt
    assert prompt.endswith("ASSISTANT:")
    assert "<image>" not in HFLlavaNextAdapter({}).build_prompt(question, with_image=False)


def test_registry_resolves_lazily():
    cls = adapter_class("hf-llava-next")
    assert cls.name == "hf-llava-next"
    # Inherited, and the replication depends on both staying LLaVA-1.5's.
    assert cls.answer_prefix == " "
    assert cls({}).format_answer(" Blue ") == "Blue"


def test_pad_to_square_is_refused_not_ignored():
    """AnyRes picks its tiling from the image's own size: padding changes the token count."""
    adapter = HFLlavaNextAdapter({"adapter_options": {"pad_to_square": True}})
    with pytest.raises(AdapterContractError, match="pad_to_square"):
        adapter.load()


# --------------------------------------------------------------------------- gpu


@pytest.mark.gpu
class TestOnTheRealModel:
    @pytest.fixture(scope="class")
    @classmethod
    def probe(cls):
        from tests.probes import cached, hf_llava_next_probe

        return cached(hf_llava_next_probe)

    def test_static_properties(self, probe):
        """The decoder is LLaVA-1.5's, which is the point of this replication."""
        adapter = probe.adapter
        assert adapter.n_layers == 32
        assert adapter.d_model == 4096
        assert adapter.site_dim(11, "attn_out") == 4096
        assert adapter.dtype == torch.float16

    def test_anyres_image_block_is_1176_tokens(self, probe):
        """576 base + 576 unpadded high-res + 24 image_newline slots, contiguous."""
        adapter, batch = probe.adapter, probe.batch()
        assert adapter.n_image_tokens(batch) == 1176
        span = adapter.image_token_span(batch)
        assert len(span) == 1176
        assert span == range(span.start, span.start + 1176)
        assert batch.extra["image_features"].shape[0] == 1176

    def test_question_span_is_everything_after_the_image_block(self, probe):
        adapter, batch = probe.adapter, probe.batch()
        span = adapter.image_token_span(batch)
        assert adapter.question_token_span(batch) == list(range(span.stop, batch.seq_len))
        tail = adapter.tokenizer.decode(batch.input_ids[0, span.stop:])
        assert probe.question in tail and tail.endswith("ASSISTANT:")

    def test_cached_features_match_the_pixel_path(self, probe):
        """inputs_embeds built from cached image features reproduce the pixel_values forward."""
        from vlmflowprobe.adapters.base import ModelBatch

        adapter, batch = probe.adapter, probe.batch()
        uncached = ModelBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            prompt=batch.prompt,
            question=batch.question,
            extra={"image_sizes": batch.extra["image_sizes"]},
        )
        with torch.inference_mode():
            a = adapter.forward(batch)[0, -1].float()
            b = adapter.forward(uncached)[0, -1].float()
        assert torch.allclose(a, b, atol=0.05, rtol=0.01), float((a - b).abs().max())

    def test_a_wrong_feature_count_raises(self, probe):
        """The count check is what keeps every downstream position honest."""
        from vlmflowprobe.adapters.base import ModelBatch

        adapter, batch = probe.adapter, probe.batch()
        truncated = ModelBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            prompt=batch.prompt,
            question=batch.question,
            extra={"image_features": batch.extra["image_features"][:-1]},
        )
        with pytest.raises(AdapterContractError, match="cached image features"):
            adapter.forward(truncated)

    def test_knockout_moves_the_margin_and_leaves_no_hooks(self, probe):
        from vlmflowprobe.knockout.block_config import build_block_config, flow_block_pairs
        from vlmflowprobe.knockout.scoring import sequence_logprob

        adapter, batch = probe.adapter, probe.batch()
        pairs = flow_block_pairs("Image->Question", batch, adapter)
        config = build_block_config(11, adapter.n_layers, 1, pairs)

        def margin(**kw):
            return (sequence_logprob(adapter, batch, probe.true_answer, **kw)
                    - sequence_logprob(adapter, batch, probe.false_answer, **kw))

        base = margin()
        blocked = margin(block_config=config, flow_target="Question")
        assert abs(base - blocked) > 1e-4, (base, blocked)
        assert margin() == pytest.approx(base), "knockout left state behind"
        for layer in adapter._decoder_layers():
            assert not layer._forward_pre_hooks, "a mask hook survived"
