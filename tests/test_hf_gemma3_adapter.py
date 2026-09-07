"""Gemma 3 adapter: the CPU-checkable parts, and the model-specific GPU checks.

The model-agnostic invariants run through ``test_adapter_contract`` (the
``hf-gemma3`` probe is marked gpu there). What is left here: the prompt
string, the ``o_proj`` tap that gives the pre-projection attention output a
module of its own, the whole-post-image question span, and -- on the GPU --
the 256-token image block, the tap width the dictionaries expect, the empty
answer prefix, and the knockout reaching both a sliding and a global layer.
"""

import pytest
import torch
from torch import nn

from vlmflowprobe.adapters.hf_gemma3 import ANSWER_SUFFIX, _OProjTap, build_gemma_prompt
from vlmflowprobe.adapters.registry import adapter_class


def test_prompt_is_one_user_turn_with_image_then_generation_prompt():
    prompt = build_gemma_prompt("What color is the circle?")
    assert prompt.startswith("<start_of_turn>user\n<start_of_image>What color is the circle?")
    assert prompt.endswith("<end_of_turn>\n<start_of_turn>model\n")
    assert ANSWER_SUFFIX.strip() in prompt
    assert "<bos>" not in prompt  # the processor adds exactly one
    assert "<start_of_image>" not in build_gemma_prompt("q", with_image=False)


def test_o_proj_tap_is_transparent_and_hookable():
    o_proj = nn.Linear(4, 3, bias=False)
    tapped = _OProjTap(o_proj)
    x = torch.randn(1, 5, 4)
    assert torch.equal(tapped(x), o_proj(x))
    assert tapped.weight is o_proj.weight
    seen = {}

    def zero_the_input(module, inputs, output):
        seen["out"] = output
        return output * 0

    handle = tapped.tap.register_forward_hook(zero_the_input)
    try:
        out = tapped(x)
    finally:
        handle.remove()
    # The hook saw o_proj's input and its rewrite reached o_proj.
    assert torch.equal(seen["out"], x)
    assert torch.equal(out, o_proj(torch.zeros_like(x)))
    assert torch.equal(tapped(x), o_proj(x))


def test_registry_resolves_lazily():
    cls = adapter_class("hf-gemma3")
    assert cls.name == "hf-gemma3"
    assert cls.answer_prefix == ""


# --------------------------------------------------------------------------- gpu


@pytest.mark.gpu
class TestOnTheRealModel:
    @pytest.fixture(scope="class")
    @classmethod
    def probe(cls):
        from tests.probes import cached, hf_gemma3_probe

        return cached(hf_gemma3_probe)

    def test_static_properties(self, probe):
        adapter = probe.adapter
        assert adapter.n_layers == 34
        assert adapter.d_model == 2560
        assert adapter.site_dim(12, "attn_z") == 2048
        assert adapter.dtype == torch.bfloat16

    def test_image_block_and_question_span(self, probe):
        adapter, batch = probe.adapter, probe.batch()
        assert adapter.n_image_tokens(batch) == 256
        span = adapter.image_token_span(batch)
        assert len(span) == 256
        assert int(batch.input_ids[0, 0]) == adapter.tokenizer.bos_token_id
        assert int(batch.input_ids[0, 1]) != adapter.tokenizer.bos_token_id
        assert adapter.question_token_span(batch) == list(range(span.stop, batch.seq_len))
        tail = adapter.tokenizer.decode(batch.input_ids[0, span.stop:])
        assert probe.question in tail and tail.endswith("model\n")

    def test_attn_z_hook_sees_the_o_proj_input(self, probe):
        adapter, batch = probe.adapter, probe.batch()
        seen = {}

        def record(module, inputs, output):
            seen["shape"] = tuple(output.shape)  # return None: do not replace the output

        handle = adapter.layer_module(12, "attn_z").register_forward_hook(record)
        try:
            with torch.inference_mode():
                adapter.forward(batch)
        finally:
            handle.remove()
        assert seen["shape"] == (1, batch.seq_len, 2048)

    def test_answer_form_is_capitalized_without_a_space(self, probe):
        from vlmflowprobe.knockout.scoring import sequence_logprob

        adapter, batch = probe.adapter, probe.batch()
        assert adapter.format_answer("blue") == "Blue"
        formatted = sequence_logprob(adapter, batch, probe.true_answer)
        adapter.answer_prefix = " "
        try:
            spaced = sequence_logprob(adapter, batch, probe.true_answer)
        finally:
            adapter.answer_prefix = ""
        adapter.format_answer = str.strip
        try:
            lower = sequence_logprob(adapter, batch, probe.true_answer)
        finally:
            del adapter.format_answer
        assert formatted > spaced and formatted > lower, (formatted, spaced, lower)
        assert formatted > -1.0, f"the model's own answer form should be near-certain, got {formatted:.3f}"

    @pytest.mark.parametrize("layer", [0, 5])
    def test_knockout_reaches_sliding_and_global_layers(self, probe, layer):
        from vlmflowprobe.knockout.block_config import build_block_config, flow_block_pairs
        from vlmflowprobe.knockout.scoring import sequence_logprob

        adapter, batch = probe.adapter, probe.batch()
        kind = adapter._decoder_layers()[layer].attention_type
        assert kind == ("full_attention" if layer == 5 else "sliding_attention")
        pairs = flow_block_pairs("Image->Question", batch, adapter)
        config = build_block_config(layer, adapter.n_layers, 1, pairs)

        def margin(**kw):
            return (sequence_logprob(adapter, batch, probe.true_answer, **kw)
                    - sequence_logprob(adapter, batch, probe.false_answer, **kw))

        base, blocked = margin(), margin(block_config=config, flow_target="Question")
        assert abs(base - blocked) > 1e-4, (base, blocked)

    def test_cached_features_match_the_pixel_path(self, probe):
        """inputs_embeds built from cached image features reproduce the pixel_values forward."""
        adapter, batch = probe.adapter, probe.batch()
        from vlmflowprobe.adapters.base import ModelBatch

        uncached = ModelBatch(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                              pixel_values=batch.pixel_values, prompt=batch.prompt, question=batch.question,
                              extra={"token_type_ids": batch.extra["token_type_ids"]})
        with torch.inference_mode():
            a = adapter.forward(batch)[0, -1].float()
            b = adapter.forward(uncached)[0, -1].float()
        assert torch.allclose(a, b, atol=0.05, rtol=0.01), float((a - b).abs().max())

    def test_logits_are_float32(self, probe):
        adapter, batch = probe.adapter, probe.batch()
        with torch.inference_mode():
            logits = adapter.forward(batch)
        assert logits.dtype == torch.float32
