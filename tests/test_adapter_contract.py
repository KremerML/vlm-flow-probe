"""Invariants every ModelAdapter must satisfy, checked on the stub.

The same checks run against real adapters under ``-m gpu`` (see
test_hf_llava_adapter.py once Phase 3 lands).
"""

import unittest

import torch

from tests.stubs import CountingModel, adapt
from vlmflowprobe.adapters.base import AdapterContractError
from vlmflowprobe.adapters.registry import adapter_class, create_adapter


class TestLayerModule(unittest.TestCase):
    def setUp(self):
        self.adapter = adapt(CountingModel(d_model=4))

    def test_residual_site_is_the_decoder_layer(self):
        self.assertIs(self.adapter.layer_module(0, "residual"), self.adapter.model.layers[0])

    def test_unknown_site_raises(self):
        # The archive's get_target_module silently fell back to the residual
        # stream on a typo'd site; that is now an error.
        with self.assertRaises(AdapterContractError):
            self.adapter.layer_module(0, "attn_output")

    def test_missing_submodule_raises(self):
        # nn.Linear layers have no self_attn/mlp: asking for those sites must raise.
        with self.assertRaises(AdapterContractError):
            self.adapter.layer_module(0, "attn_out")
        with self.assertRaises(AdapterContractError):
            self.adapter.layer_module(0, "mlp_out")

    def test_out_of_range_layer_raises(self):
        with self.assertRaises(AdapterContractError):
            self.adapter.layer_module(99, "residual")


class TestGeometry(unittest.TestCase):
    def test_indices_are_post_expansion_and_consistent(self):
        adapter = adapt(CountingModel(d_model=4), question_span=(1,), image_span=range(0, 1))
        batch = adapter.build_inputs("what color is it")
        self.assertEqual(batch.seq_len, batch.input_ids.shape[1])
        self.assertEqual(adapter.last_token_index(batch), batch.seq_len - 1)
        self.assertEqual(adapter.answer_start_index(batch), batch.seq_len)
        self.assertEqual(adapter.n_image_tokens(batch), 1)
        span = adapter.image_token_span(batch)
        self.assertEqual(len(span), adapter.n_image_tokens(batch))
        for pos in list(span) + adapter.question_token_span(batch):
            self.assertTrue(0 <= pos < batch.seq_len)

    def test_forward_appends_extra_input_ids(self):
        adapter = adapt(CountingModel(d_model=4, vocab=16))
        batch = adapter.build_inputs("q")
        extra = torch.tensor([[1, 1]], dtype=torch.long)
        logits = adapter.forward(batch, extra_input_ids=extra)
        self.assertEqual(logits.shape[1], batch.seq_len + 2)


class TestKnockoutPairing(unittest.TestCase):
    def test_context_manager_removes_on_success_and_failure(self):
        adapter = adapt(CountingModel(d_model=4))
        with adapter.attention_knockout({0: [(1, 0)]}):
            pass
        with self.assertRaises(ValueError):
            with adapter.attention_knockout({0: [(1, 0)]}):
                raise ValueError("boom")
        adapter.assert_knockouts_balanced()
        self.assertEqual(sum(1 for e in adapter.knockout_log if e[0] == "install"), 2)


class TestRegistry(unittest.TestCase):
    def test_stub_is_registered(self):
        self.assertIsNotNone(adapter_class("stub"))

    def test_unknown_adapter_raises_with_known_list(self):
        with self.assertRaises(KeyError):
            adapter_class("no-such-adapter")

    def test_create_adapter_requires_key(self):
        with self.assertRaises(KeyError):
            create_adapter({"name": "some/model"})


if __name__ == "__main__":
    unittest.main()
