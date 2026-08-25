"""Pin the attention-knockout mechanism against the installed transformers.

Builds a tiny random-weight Llama (the same decoder-layer call contract the
HF-LLaVA language model uses), installs a mask knockout, and asserts the
blocked attention entries are actually zero while everything else is intact.
If a transformers upgrade changes the layer-call contract, this fails on CPU
in seconds instead of silently corrupting a GPU run.
"""

import unittest

import torch

from vlmflowprobe.adapters.base import AdapterContractError
from vlmflowprobe.knockout.mask_hooks import install_mask_knockout, remove_mask_knockout


def make_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


class TestMaskKnockout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_tiny_llama()
        cls.input_ids = torch.arange(1, 9).unsqueeze(0)  # [1, 8]

    def _attentions(self):
        with torch.no_grad():
            out = self.model(self.input_ids, output_attentions=True, use_cache=False)
        return out.attentions, out.logits

    def test_blocked_pairs_are_zero_only_at_hooked_layers(self):
        pairs = [(5, 2), (6, 2)]  # queries 5 and 6 may not attend to key 2
        handles = install_mask_knockout(self.model.model.layers, {1: pairs})
        try:
            attns, _ = self._attentions()
        finally:
            remove_mask_knockout(handles)

        for tgt, src in pairs:
            # hooked layer: attention severed across all heads
            self.assertTrue(torch.all(attns[1][0, :, tgt, src] == 0.0), (tgt, src))
            # unhooked layer: untouched
            self.assertTrue(torch.all(attns[0][0, :, tgt, src] > 0.0), (tgt, src))
        # a neighbouring allowed entry at the hooked layer survives
        self.assertTrue(torch.all(attns[1][0, :, 5, 1] > 0.0))

    def test_logits_change_and_removal_restores_baseline(self):
        _, base_logits = self._attentions()
        handles = install_mask_knockout(self.model.model.layers, {0: [(7, 0)], 1: [(7, 0)]})
        try:
            _, blocked_logits = self._attentions()
        finally:
            remove_mask_knockout(handles)
        _, restored_logits = self._attentions()

        self.assertFalse(torch.allclose(base_logits, blocked_logits))
        self.assertTrue(torch.equal(base_logits, restored_logits))

    def test_out_of_range_layer_raises_and_leaves_no_hooks(self):
        with self.assertRaises(AdapterContractError):
            install_mask_knockout(self.model.model.layers, {0: [(1, 0)], 9: [(1, 0)]})
        for layer in self.model.model.layers:
            self.assertEqual(len(layer._forward_pre_hooks), 0)

    def test_decode_step_behavior(self):
        # q_len == 1 simulates a decode step: pairs are dropped unless the flow
        # targets Last, where they remap to the single query row.
        from vlmflowprobe.knockout.mask_hooks import _make_pre_hook

        mask = torch.zeros(1, 1, 1, 8)
        hidden = torch.zeros(1, 1, 32)

        hook = _make_pre_hook([(5, 2)], flow_target=None)
        _, kwargs = hook(None, (hidden,), {"attention_mask": mask.clone()})
        self.assertTrue(torch.all(kwargs["attention_mask"] == 0.0))

        hook = _make_pre_hook([(5, 2)], flow_target="Last")
        _, kwargs = hook(None, (hidden,), {"attention_mask": mask.clone()})
        self.assertEqual(kwargs["attention_mask"][0, 0, 0, 2].item(), torch.finfo(torch.float32).min)

    def test_sdpa_mask_shape_raises_loudly(self):
        hook_input = {"attention_mask": None}
        from vlmflowprobe.knockout.mask_hooks import _make_pre_hook

        hook = _make_pre_hook([(1, 0)], None)
        with self.assertRaises(AdapterContractError):
            hook(None, (torch.zeros(1, 4, 32),), hook_input)


if __name__ == "__main__":
    unittest.main()
