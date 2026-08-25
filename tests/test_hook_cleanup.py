import unittest
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from vlmflowprobe.ablation.feature_ablator import FeatureAblator
from vlmflowprobe.data.collection import ActivationCollector
from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder


class _DatasetStub:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.questions = [{"q_id": "q1", "attribute_tokens": []}]
        self.dataset_dict = {
            "q1": {
                "question": "what color is it",
                "answer": "yes",
                "true option": "yes",
                "false option": "no",
            }
        }
        self._batch = (
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            [torch.zeros(1, 3, 4, 4)],
            [(4, 4)],
            "",
            torch.zeros(1, 3, 4, 4),
        )

    def create_dataloader(self):
        return [self._batch]


class _TokenizerStub:
    eos_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [1]

    @staticmethod
    def batch_decode(sequences, skip_special_tokens=True):
        return ["yes"]


class _ExplodingCollectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, input_ids=None, images=None, image_sizes=None, use_cache=False):
        x = self.embedding(input_ids)
        _ = self.layers[0](x)
        raise RuntimeError("collector failure")


class _ExplodingAblationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self._gen_calls = 0

    def forward(self, input_ids=None, images=None, image_sizes=None, use_cache=False):
        x = self.embedding(input_ids)
        x = self.layers[0](x)
        logits = torch.zeros(1, x.shape[1], 64, device=x.device, dtype=x.dtype)
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self._gen_calls += 1
        if self._gen_calls >= 2:
            raise RuntimeError("ablation failure")
        return {
            "sequences": torch.tensor([[1]], dtype=torch.long),
            "scores": [torch.zeros(1, 64)],
        }


class _ExplodingKnockoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1)

    def forward(self, input_ids=None, images=None, image_sizes=None, use_cache=False):
        raise RuntimeError("knockout failure")


class TestHookCleanup(unittest.TestCase):
    def test_activation_collector_removes_hook_on_exception(self):
        model = _ExplodingCollectorModel()
        dataset = _DatasetStub(_TokenizerStub())
        collector = ActivationCollector(model, layer_idx=0, activation_site="residual")

        with self.assertRaises(RuntimeError):
            collector.collect_from_dataset(dataset, position_type="question", tokenizer=None, max_samples=1)

        self.assertEqual(len(model.layers[0]._forward_hooks), 0)

    def test_feature_ablator_removes_hook_on_exception(self):
        model = _ExplodingAblationModel()
        sae = SparseAutoencoder(d_model=4, n_features=8)
        ablator = FeatureAblator(model, sae, layer_idx=0, activation_site="residual")
        dataset = _DatasetStub(_TokenizerStub())

        with self.assertRaises(RuntimeError):
            ablator.batch_ablation_experiment(
                dataset,
                feature_indices=[0, 1],
                position_type="question",
                max_samples=1,
            )

        self.assertEqual(len(model.layers[0]._forward_hooks), 0)

    def test_knockout_logprob_removes_attn_hooks_on_exception(self):
        model = _ExplodingKnockoutModel()
        tokenizer = _TokenizerStub()
        called = []

        from vlmflowprobe.knockout import knockout_utils, attention_hooks

        with patch.object(attention_hooks, "set_block_attn_hooks_llava", return_value="hooks"), patch.object(
            attention_hooks,
            "remove_wrapper_llava",
            side_effect=lambda m, h: called.append((m, h)),
        ):
            with self.assertRaises(RuntimeError):
                knockout_utils.sequence_logprob(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                    image_tensor=[torch.zeros(1, 3, 4, 4)],
                    image_sizes=[(4, 4)],
                    answer_text="yes",
                    normalize=True,
                    block_config={0: [(0, 0)]},
                )

        self.assertEqual(len(called), 1)
        self.assertIs(called[0][0], model)
        self.assertEqual(called[0][1], "hooks")


if __name__ == "__main__":
    unittest.main()
