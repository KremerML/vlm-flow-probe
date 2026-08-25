"""Interventions must clean up their hooks when the forward pass raises.

The three failure surfaces: the activation collector's capture hook, the
feature ablator's SAE hooks, and the attention-knockout install around
sequence scoring. Each test forces an exception mid-flight and asserts the
model is left unhooked (or the knockout removed) afterwards.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from tests.stubs import DatasetStub, adapt
from vlmflowprobe.ablation.feature_ablator import FeatureAblator
from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder
from vlmflowprobe.data.collection import ActivationCollector
from vlmflowprobe.knockout.scoring import sequence_logprob


class _ExplodingCollectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        x = self.embedding(input_ids)
        _ = self.layers[0](x)
        raise RuntimeError("collector failure")


class _ExplodingAblationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self._gen_calls = 0

    def forward(self, input_ids=None, use_cache=False, **kwargs):
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
        self.layers = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        raise RuntimeError("knockout failure")


class TestHookCleanup(unittest.TestCase):
    def test_activation_collector_removes_hook_on_exception(self):
        model = _ExplodingCollectorModel()
        dataset = DatasetStub(num_samples=1)
        collector = ActivationCollector(adapt(model), layer_idx=0, activation_site="residual")

        with self.assertRaises(RuntimeError):
            collector.collect_from_dataset(dataset, position_type="question", max_samples=1)

        self.assertEqual(len(model.layers[0]._forward_hooks), 0)

    def test_feature_ablator_removes_hook_on_exception(self):
        model = _ExplodingAblationModel()
        sae = SparseAutoencoder(d_model=4, n_features=8)
        ablator = FeatureAblator(adapt(model), sae, layer_idx=0, activation_site="residual")
        dataset = DatasetStub(num_samples=1)

        with self.assertRaises(RuntimeError):
            ablator.batch_ablation_experiment(
                dataset,
                feature_indices=[0, 1],
                position_type="question",
                max_samples=1,
            )

        self.assertEqual(len(model.layers[0]._forward_hooks), 0)

    def test_sequence_logprob_removes_knockout_on_exception(self):
        adapter = adapt(_ExplodingKnockoutModel())
        batch = adapter.build_inputs("what color is it")

        with self.assertRaises(RuntimeError):
            sequence_logprob(
                adapter,
                batch,
                answer_text="yes",
                normalize=True,
                block_config={0: [(0, 0)]},
            )

        adapter.assert_knockouts_balanced()
        installs = [e for e in adapter.knockout_log if e[0] == "install"]
        self.assertEqual(len(installs), 1)
        self.assertEqual(installs[0][2], {0: [(0, 0)]})


if __name__ == "__main__":
    unittest.main()
