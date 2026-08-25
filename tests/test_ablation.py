import unittest
from types import SimpleNamespace
import torch
from torch import nn

from vlmflowprobe.ablation.feature_ablator import FeatureAblator
from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder


class DummyModel(nn.Module):
    def __init__(self, d_model=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model)])

    def forward(self, input_ids=None, images=None, image_sizes=None, use_cache=False):
        x = torch.randn(1, 2, self.layers[0].in_features)
        return self.layers[0](x)


class CountingModel(nn.Module):
    def __init__(self, d_model=4, vocab=16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model)])
        self.generate_calls = 0
        self.forward_calls = 0
        self.vocab = vocab

    def forward(self, input_ids=None, images=None, image_sizes=None, use_cache=False):
        self.forward_calls += 1
        seq_len = input_ids.shape[1]
        logits = torch.zeros(1, seq_len, self.vocab, device=input_ids.device)
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self.generate_calls += 1
        scores = torch.zeros(1, self.vocab)
        scores[0, 1] = 1.0
        return {
            "sequences": torch.tensor([[1]], dtype=torch.long),
            "scores": [scores],
        }


class TokenizerStub:
    eos_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [1]

    @staticmethod
    def batch_decode(sequences, skip_special_tokens=True):
        return ["yes"]


class DatasetStub:
    def __init__(self, num_samples=2):
        self.tokenizer = TokenizerStub()
        self.questions = []
        self.dataset_dict = {}
        self._batches = []
        for idx in range(num_samples):
            qid = f"q{idx}"
            self.questions.append({"q_id": qid, "attribute_tokens": []})
            self.dataset_dict[qid] = {
                "question": "what color is it",
                "answer": "yes",
                "true option": "yes",
                "false option": "no",
            }
            self._batches.append(
                (
                    torch.tensor([[1, 2, 3]], dtype=torch.long),
                    [torch.zeros(1, 3, 4, 4)],
                    [(4, 4)],
                    "",
                    torch.zeros(1, 3, 4, 4),
                )
            )

    def create_dataloader(self):
        return list(self._batches)


class TestFeatureAblator(unittest.TestCase):
    def test_ablation_hook_shape(self):
        model = DummyModel(d_model=4)
        sae = SparseAutoencoder(d_model=4, n_features=8)
        ablator = FeatureAblator(model, sae, layer_idx=0)

        acts = torch.randn(1, 2, 4)
        hook = ablator.create_ablation_hook([0, 1, 2])
        output = hook(None, None, acts)
        self.assertEqual(output.shape, acts.shape)

    def test_baseline_cache_reduces_redundant_generate_calls(self):
        dataset = DatasetStub(num_samples=2)
        sae = SparseAutoencoder(d_model=4, n_features=8)

        no_cache_model = CountingModel(d_model=4)
        no_cache_ablator = FeatureAblator(no_cache_model, sae, layer_idx=0)
        no_cache_ablator.batch_ablation_experiment(
            dataset,
            feature_indices=[0],
            apply_sae=False,
            max_samples=2,
            score_options=False,
        )
        no_cache_ablator.batch_ablation_experiment(
            dataset,
            feature_indices=[1],
            apply_sae=False,
            max_samples=2,
            score_options=False,
        )
        self.assertEqual(no_cache_model.generate_calls, 8)

        cache_model = CountingModel(d_model=4)
        cache_ablator = FeatureAblator(cache_model, sae, layer_idx=0)
        cache = cache_ablator.compute_baseline_cache(
            dataset,
            max_samples=2,
            score_options=False,
        )
        cache_ablator.batch_ablation_experiment(
            dataset,
            feature_indices=[0],
            apply_sae=False,
            max_samples=2,
            baseline_cache=cache,
            score_options=False,
        )
        cache_ablator.batch_ablation_experiment(
            dataset,
            feature_indices=[1],
            apply_sae=False,
            max_samples=2,
            baseline_cache=cache,
            score_options=False,
        )
        self.assertEqual(cache_model.generate_calls, 6)

    def test_skip_option_scores_disables_margin_scoring(self):
        dataset = DatasetStub(num_samples=2)
        model = CountingModel(d_model=4)
        sae = SparseAutoencoder(d_model=4, n_features=8)
        ablator = FeatureAblator(model, sae, layer_idx=0)

        results = ablator.batch_ablation_experiment(
            dataset,
            feature_indices=[0],
            apply_sae=False,
            max_samples=2,
            score_options=False,
        )

        self.assertEqual(model.forward_calls, 0)
        self.assertTrue(all(item["baseline_true_logprob"] is None for item in results))
        self.assertTrue(all(item["ablated_true_logprob"] is None for item in results))
        summary = ablator.compute_ablation_effect(results)
        self.assertIsNone(summary["mean_margin_drop"])


if __name__ == "__main__":
    unittest.main()
