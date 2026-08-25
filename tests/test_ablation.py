import unittest

import torch

from tests.stubs import CountingModel, DatasetStub, DummyModel, adapt
from vlmflowprobe.ablation.feature_ablator import FeatureAblator
from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder


class TestFeatureAblator(unittest.TestCase):
    def test_ablation_hook_shape(self):
        adapter = adapt(DummyModel(d_model=4))
        sae = SparseAutoencoder(d_model=4, n_features=8)
        ablator = FeatureAblator(adapter, sae, layer_idx=0)

        acts = torch.randn(1, 2, 4)
        hook = ablator.create_ablation_hook([0, 1, 2])
        output = hook(None, None, acts)
        self.assertEqual(output.shape, acts.shape)

    def test_baseline_cache_reduces_redundant_generate_calls(self):
        dataset = DatasetStub(num_samples=2)
        sae = SparseAutoencoder(d_model=4, n_features=8)

        no_cache_model = CountingModel(d_model=4)
        no_cache_ablator = FeatureAblator(adapt(no_cache_model), sae, layer_idx=0)
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
        cache_ablator = FeatureAblator(adapt(cache_model), sae, layer_idx=0)
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
        ablator = FeatureAblator(adapt(model), sae, layer_idx=0)

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
