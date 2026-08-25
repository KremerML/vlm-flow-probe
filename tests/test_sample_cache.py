import tempfile
import unittest

from vlmflowprobe.ablation.feature_ablator import FeatureAblator
from vlmflowprobe.ablation.sample_cache import (
    baseline_cache,
    build_sample_cache,
    load_sample_cache,
    positions_cache,
    save_sample_cache,
)
from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder
from tests.stubs import CountingModel, DatasetStub

BASELINE_FIELDS = (
    "baseline_pred",
    "baseline_prob",
    "baseline_gt_prob",
    "baseline_true_logprob",
    "baseline_false_logprob",
    "baseline_margin",
)


def _ablator(model=None):
    model = model or CountingModel(d_model=4)
    sae = SparseAutoencoder(d_model=4, n_features=8)
    return FeatureAblator(model, sae, layer_idx=0), model


class TestSampleCache(unittest.TestCase):
    def test_cache_removes_redundant_generate_calls(self):
        dataset = DatasetStub(num_samples=2)

        uncached_ablator, uncached_model = _ablator()
        for features in ([0], [1], [2]):
            uncached_ablator.batch_ablation_experiment(
                dataset,
                feature_indices=features,
                apply_sae=False,
                max_samples=2,
                score_options=False,
            )

        cached_ablator, cached_model = _ablator()
        records = build_sample_cache(
            cached_ablator, dataset, max_samples=2, score_options=False
        )
        for features in ([0], [1], [2]):
            cached_ablator.batch_ablation_experiment(
                dataset,
                feature_indices=features,
                apply_sae=False,
                max_samples=2,
                baseline_cache=baseline_cache(records),
                positions_cache=positions_cache(records),
                strict_cache=True,
                score_options=False,
            )

        # Uncached pays one baseline generate per sample per condition on top of the
        # ablated one; cached pays the baselines once, during the cache build.
        self.assertEqual(uncached_model.generate_calls, 12)
        self.assertEqual(cached_model.generate_calls, 8)

    def test_cached_baseline_fields_are_identical(self):
        dataset = DatasetStub(num_samples=2)

        uncached_ablator, _ = _ablator()
        uncached = uncached_ablator.batch_ablation_experiment(
            dataset,
            feature_indices=[0],
            apply_sae=False,
            max_samples=2,
            score_options=False,
        )

        cached_ablator, _ = _ablator()
        records = build_sample_cache(
            cached_ablator, dataset, max_samples=2, score_options=False
        )
        cached = cached_ablator.batch_ablation_experiment(
            dataset,
            feature_indices=[0],
            apply_sae=False,
            max_samples=2,
            baseline_cache=baseline_cache(records),
            positions_cache=positions_cache(records),
            strict_cache=True,
            score_options=False,
        )

        self.assertEqual(len(uncached), len(cached))
        for left, right in zip(uncached, cached):
            for field in BASELINE_FIELDS:
                self.assertEqual(left[field], right[field], msg=field)

    def test_strict_cache_raises_when_cache_and_dataset_disagree(self):
        dataset = DatasetStub(num_samples=2)
        ablator, _ = _ablator()
        records = build_sample_cache(
            ablator, dataset, max_samples=2, score_options=False
        )

        # A cache built against a different sample order must not silently recompute.
        shuffled = list(reversed(baseline_cache(records)))
        with self.assertRaises(ValueError):
            ablator.batch_ablation_experiment(
                dataset,
                feature_indices=[0],
                apply_sae=False,
                max_samples=2,
                baseline_cache=shuffled,
                strict_cache=True,
                score_options=False,
            )

    def test_non_strict_cache_miss_is_counted_not_raised(self):
        dataset = DatasetStub(num_samples=2)
        ablator, _ = _ablator()
        records = build_sample_cache(
            ablator, dataset, max_samples=2, score_options=False
        )
        shuffled = list(reversed(baseline_cache(records)))

        ablator.batch_ablation_experiment(
            dataset,
            feature_indices=[0],
            apply_sae=False,
            max_samples=2,
            baseline_cache=shuffled,
            strict_cache=False,
            score_options=False,
        )
        self.assertEqual(ablator._cache_misses, 2)

    def test_positions_none_survives_a_round_trip(self):
        # `None` positions mean "every position" and must not be confused with a cache miss.
        dataset = DatasetStub(num_samples=2)
        ablator, _ = _ablator()
        records = build_sample_cache(
            ablator, dataset, position_type="all", max_samples=2, score_options=False
        )
        self.assertTrue(all(record.positions is None for record in records))

        resolved = FeatureAblator._resolve_cached_positions(
            positions_cache(records), sample_idx=0, question_id="q0"
        )
        self.assertIsNone(resolved)

    def test_save_and_load_round_trip(self):
        dataset = DatasetStub(num_samples=2)
        ablator, _ = _ablator()
        records = build_sample_cache(
            ablator, dataset, max_samples=2, score_options=False
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/sample_cache.json"
            save_sample_cache(records, path)
            restored = load_sample_cache(path)

        self.assertEqual(len(restored), len(records))
        for left, right in zip(records, restored):
            self.assertEqual(left.question_id, right.question_id)
            self.assertEqual(left.positions, right.positions)
            self.assertEqual(left.baseline, right.baseline)


if __name__ == "__main__":
    unittest.main()
