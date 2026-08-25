import random
import unittest
from unittest.mock import patch

from vlmflowprobe.ablation.ablation_experiments import AblationExperiment


class DummySAE:
    def __init__(self, n_features=512):
        self.n_features = n_features


class TestRandomControlSampling(unittest.TestCase):
    def _make_experiment(self, n_features=512, n_random_sets=6):
        config = {
            "model": {"target_layer": 0, "activation_site": "residual"},
            "ablation": {
                "n_random_features": 32,
                "n_random_sets": n_random_sets,
                "random_sampling": "matched",
                "position_type": "attribute",
                "mode": "residual",
                "delta_scale": 1.0,
                "operation": "zero",
                "operation_scale": 1.0,
            },
            "random_control": {
                "n_random_sets": n_random_sets,
                "sampling": "matched",
                "seed": 123,
                "matched_metric": "correct_mean",
            },
            "evaluation": {"logprob_normalize": True},
        }
        return AblationExperiment(model=object(), sae=DummySAE(n_features=n_features), config=config)

    def test_matched_sampling_produces_distinct_sets(self):
        exp = self._make_experiment(n_features=512, n_random_sets=8)
        binding_features = list(range(64))
        feature_stats = {
            idx: {"correct_mean": float(idx % 19), "ratio": float((idx % 7) + 1)}
            for idx in range(512)
        }
        rng = random.Random(123)

        sets = []
        for _ in range(8):
            sampled = exp._sample_random_features(
                binding_features=binding_features,
                n_random_features=32,
                sampling="matched",
                feature_stats=feature_stats,
                matched_metric="correct_mean",
                rng=rng,
            )
            sets.append(tuple(sampled))

        self.assertEqual(len(sets), 8)
        self.assertGreater(len(set(sets)), 1, "Matched random sets should vary across repeats")
        for sampled in sets:
            self.assertEqual(len(sampled), 32)
            self.assertTrue(set(sampled).isdisjoint(binding_features))

    def test_same_seed_reproducibility_for_first_random_set(self):
        exp = self._make_experiment(n_features=256, n_random_sets=2)
        binding_features = list(range(32))
        feature_stats = {idx: {"correct_mean": float(idx % 11)} for idx in range(256)}

        rng1 = random.Random(777)
        rng2 = random.Random(777)
        set1 = exp._sample_random_features(
            binding_features=binding_features,
            n_random_features=16,
            sampling="matched",
            feature_stats=feature_stats,
            matched_metric="correct_mean",
            rng=rng1,
        )
        set2 = exp._sample_random_features(
            binding_features=binding_features,
            n_random_features=16,
            sampling="matched",
            feature_stats=feature_stats,
            matched_metric="correct_mean",
            rng=rng2,
        )
        self.assertEqual(set1, set2)

    def test_run_three_condition_test_records_all_random_sets(self):
        exp = self._make_experiment(n_features=256, n_random_sets=5)
        binding_features = list(range(32))
        feature_stats = {idx: {"correct_mean": float(idx % 13)} for idx in range(256)}

        fake_rows = [
            {
                "baseline_pred": "yes",
                "ablated_pred": "yes",
                "answer": "yes",
                "baseline_prob": 0.9,
                "ablated_prob": 0.9,
                "baseline_gt_prob": None,
                "ablated_gt_prob": None,
                "baseline_margin": None,
                "ablated_margin": None,
            }
        ]

        # run_three_condition_test now resolves the baseline and the intervention positions
        # once up front, so the stub dataset needs that call patched out too.
        with patch(
            "vlmflowprobe.ablation.ablation_experiments.FeatureAblator.batch_ablation_experiment",
            return_value=fake_rows,
        ), patch(
            "vlmflowprobe.ablation.ablation_experiments.build_sample_cache",
            return_value=[],
        ):
            results = exp.run_three_condition_test(
                dataset=object(),
                binding_features=binding_features,
                feature_stats=feature_stats,
                show_progress=False,
            )

        self.assertEqual(len(results["random_feature_sets"]), 5)
        self.assertEqual(len(results["random_set_summaries"]), 5)
        self.assertEqual(results["random_control_settings"]["n_random_sets"], 5)
        self.assertGreater(
            len({tuple(s) for s in results["random_feature_sets"]}),
            1,
            "Random feature sets should not all collapse to one set",
        )


if __name__ == "__main__":
    unittest.main()
