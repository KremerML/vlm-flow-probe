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
        return AblationExperiment(adapter=object(), sae=DummySAE(n_features=n_features), config=config)

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


class TestMatchingDiagnostics(unittest.TestCase):
    """The opt-in matching instrumentation and the pool-depth ceiling."""

    def _stats(self, n=500):
        # activation_mean spread over three orders of magnitude
        return {i: {"activation_mean": 1e-3 * (1.02 ** i)} for i in range(n)}

    def test_pool_depth_reports_shallow_pool(self):
        from vlmflowprobe.ablation.matching_diagnostics import matched_pool_depth

        stats = self._stats()
        binding = list(range(400, 500))  # the top 100 by activation
        depth = matched_pool_depth(stats, binding, "activation_mean")
        # only features inside the binding range qualify, and they are binding
        self.assertLess(depth["max_disjoint_matched_sets"], 1.0)
        self.assertFalse(depth["sufficient_for_one_set"])

    def test_pool_depth_reports_deep_pool(self):
        from vlmflowprobe.ablation.matching_diagnostics import matched_pool_depth

        stats = self._stats()
        binding = list(range(0, 20)) + list(range(480, 500))  # spans the range
        depth = matched_pool_depth(stats, binding, "activation_mean")
        self.assertTrue(depth["sufficient_for_one_set"])
        self.assertGreater(depth["max_disjoint_matched_sets"], 5)

    def test_strict_extract_refuses_fallback_keys(self):
        from vlmflowprobe.ablation.ablation_experiments import AblationExperiment

        stats = {"ratio": 3.0}  # a v1 key, not what we asked for
        self.assertEqual(
            AblationExperiment._extract_metric_value(stats, "activation_mean"), 3.0
        )
        self.assertIsNone(
            AblationExperiment._extract_metric_value(stats, "activation_mean", strict=True)
        )
        value, used_fallback = AblationExperiment._extract_metric_value(
            stats, "activation_mean", with_provenance=True
        )
        self.assertEqual((value, used_fallback), (3.0, True))

    def test_diagnostics_flags_uniform_vs_matched(self):
        from vlmflowprobe.ablation.matching_diagnostics import MatchingDiagnostics

        diag = MatchingDiagnostics(metric="activation_mean", strict=True)
        diag.start_set()
        for i in range(10):
            diag.record(binding_feature=i, binding_value=1.0,
                        control_feature=100 + i, control_value=0.95,
                        path="matched")
        report = diag.summarize()
        self.assertEqual(report["matched_fraction"], 1.0)
        self.assertAlmostEqual(report["control_over_binding_median_ratio"], 0.95)

        ghost = MatchingDiagnostics(metric="correct_mean")
        ghost.start_set()
        for i in range(10):
            ghost.record(binding_feature=i, binding_value=None,
                         control_feature=100 + i, control_value=6e-8,
                         path="uniform_no_target")
        r2 = ghost.summarize(binding_values=[0.12] * 10)
        self.assertEqual(r2["matched_fraction"], 0.0)
        self.assertLess(r2["control_over_binding_median_ratio"], 1e-5)
