"""Controls that are disjoint by construction, and the properties they must have."""

import unittest

from vlmflowprobe.ablation.rank_band_controls import (
    activation_matched_control,
    activation_top_control,
    adjacent_band_control,
    dose_response_bands,
    perturbation_match_quality,
    random_subsets,
    rank_band,
    rank_features,
    ranking_difference,
)


def make_stats(n=1000):
    """A dictionary where causal_score and activation_mean are correlated but
    not identical, as in the real data (rank correlation about 0.95)."""
    stats = {}
    for i in range(n):
        activation = 1.0 / (1.0 + i * 0.01)
        # perturb the causal ordering slightly relative to activation
        gradient = 1.0 + 0.3 * ((i * 7919) % 11) / 11.0
        stats[i] = {
            "activation_mean": activation,
            "gradient_mean": gradient,
            "causal_score": activation * gradient,
        }
    return stats


class TestRanking(unittest.TestCase):
    def test_ordering_is_descending_and_deterministic(self):
        stats = make_stats(50)
        order = rank_features(stats, "activation_mean")
        values = [stats[f]["activation_mean"] for f in order]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(order, rank_features(stats, "activation_mean"))

    def test_ties_broken_by_feature_id(self):
        stats = {5: {"x": 1.0}, 2: {"x": 1.0}, 9: {"x": 1.0}}
        self.assertEqual(rank_features(stats, "x"), [2, 5, 9])

    def test_missing_key_treated_as_zero(self):
        stats = {0: {"x": 1.0}, 1: {}, 2: {"x": None}}
        self.assertEqual(rank_features(stats, "x")[0], 0)


class TestAdjacentBand(unittest.TestCase):
    def test_control_is_disjoint_from_binding_and_same_size(self):
        stats = make_stats()
        binding = rank_band(stats, 0, 200)
        control = adjacent_band_control(stats, 200)
        self.assertEqual(len(control), len(binding))
        self.assertFalse(set(control) & set(binding))

    def test_raises_when_dictionary_too_small(self):
        # 150 features cannot supply ranks 100..200 for a k=100 binding set
        stats = make_stats(150)
        with self.assertRaises(ValueError):
            adjacent_band_control(stats, 100)

    def test_band_bounds_validated(self):
        stats = make_stats(50)
        with self.assertRaises(ValueError):
            rank_band(stats, 10, 5)


class TestActivationControls(unittest.TestCase):
    def test_activation_top_carries_at_least_as_much_mass(self):
        # The conservative control must not be handicapped: by construction the
        # top-k by activation holds the most activation any k features can.
        stats = make_stats()
        binding = rank_band(stats, 0, 200, key="causal_score")
        control = activation_top_control(stats, 200)
        mass = lambda fs: sum(stats[f]["activation_mean"] for f in fs)
        self.assertGreaterEqual(mass(control), mass(binding))

    def test_activation_matched_excludes_binding(self):
        stats = make_stats()
        binding = rank_band(stats, 0, 100)
        matched = activation_matched_control(stats, binding)
        self.assertFalse(set(matched) & set(binding))
        self.assertEqual(len(set(matched)), len(matched))


class TestRankingDifference(unittest.TestCase):
    def test_partition_is_exact(self):
        stats = make_stats()
        parts = ranking_difference(stats, 200)
        causal = set(rank_band(stats, 0, 200, key="causal_score"))
        activation = set(rank_band(stats, 0, 200, key="activation_mean"))
        self.assertEqual(set(parts["shared"]) | set(parts["causal_only"]), causal)
        self.assertEqual(set(parts["shared"]) | set(parts["activation_only"]), activation)
        self.assertFalse(set(parts["causal_only"]) & set(parts["activation_only"]))
        self.assertEqual(len(parts["causal_only"]), len(parts["activation_only"]))


class TestDoseResponse(unittest.TestCase):
    def test_bands_are_disjoint_equal_size_and_ordered(self):
        stats = make_stats()
        bands = dose_response_bands(stats, width=40, n_bands=10)
        self.assertEqual(len(bands), 10)
        seen = set()
        for b in bands:
            self.assertEqual(len(b["features"]), 40)
            self.assertFalse(seen & set(b["features"]))
            seen |= set(b["features"])

    def test_stops_when_the_dictionary_runs_out(self):
        stats = make_stats(100)
        bands = dose_response_bands(stats, width=40, n_bands=10)
        self.assertEqual(len(bands), 2)


class TestRandomSubsets(unittest.TestCase):
    def test_subsets_are_varied_unlike_matched_sets(self):
        # The point of this arm: overlap should be near size/pool, not 97%.
        pool = list(range(200))
        sets = random_subsets(pool, size=40, n_sets=12, seed=1)
        overlaps = [
            len(set(a) & set(b)) / 40
            for i, a in enumerate(sets) for b in sets[i + 1:]
        ]
        mean_overlap = sum(overlaps) / len(overlaps)
        self.assertLess(mean_overlap, 0.40)

    def test_deterministic_given_seed(self):
        pool = list(range(200))
        self.assertEqual(random_subsets(pool, 40, 5, seed=7),
                         random_subsets(pool, 40, 5, seed=7))

    def test_rejects_oversized_draw(self):
        with self.assertRaises(ValueError):
            random_subsets(list(range(10)), size=40, n_sets=1)


class TestPerturbationMatch(unittest.TestCase):
    def test_reports_when_control_perturbs_more(self):
        # Measured layer-11 case: the activation control perturbs 43% more.
        q = perturbation_match_quality(0.01855, 0.02652)
        self.assertTrue(q["control_perturbs_more"])
        self.assertAlmostEqual(q["ratio"], 1.4297, places=3)

    def test_close_match_reported_as_such(self):
        # Measured layer-11 adjacent band: within 0.6%.
        q = perturbation_match_quality(0.01855, 0.01844)
        self.assertLess(abs(q["relative_difference"]), 0.01)
        self.assertFalse(q["control_perturbs_more"])

    def test_none_when_unavailable(self):
        self.assertIsNone(perturbation_match_quality(None, 0.01))
        self.assertIsNone(perturbation_match_quality(0.01, None))



class TestConditionBuilding(unittest.TestCase):
    """The runner's condition families, built without touching a model."""

    def setUp(self):
        from vlmflowprobe.cli.run_controls import build_conditions
        self.build = build_conditions
        self.stats = {11: make_stats(1200), 14: make_stats(1200)}

    def test_single_phase_pairs_binding_with_disjoint_controls(self):
        conds = self.build(self.stats, ["single"], k=200)
        by_id = {c.condition_id: c for c in conds}
        binding = by_id["bind_causal_1_200_L11"].features[11]
        control = by_id["ctl_band_200_400_L11"].features[11]
        self.assertEqual(len(binding), len(control))
        self.assertFalse(set(binding) & set(control))
        # the pass-through hooks the layer but ablates nothing
        self.assertEqual(by_id["ctl_passthrough_L11"].features[11], [])

    def test_dose_bands_are_disjoint_and_equal_size(self):
        conds = self.build(self.stats, ["doseresponse"], k=200)
        bands = [c for c in conds if c.condition_id.startswith("dose_L11_")]
        seen = set()
        for c in bands:
            feats = c.features[11]
            self.assertEqual(len(feats), 40)
            self.assertFalse(seen & set(feats))
            seen |= set(feats)

    def test_multi_phase_spans_every_layer(self):
        conds = self.build(self.stats, ["multi"], k=200)
        joint = next(c for c in conds if c.condition_id == "ctl_joint")
        self.assertEqual(sorted(joint.features), [11, 14])
        for feats in joint.features.values():
            self.assertEqual(len(feats), 200)

    def test_unknown_phase_yields_nothing(self):
        self.assertEqual(self.build(self.stats, [], k=200), [])


if __name__ == "__main__":
    unittest.main()
