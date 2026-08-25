import unittest

import numpy as np

from vlmflowprobe.ablation.statistical_analysis import (
    paired_bootstrap_ci,
    ratio_statistic,
    z_score_standard_error,
)


class TestPairedBootstrap(unittest.TestCase):
    def test_pairing_narrows_the_interval_on_a_ratio(self):
        """The reason the bootstrap must be paired.

        Ablation and knockout are measured on the same questions and are strongly
        correlated, so a hard question inflates both. Resampling them together preserves
        that; resampling independently discards it and widens the interval on A/K.
        """
        rng = np.random.default_rng(0)
        difficulty = rng.normal(1.0, 0.5, 256)
        knockout = difficulty * 1.0
        ablation = difficulty * 0.6

        _, low, high = paired_bootstrap_ci(
            {"a": ablation, "k": knockout}, ratio_statistic("a", "k"), n_bootstrap=2000
        )
        paired_width = high - low

        # Independent resampling, i.e. what you get by bootstrapping each condition alone.
        widths = []
        for _ in range(2000):
            i, j = rng.integers(0, 256, 256), rng.integers(0, 256, 256)
            widths.append(np.mean(ablation[i]) / np.mean(knockout[j]))
        unpaired_width = float(np.percentile(widths, 97.5) - np.percentile(widths, 2.5))

        self.assertLess(paired_width, unpaired_width)

    def test_point_estimate_is_the_ratio_of_means(self):
        ablation = np.full(100, 0.2)
        knockout = np.full(100, 0.4)
        point, _, _ = paired_bootstrap_ci(
            {"a": ablation, "k": knockout}, ratio_statistic("a", "k"), n_bootstrap=200
        )
        self.assertAlmostEqual(point, 0.5, places=6)

    def test_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(3)
        ablation = rng.normal(0.24, 0.2, 256)
        knockout = rng.normal(0.40, 0.2, 256)
        point, low, high = paired_bootstrap_ci(
            {"a": ablation, "k": knockout}, ratio_statistic("a", "k"), n_bootstrap=2000
        )
        self.assertLessEqual(low, point)
        self.assertLessEqual(point, high)

    def test_seed_makes_the_interval_reproducible(self):
        rng = np.random.default_rng(5)
        data = {"a": rng.normal(0.2, 0.1, 128), "k": rng.normal(0.4, 0.1, 128)}
        first = paired_bootstrap_ci(data, ratio_statistic("a", "k"), n_bootstrap=500, seed=7)
        second = paired_bootstrap_ci(data, ratio_statistic("a", "k"), n_bootstrap=500, seed=7)
        self.assertEqual(first, second)

    def test_non_positive_denominator_yields_nan_bounds(self):
        # Layer 13's knockout drop is negative, so its ratio is genuinely undefined.
        ablation = np.full(64, 0.12)
        knockout = np.full(64, -0.05)
        point, low, high = paired_bootstrap_ci(
            {"a": ablation, "k": knockout}, ratio_statistic("a", "k"), n_bootstrap=200
        )
        self.assertTrue(np.isnan(point))
        self.assertTrue(np.isnan(low) and np.isnan(high))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_ci(
                {"a": np.zeros(10), "k": np.zeros(11)}, ratio_statistic("a", "k")
            )

    def test_empty_input_is_nan_not_a_crash(self):
        point, low, high = paired_bootstrap_ci({}, ratio_statistic("a", "k"))
        self.assertTrue(np.isnan(point) and np.isnan(low) and np.isnan(high))


class TestZScoreStandardError(unittest.TestCase):
    def test_published_z_cannot_carry_three_significant_figures(self):
        # z = 81.6 from 15 control sets: the SE is around 15.
        se = z_score_standard_error(81.6, n_sets=15)
        self.assertGreater(se, 10.0)
        self.assertLess(se, 20.0)

    def test_more_control_sets_shrink_the_error(self):
        self.assertLess(
            z_score_standard_error(50.0, n_sets=100),
            z_score_standard_error(50.0, n_sets=15),
        )

    def test_too_few_sets_is_nan(self):
        self.assertTrue(np.isnan(z_score_standard_error(50.0, n_sets=1)))


if __name__ == "__main__":
    unittest.main()
