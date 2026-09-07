"""Tests for the per-span redundancy analysis in vlmflowprobe/cli/analyze_multilayer.py.

The bootstrap primitives are covered in test_paired_bootstrap.py. What is tested here is the
layer above them: that spans are paired correctly, that the pairing guards actually fire, and
that the trend test can tell a flat R series from a rising one -- which is the distinction the
writeup's redundancy claim rests on.
"""

import unittest

import numpy as np

from vlmflowprobe.cli import analyze_multilayer as amla
from vlmflowprobe.utils.run_context import load_run_context

SHA = "a" * 40

#: The spans the default (LLaVA) config describes; the analysis derives these
#: from the run instead of carrying them as literals.
PAIRS = load_run_context().span_pairs()


def condition(drops, sha=SHA, layers=(), knockout_layers=()):
    return {
        "summary": {
            "mean_margin_drop": float(np.mean(drops)) if len(drops) else None,
            "layers": list(layers),
            "knockout_layers": list(knockout_layers),
            "kind": "knockout" if knockout_layers else "sae",
        },
        "significance": {},
        "meta": {},
        "control_summaries": [],
        "margin_drops": list(map(float, drops)),
        "question_ids_sha1": sha,
    }


def nested_matrix(ratios, n=256, seed=0, noise=0.25):
    """Five nested spans whose R values are `ratios`, sharing per-question difficulty.

    Both arms scale with a shared difficulty term, which is what makes the pairing worth
    doing. The ablation arm also carries independent noise -- without it A/K would be exact
    and every interval would collapse to zero width, so the constancy and trend tests could
    never fail on anything. That noise is mean-centred so each span's point estimate lands
    exactly on its target ratio, leaving the tests to turn on the interval widths rather than
    on which way a particular seed happened to scatter the points.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(1.0, 0.2, n)
    conditions = {}
    for pair, ratio in zip(PAIRS, ratios):
        size = pair.span_size
        knockout = difficulty * size
        wobble = rng.normal(0.0, noise * size, n)
        wobble -= wobble.mean()  # centred, so the realised R is exactly `ratio`
        ablation = knockout * ratio + wobble
        conditions[pair.knockout_id] = condition(knockout, knockout_layers=pair.layers)
        conditions[pair.ablation_id] = condition(ablation, layers=pair.layers)
    return conditions


class TestRedundancyBySpan(unittest.TestCase):
    def test_rows_carry_span_shape_and_ratio(self):
        conditions = nested_matrix([0.7] * 5)
        rows = [r for r in amla.redundancy_by_span(conditions, PAIRS) if r["kind"] == "nested"]

        self.assertEqual([r["span_size"] for r in rows], [1, 2, 3, 4, 5])
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertAlmostEqual(row["ratio"], 0.7, places=9)
            self.assertLessEqual(row["ci_low"], row["ratio"])
            self.assertGreaterEqual(row["ci_high"], row["ratio"])

    def test_missing_condition_is_reported_not_skipped(self):
        """A dashed row must say why it is dashed; silently dropping it hides coverage gaps."""
        conditions = nested_matrix([0.7] * 5)
        del conditions["nested_L14"]

        rows = amla.redundancy_by_span(conditions, PAIRS)
        row = next(r for r in rows if r["label"] == "{14}")
        self.assertEqual(row["status"], "missing_condition")
        self.assertEqual(row["span_size"], 1)

    def test_negative_ceiling_is_undefined(self):
        """Layer 13's knockout is inhibitory, so A/K is meaningless rather than merely noisy."""
        conditions = nested_matrix([0.7] * 5)
        conditions["nested_knockout_L14"] = condition(
            -np.abs(np.asarray(conditions["nested_knockout_L14"]["margin_drops"])),
            knockout_layers=(14,),
        )
        row = next(r for r in amla.redundancy_by_span(conditions, PAIRS) if r["label"] == "{14}")
        self.assertEqual(row["status"], "undefined_negative_ceiling")
        self.assertIsNone(row["ratio"])

    def test_mismatched_question_hash_is_refused(self):
        """Equal length is not the same as same questions in the same order."""
        conditions = nested_matrix([0.7] * 5)
        conditions["nested_L14"]["question_ids_sha1"] = "b" * 40

        row = next(r for r in amla.redundancy_by_span(conditions, PAIRS) if r["label"] == "{14}")
        self.assertEqual(row["status"], "sha1_mismatch")

    def test_seed_is_reproducible(self):
        conditions = nested_matrix([0.7] * 5)
        first = amla.redundancy_by_span(conditions, PAIRS, seed=7)
        second = amla.redundancy_by_span(conditions, PAIRS, seed=7)
        self.assertEqual(
            [(r.get("ci_low"), r.get("ci_high")) for r in first],
            [(r.get("ci_low"), r.get("ci_high")) for r in second],
        )


class TestRedundancyTrend(unittest.TestCase):
    def test_flat_series_shows_no_trend_and_a_common_value(self):
        trend = amla.redundancy_trend(nested_matrix([0.7] * 5), PAIRS, n_bootstrap=2000)

        self.assertEqual(trend["status"], "ok")
        self.assertFalse(trend["slope_excludes_zero"])
        self.assertTrue(trend["common_value"])
        self.assertLessEqual(trend["slope_ci_low"], 0.0)
        self.assertGreaterEqual(trend["slope_ci_high"], 0.0)
        self.assertAlmostEqual(trend["pooled_ratio"], 0.7, places=9)

    def test_rising_series_is_detected(self):
        """The redundancy signature: R grows as the span covers more of the compensating layers."""
        trend = amla.redundancy_trend(
            nested_matrix([0.50, 0.60, 0.70, 0.80, 0.90]), PAIRS, n_bootstrap=2000
        )

        self.assertTrue(trend["slope_excludes_zero"])
        self.assertGreater(trend["slope_per_layer"], 0.0)
        self.assertGreater(trend["slope_ci_low"], 0.0)
        self.assertFalse(trend["common_value"])

    def test_dispersed_series_has_no_common_value(self):
        """No trend, yet not constant either -- which is what the real run turns out to be.

        A symmetric hump has a slope of exactly zero, so the two properties are separated:
        a flat trend line does not by itself license calling R constant.
        """
        trend = amla.redundancy_trend(
            nested_matrix([0.70, 0.85, 0.90, 0.85, 0.70]), PAIRS, n_bootstrap=2000
        )

        self.assertFalse(trend["slope_excludes_zero"])
        self.assertFalse(trend["common_value"])
        self.assertAlmostEqual(trend["spread"], 0.20, places=9)
        self.assertAlmostEqual(trend["slope_per_layer"], 0.0, places=9)
        self.assertGreater(trend["spread_ci_low"], 0.0)
        self.assertIn("not constant across spans", trend["interpretation"])

    def test_mismatched_hash_across_spans_is_refused(self):
        conditions = nested_matrix([0.7] * 5)
        conditions["joint_L10-14"]["question_ids_sha1"] = "c" * 40

        self.assertEqual(
            amla.redundancy_trend(conditions, PAIRS, n_bootstrap=200)["status"], "sha1_mismatch"
        )

    def test_missing_span_is_refused(self):
        conditions = nested_matrix([0.7] * 5)
        del conditions["span_knockout_L10-14"]

        self.assertEqual(
            amla.redundancy_trend(conditions, PAIRS, n_bootstrap=200)["status"], "missing_condition"
        )


if __name__ == "__main__":
    unittest.main()
