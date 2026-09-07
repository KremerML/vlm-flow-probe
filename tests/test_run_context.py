"""Where an analysis gets its layers, spans and model depth.

The point of ``RunContext`` is that the analysis CLIs stopped carrying LLaVA's
layer numbers as literals. The load-bearing test is therefore the first one: the
derived spans must still be byte-identical to the table that was deleted, or the
published multi-layer analysis would silently start reading different conditions.
"""

import json
import unittest

from vlmflowprobe.utils.run_context import RunContext, load_run_context, span_label

# The table that used to live in cli/analyze_multilayer.py as SPAN_PAIRS,
# frozen here as the regression target for the derived version.
HISTORICAL_SPAN_PAIRS = [
    ("nested", "{14}", "nested_L14", "nested_knockout_L14", (14,)),
    ("nested", "{13,14}", "nested_L13-14", "nested_knockout_L13-14", (13, 14)),
    ("nested", "{12,13,14}", "nested_L12-14", "nested_knockout_L12-14", (12, 13, 14)),
    ("nested", "{11-14}", "nested_L11-14", "nested_knockout_L11-14", (11, 12, 13, 14)),
    ("nested", "{10-14}", "joint_L10-14", "span_knockout_L10-14", (10, 11, 12, 13, 14)),
    ("single", "{11}", "A0_regression_L11", "knockout_L11", (11,)),
    ("non-nested", "{10,11,12}", "nonnested_L10-12", "nonnested_knockout_L10-12", (10, 11, 12)),
    ("non-nested", "{10,12,14}", "nonnested_L10,12,14", "nonnested_knockout_L10,12,14",
     (10, 12, 14)),
    ("sensitivity", "{10,11,12,14}", "sensitivity_joint_L10,11,12,14",
     "sensitivity_knockout_L10,11,12,14", (10, 11, 12, 14)),
]


class TestDerivedSpans(unittest.TestCase):
    def test_default_config_reproduces_the_historical_table(self):
        pairs = load_run_context().span_pairs()
        derived = [
            (p.kind, p.label, p.ablation_id, p.knockout_id, p.layers) for p in pairs
        ]
        self.assertEqual(derived, HISTORICAL_SPAN_PAIRS)

    def test_spans_follow_the_config_not_the_model(self):
        context = RunContext(config={
            "multilayer": {"layers": [20, 21, 22]},
            "conditions": {"nested": [[22], [21, 22], [20, 21, 22]], "non_nested": []},
        })
        self.assertEqual(context.layers, [20, 21, 22])
        ids = [p.ablation_id for p in context.span_pairs()]
        self.assertEqual(ids, ["nested_L22", "nested_L21-22", "joint_L20-22"])

    def test_no_a0_layer_means_no_regression_pair(self):
        context = RunContext(config={
            "multilayer": {"layers": [3]},
            "conditions": {"nested": [[3]], "a0_regression_layer": None},
        })
        self.assertEqual([p.kind for p in context.span_pairs()], ["nested"])


class TestSpanLabel(unittest.TestCase):
    def test_short_spans_are_listed_and_long_contiguous_ones_compacted(self):
        self.assertEqual(span_label([14]), "{14}")
        self.assertEqual(span_label([10, 11, 12]), "{10,11,12}")
        self.assertEqual(span_label([11, 12, 13, 14]), "{11-14}")
        self.assertEqual(span_label([10, 11, 12, 14]), "{10,11,12,14}")


class TestDepth(unittest.TestCase):
    def test_depth_fraction_makes_layers_comparable_across_models(self):
        shallow = RunContext(n_layers=32)
        deep = RunContext(n_layers=64)
        self.assertAlmostEqual(shallow.depth(16), 0.5)
        self.assertAlmostEqual(deep.depth(32), 0.5)
        self.assertAlmostEqual(shallow.depth_of([10, 14]), 12 / 32)

    def test_unknown_depth_is_reported_as_unknown_not_guessed(self):
        context = RunContext()
        self.assertIsNone(context.depth(11))
        self.assertEqual(context.depth_label(11), "11")

    def test_depth_label_carries_the_fraction_when_known(self):
        self.assertEqual(RunContext(n_layers=32).depth_label(11), "11 (depth 0.34)")


class TestLoading(unittest.TestCase):
    def _write_provenance(self, tmp, config, geometry=None):
        import os

        os.makedirs(tmp, exist_ok=True)
        payload = {"resolved_config": config}
        if geometry:
            payload["model_geometry"] = geometry
        with open(os.path.join(tmp, "provenance.json"), "w") as handle:
            json.dump(payload, handle)

    def test_run_provenance_supplies_layers_and_depth(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._write_provenance(
                tmp,
                {"multilayer": {"layers": [4, 5]}, "conditions": {}},
                {"n_layers": 26, "d_model": 2304},
            )
            context = load_run_context(experiment_dir=tmp)
            self.assertEqual(context.layers, [4, 5])
            self.assertEqual(context.n_layers, 26)
            self.assertTrue(context.source.endswith("provenance.json"))

    def test_missing_provenance_falls_back_to_defaults_and_says_so(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            context = load_run_context(experiment_dir=tmp)
            self.assertEqual(context.layers, [10, 11, 12, 13, 14])
            self.assertIn("package defaults", context.source)
            self.assertIsNone(context.n_layers)

    def test_explicit_n_layers_overrides_provenance(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._write_provenance(tmp, {"multilayer": {"layers": [1]}}, {"n_layers": 32})
            self.assertEqual(load_run_context(experiment_dir=tmp, n_layers=40).n_layers, 40)

    def test_layers_fall_back_to_the_layers_that_have_statistics(self):
        context = RunContext(config={
            "multilayer": {"stats_paths": {"14": "a.json", "10": "b.json"}}
        })
        self.assertEqual(context.layers, [10, 14])


if __name__ == "__main__":
    unittest.main()
