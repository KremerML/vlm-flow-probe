import random
import unittest

from vlmflowprobe.ablation.multilayer_experiments import (
    Condition,
    MultiLayerAblationExperiment,
    _compact_layer_range,
)
from vlmflowprobe.core.config import Config, DEFAULT_CONFIG


class FakeSAE:
    def __init__(self, n_features):
        self.n_features = n_features


def v2_stats(n_features, offset=0.0):
    """Stats shaped exactly like a real causal_feature_stats.json.

    This is the whole point of the fixture: the published runs matched on `correct_mean`,
    which these files do not have. Building fixtures that *do* have it is how
    tests/test_random_control_sampling.py missed the bug for the entire project.
    """
    return {
        idx: {
            "causal_score": 1e-3 / (idx + 1),
            "activation_mean": offset + (idx % 50) / 100.0,
            "gradient_mean": 1e-2 / (idx + 1),
        }
        for idx in range(n_features)
    }


def make_experiment(layers=(10, 11, 12), n_features=400, with_stats=True):
    config = Config(dict(DEFAULT_CONFIG))
    config.data["multilayer"] = {"encode_positions_only": True}
    config.data["ablation"] = dict(config.data["ablation"], position_type="question")

    saes = {layer: FakeSAE(n_features) for layer in layers}
    catalogs = {layer: list(range(n_features)) for layer in layers}
    stats = {layer: v2_stats(n_features, offset=layer / 1000.0) for layer in layers} if with_stats else {}

    experiment = MultiLayerAblationExperiment.__new__(MultiLayerAblationExperiment)
    experiment.model = None
    experiment.saes = dict(saes)
    experiment.catalogs = {k: list(v) for k, v in catalogs.items()}
    experiment.feature_stats = dict(stats)
    experiment.config = config
    return experiment


class TestJointRandomSampling(unittest.TestCase):
    def test_per_layer_counts_match_the_binding_set(self):
        exp = make_experiment()
        binding = {10: list(range(200)), 11: list(range(150)), 12: list(range(40))}

        control = exp.sample_joint_random(binding, rng=random.Random(0))

        self.assertEqual(sorted(control), sorted(binding))
        for layer, features in control.items():
            self.assertEqual(len(features), len(binding[layer]), msg=f"layer {layer}")

    def test_control_is_disjoint_from_binding_per_layer(self):
        exp = make_experiment()
        binding = {10: list(range(100)), 11: list(range(100))}

        control = exp.sample_joint_random(binding, rng=random.Random(1))

        for layer, features in control.items():
            self.assertEqual(set(features) & set(binding[layer]), set())
            self.assertEqual(len(set(features)), len(features), "no repeats within a layer")

    def test_draws_are_independent_across_layers_and_repeats(self):
        exp = make_experiment()
        binding = {10: list(range(80)), 11: list(range(80)), 12: list(range(80))}
        rng = random.Random(7)

        draws = [exp.sample_joint_random(binding, rng=rng) for _ in range(5)]

        # Different sets on each repeat...
        signatures = {tuple(tuple(d[layer]) for layer in sorted(d)) for d in draws}
        self.assertEqual(len(signatures), 5, "control sets collapsed across repeats")
        # ...and a layer must not simply reuse another layer's set.
        for draw in draws:
            self.assertNotEqual(draw[10], draw[11])
            self.assertNotEqual(draw[11], draw[12])

    def test_strict_matching_raises_when_the_metric_is_absent(self):
        # The exact failure that produced the published uniform controls.
        exp = make_experiment()
        binding = {10: list(range(50))}

        with self.assertRaises(ValueError) as ctx:
            exp.sample_joint_random(
                binding,
                rng=random.Random(0),
                matched_metric="correct_mean",
                strict=True,
            )
        self.assertIn("correct_mean", str(ctx.exception))

    def test_strict_matching_raises_when_no_stats_are_loaded(self):
        exp = make_experiment(with_stats=False)
        with self.assertRaises(ValueError):
            exp.sample_joint_random(
                {10: list(range(20))}, rng=random.Random(0), strict=True
            )

    def test_activation_mean_matching_succeeds(self):
        exp = make_experiment()
        control = exp.sample_joint_random(
            {10: list(range(50))},
            rng=random.Random(0),
            matched_metric="activation_mean",
            strict=True,
        )
        self.assertEqual(len(control[10]), 50)

    def test_uniform_sampling_needs_no_stats(self):
        exp = make_experiment(with_stats=False)
        control = exp.sample_joint_random(
            {10: list(range(30))}, rng=random.Random(0), sampling="uniform", strict=True
        )
        self.assertEqual(len(control[10]), 30)

    def test_empty_binding_layer_yields_empty_control_layer(self):
        exp = make_experiment()
        control = exp.sample_joint_random(
            {10: list(range(20)), 11: []}, rng=random.Random(0)
        )
        self.assertEqual(control[11], [])

    def test_effective_sampling_reports_the_real_regime_per_layer(self):
        exp = make_experiment(layers=(10, 11))
        self.assertEqual(
            exp.effective_sampling("matched", "correct_mean"),
            {"10": "uniform_metric_missing", "11": "uniform_metric_missing"},
        )
        self.assertEqual(
            exp.effective_sampling("matched", "activation_mean"),
            {"10": "matched_activation_mean", "11": "matched_activation_mean"},
        )


class TestFeatureSelection(unittest.TestCase):
    def test_top_k_reads_the_head_of_the_catalog(self):
        exp = make_experiment()
        self.assertEqual(exp.top_k(10, 5), [0, 1, 2, 3, 4])

    def test_top_k_beyond_the_catalog_raises(self):
        exp = make_experiment(n_features=400)
        exp.catalogs[10] = list(range(200))
        with self.assertRaises(ValueError):
            exp.top_k(10, 500)

    def test_features_for_accepts_a_scalar_or_per_layer_budget(self):
        exp = make_experiment()
        flat = exp.features_for([10, 11], 40)
        self.assertEqual({l: len(v) for l, v in flat.items()}, {10: 40, 11: 40})

        # The budget- and mass-matched arms need different k per layer.
        uneven = exp.features_for([10, 11], {10: 200, 11: 350})
        self.assertEqual({l: len(v) for l, v in uneven.items()}, {10: 200, 11: 350})


class TestCondition(unittest.TestCase):
    def test_totals_and_layers(self):
        cond = Condition(
            condition_id="joint_L10-12",
            kind="sae",
            features={12: [1, 2], 10: [1], 11: [1, 2, 3]},
        )
        self.assertEqual(cond.layers, [10, 11, 12])
        self.assertEqual(cond.total_features, 6)

    def test_describe_mentions_layers_counts_and_knockout(self):
        cond = Condition(
            condition_id="combined_L11",
            kind="combined",
            features={11: [1, 2]},
            knockout_layers=tuple(range(12, 32)),
        )
        text = cond.describe()
        self.assertIn("combined_L11", text)
        self.assertIn("2 total", text)
        self.assertIn("12-31", text)

    def test_compact_layer_range(self):
        self.assertEqual(_compact_layer_range([10, 11, 12, 13, 14]), "10-14")
        self.assertEqual(_compact_layer_range([10, 12, 14]), "10,12,14")
        self.assertEqual(_compact_layer_range([7]), "7")
        self.assertEqual(_compact_layer_range([]), "none")


if __name__ == "__main__":
    unittest.main()
