import unittest

from vlmflowprobe.ablation.hypothesis_tester import HypothesisTester


class TestHypothesisTester(unittest.TestCase):
    def test_prefers_random_set_distribution_when_available(self):
        tester = HypothesisTester(
            {
                "evaluation": {
                    "primary_metric": "forced_choice_margin",
                    "significance_level": 0.05,
                }
            }
        )
        results = {
            "binding": {"mean_margin_drop": 0.2},
            "random_set_summaries": [
                {"mean_margin_drop": 0.05},
                {"mean_margin_drop": 0.06},
                {"mean_margin_drop": 0.07},
                {"mean_margin_drop": 0.08},
                {"mean_margin_drop": 0.09},
            ],
            "binding_results": [],
            "random_results": [],
        }

        out = tester.test_causal_necessity(results)
        self.assertEqual(out.get("test_type"), "empirical_random_set")
        self.assertIn("random_set_count", out)
        self.assertIsInstance(out.get("p_value"), float)

    def test_falls_back_to_paired_test(self):
        tester = HypothesisTester(
            {
                "evaluation": {
                    "primary_metric": "pred_token_prob",
                    "significance_level": 0.05,
                }
            }
        )
        rows_binding = [
            {"baseline_prob": 0.9, "ablated_prob": 0.8},
            {"baseline_prob": 0.7, "ablated_prob": 0.6},
        ]
        rows_random = [
            {"baseline_prob": 0.9, "ablated_prob": 0.89},
            {"baseline_prob": 0.7, "ablated_prob": 0.69},
        ]
        results = {
            "binding_results": rows_binding,
            "random_results": rows_random,
            "random_set_summaries": [{"mean_probability_drop": 0.01}],
        }

        out = tester.test_causal_necessity(results)
        self.assertEqual(out.get("test_type"), "paired_t")
        self.assertIn("p_value", out)
        self.assertIn("effect_size", out)


if __name__ == "__main__":
    unittest.main()

