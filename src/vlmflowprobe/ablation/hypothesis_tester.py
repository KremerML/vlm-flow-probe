"""Hypothesis testing for attribute-binding features."""

from typing import Dict
import math

from vlmflowprobe.ablation import statistical_analysis as stats


class HypothesisTester:
    """Run hypothesis tests on ablation results."""

    def __init__(self, config):
        self.config = config

    def test_causal_necessity(self, ablation_results: Dict) -> Dict[str, float]:
        binding = ablation_results.get("binding_results", [])
        random_results = ablation_results.get("random_results", [])
        random_set_summaries = ablation_results.get("random_set_summaries", [])
        metric = self.config.get("evaluation", {}).get("primary_metric", "pred_token_prob")
        alpha = self.config.get("evaluation", {}).get("significance_level", 0.05)

        # Prefer random-set summary distribution if available (multi-set control).
        if isinstance(random_set_summaries, list) and len(random_set_summaries) > 1:
            metric_key = self._metric_to_summary_key(metric)
            binding_value = ablation_results.get("binding", {}).get(metric_key)
            random_values = [
                row.get(metric_key) for row in random_set_summaries if row.get(metric_key) is not None
            ]
            if binding_value is not None and random_values:
                random_values = [float(v) for v in random_values]
                random_mean = sum(random_values) / len(random_values)
                var = sum((v - random_mean) ** 2 for v in random_values) / max(1, len(random_values))
                random_std = math.sqrt(var)
                empirical_p = (sum(1 for v in random_values if v >= float(binding_value)) + 1) / (
                    len(random_values) + 1
                )
                supported = bool(empirical_p < alpha)
                effect = (float(binding_value) - random_mean) / (random_std + 1e-8)
                return {
                    "hypothesis_supported": supported,
                    "p_value": float(empirical_p),
                    "effect_size": float(effect),
                    "primary_metric": metric,
                    "test_type": "empirical_random_set",
                    "random_set_count": float(len(random_values)),
                }

        binding_drop = self._extract_drops(binding, metric)
        random_drop = self._extract_drops(random_results, metric)

        t_stat, p_val = stats.paired_t_test(binding_drop, random_drop)
        effect = stats.effect_size_cohens_d(binding_drop, random_drop)

        supported = bool(p_val < alpha)
        return {
            "hypothesis_supported": supported,
            "p_value": p_val,
            "effect_size": effect,
            "primary_metric": metric,
            "test_type": "paired_t",
        }

    @staticmethod
    def _extract_drops(results, metric: str):
        if metric == "gt_token_prob":
            pairs = [
                (r.get("baseline_gt_prob"), r.get("ablated_gt_prob"))
                for r in results
                if r.get("baseline_gt_prob") is not None and r.get("ablated_gt_prob") is not None
            ]
            return [b - a for b, a in pairs]
        return [r.get("baseline_prob", 0.0) - r.get("ablated_prob", 0.0) for r in results]

    @staticmethod
    def _metric_to_summary_key(metric: str) -> str:
        if metric == "forced_choice_margin":
            return "mean_margin_drop"
        if metric == "gt_token_prob":
            return "mean_gt_probability_drop"
        return "mean_probability_drop"
