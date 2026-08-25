"""High-level ablation experiment runner."""

from __future__ import annotations

from typing import Dict, List, Optional
import json
import os
import random
import statistics
import time
import warnings

import numpy as np
import torch

from vlmflowprobe.ablation.feature_ablator import FeatureAblator
from vlmflowprobe.ablation.sample_cache import (
    baseline_cache,
    build_sample_cache,
    positions_cache,
)


def _warn_uniform_fallback(message: str) -> None:
    """Make a silent degradation from matched to uniform sampling impossible to miss.

    This is printed as well as warned because these runs are driven from shell scripts
    whose stdout is the log that gets read afterwards.
    """
    banner = f"MATCHED SAMPLING FELL BACK TO UNIFORM: {message}"
    warnings.warn(banner, RuntimeWarning, stacklevel=3)
    print(f"[random_control] !! {banner}", flush=True)


class AblationExperiment:
    """Runs ablation studies for identified features."""

    def __init__(self, model, sae, config):
        self.model = model
        self.sae = sae
        self.config = config

    def run_three_condition_test(
        self,
        dataset,
        binding_features: List[int],
        feature_stats: Optional[Dict[int, dict]] = None,
        random_seed_offset: int = 0,
        progress_label: Optional[str] = None,
        show_progress: bool = False,
        max_samples: Optional[int] = None,
    ) -> Dict[str, dict]:
        model_cfg = self.config.get("model", {})
        ablation_cfg = self.config.get("ablation", {})
        eval_cfg = self.config.get("evaluation", {})
        random_cfg = self.config.get("random_control", {})

        ablator = FeatureAblator(
            self.model,
            self.sae,
            model_cfg.get("target_layer", 0),
            activation_site=model_cfg.get("activation_site", "residual"),
        )
        position_type = ablation_cfg.get("position_type", "all")
        mode = ablation_cfg.get("mode", "residual")
        delta_scale = float(ablation_cfg.get("delta_scale", 1.0))
        operation = str(ablation_cfg.get("operation", "zero")).lower()
        operation_scale = float(ablation_cfg.get("operation_scale", 1.0))
        logprob_normalize = bool(eval_cfg.get("logprob_normalize", True))

        binding_features = [int(x) for x in binding_features]
        if not binding_features:
            raise ValueError("binding_features is empty; run feature identification first.")

        # The baseline and the resolved positions do not depend on which features are
        # ablated, so resolve them once instead of once per condition. With the default 15
        # random sets that is 16 baselines per sample collapsed into one.
        cache_records = build_sample_cache(
            ablator,
            dataset,
            position_type=position_type,
            logprob_normalize=logprob_normalize,
            max_samples=max_samples,
            show_progress=show_progress,
            progress_desc="Baseline cache",
        )
        cached_baselines = baseline_cache(cache_records)
        cached_positions = positions_cache(cache_records)

        binding_results = ablator.batch_ablation_experiment(
            dataset,
            binding_features,
            position_type=position_type,
            mode=mode,
            delta_scale=delta_scale,
            operation=operation,
            operation_scale=operation_scale,
            logprob_normalize=logprob_normalize,
            show_progress=show_progress,
            max_samples=max_samples,
            baseline_cache=cached_baselines,
            positions_cache=cached_positions,
            strict_cache=True,
        )
        binding_summary = ablator.compute_ablation_effect(binding_results)

        n_random = int(ablation_cfg.get("n_random_features", len(binding_features)))
        n_random_sets = int(ablation_cfg.get("n_random_sets", random_cfg.get("n_random_sets", 1)))
        n_random_sets = max(0, n_random_sets)
        random_sampling = str(ablation_cfg.get("random_sampling", random_cfg.get("sampling", "uniform"))).lower()
        matched_metric = str(random_cfg.get("matched_metric", "correct_mean"))
        # Default False so existing configs reproduce their published numbers; new configs
        # set it true so a missing metric raises instead of silently drawing uniformly.
        strict_matching = bool(random_cfg.get("strict_matching", False))
        seed = int(
            random_cfg.get(
                "seed",
                self.config.get("reproducibility", {}).get(
                    "seed",
                    self.config.get("training", {}).get("seed", 42),
                ),
            )
        )
        rng = random.Random(seed + int(random_seed_offset))
        normalized_stats = self._normalize_feature_stats(feature_stats)

        random_results_first = []
        random_feature_sets: List[List[int]] = []
        random_set_summaries: List[dict] = []
        random_start = time.time()
        for set_idx in range(n_random_sets):
            if show_progress:
                elapsed = time.time() - random_start
                done = set_idx
                remaining = None
                if done > 0:
                    avg_per_set = elapsed / done
                    remaining = avg_per_set * (n_random_sets - done)
                label = progress_label or "Ablation"
                if remaining is None:
                    print(
                        f"[{label}] Random control set {set_idx + 1}/{n_random_sets} starting...",
                        flush=True,
                    )
                else:
                    print(
                        f"[{label}] Random control set {set_idx + 1}/{n_random_sets} starting "
                        f"(elapsed {self._format_duration(elapsed)}, "
                        f"ETA {self._format_duration(remaining)})",
                        flush=True,
                    )

            random_features = self._sample_random_features(
                binding_features=binding_features,
                n_random_features=n_random,
                sampling=random_sampling,
                feature_stats=normalized_stats,
                matched_metric=matched_metric,
                rng=rng,
                strict_matching=strict_matching,
            )
            random_feature_sets.append(random_features)

            set_start = time.time()
            set_results = ablator.batch_ablation_experiment(
                dataset,
                random_features,
                position_type=position_type,
                mode=mode,
                delta_scale=delta_scale,
                operation=operation,
                operation_scale=operation_scale,
                logprob_normalize=logprob_normalize,
                show_progress=show_progress and set_idx == 0,
                max_samples=max_samples,
                baseline_cache=cached_baselines,
                positions_cache=cached_positions,
                strict_cache=True,
            )
            set_summary = ablator.compute_ablation_effect(set_results)
            set_summary["set_index"] = set_idx
            set_summary["feature_count"] = len(random_features)
            random_set_summaries.append(set_summary)
            if show_progress:
                label = progress_label or "Ablation"
                print(
                    f"[{label}] Random control set {set_idx + 1}/{n_random_sets} completed "
                    f"in {self._format_duration(time.time() - set_start)}",
                    flush=True,
                )
            if set_idx == 0:
                random_results_first = set_results

        random_summary = self._aggregate_random_summaries(random_set_summaries)
        significance = self._compare_binding_vs_random(binding_summary, random_set_summaries)
        baseline_summary = {
            "baseline_accuracy": binding_summary["baseline_accuracy"],
        }

        return {
            "baseline": baseline_summary,
            "binding": binding_summary,
            "random": random_summary,
            "binding_results": binding_results,
            "random_results": random_results_first,
            "random_set_summaries": random_set_summaries,
            "random_feature_sets": random_feature_sets,
            "significance": significance,
            "ablation_settings": {
                "position_type": position_type,
                "mode": mode,
                "delta_scale": delta_scale,
                "operation": operation,
                "operation_scale": operation_scale,
            },
            "random_control_settings": {
                "n_random_features": n_random,
                "n_random_sets": n_random_sets,
                "sampling": random_sampling,
                "matched_metric": matched_metric,
                "strict_matching": strict_matching,
                # What the sampler actually did, not merely what was requested. The
                # published v2 runs asked for matched sampling and silently got uniform;
                # recording the effective regime makes that visible in the result file.
                "sampling_effective": self._effective_sampling(
                    random_sampling, normalized_stats, matched_metric
                ),
                "seed": seed + int(random_seed_offset),
            },
            "evaluation_settings": {
                "primary_metric": eval_cfg.get("primary_metric", "pred_token_prob"),
                "logprob_normalize": logprob_normalize,
            },
        }

    def save_results(self, results: Dict, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def _sample_random_features(
        self,
        binding_features: List[int],
        n_random_features: int,
        sampling: str,
        feature_stats: Dict[int, dict],
        matched_metric: str,
        rng: random.Random,
        strict_matching: bool = False,
    ) -> List[int]:
        n_total = int(self.sae.n_features)
        target_count = int(n_random_features) if int(n_random_features) > 0 else len(binding_features)
        target_count = max(1, min(target_count, n_total))
        binding_set = {idx for idx in binding_features if 0 <= idx < n_total}

        pool = [idx for idx in range(n_total) if idx not in binding_set]
        if not pool:
            pool = list(range(n_total))
        target_count = min(target_count, len(pool))
        if target_count == 0:
            return []

        if sampling in ("matched", "matched_activation", "matched_metric") and feature_stats:
            return self._sample_matched_random_features(
                binding_features=binding_features,
                pool=pool,
                target_count=target_count,
                feature_stats=feature_stats,
                matched_metric=matched_metric,
                rng=rng,
                strict_matching=strict_matching,
            )
        if sampling in ("matched", "matched_activation", "matched_metric"):
            _warn_uniform_fallback(
                f"matched sampling requested but no feature_stats were supplied; "
                f"drawing uniformly instead"
            )
        return rng.sample(pool, k=target_count)

    @staticmethod
    def _effective_sampling(
        sampling: str, feature_stats: Optional[Dict[int, dict]], matched_metric: str
    ) -> str:
        """Resolve what the sampler will really do, given the stats it was handed."""
        if sampling not in ("matched", "matched_activation", "matched_metric"):
            return "uniform"
        if not feature_stats:
            return "uniform_no_stats"
        has_metric = any(
            AblationExperiment._extract_metric_value(stats, matched_metric) is not None
            for stats in feature_stats.values()
        )
        return f"matched_{matched_metric}" if has_metric else "uniform_metric_missing"

    @staticmethod
    def _sample_matched_random_features(
        binding_features: List[int],
        pool: List[int],
        target_count: int,
        feature_stats: Dict[int, dict],
        matched_metric: str,
        rng: random.Random,
        strict_matching: bool = False,
    ) -> List[int]:
        selected: List[int] = []
        available = set(pool)
        candidate_stats = {
            idx: AblationExperiment._extract_metric_value(feature_stats.get(idx, {}), matched_metric)
            for idx in available
        }
        valid_metric_pool = {idx for idx in available if candidate_stats.get(idx) is not None}

        # Every published v2 run silently landed here: configs asked for matched sampling on
        # `correct_mean`, a key the v2 stats files do not carry, so every metric lookup
        # returned None, the pool was empty, and every draw fell through to uniform. The
        # controls ended up as near-dead features and the z-scores were inflated. Refuse to
        # do that quietly ever again.
        if not valid_metric_pool:
            message = (
                f"matched sampling on {matched_metric!r} found no feature with that metric "
                f"(checked {len(available)} candidates); every draw would fall back to "
                f"uniform. Check that the stats file carries the metric you asked for"
            )
            if strict_matching:
                raise ValueError(message)
            _warn_uniform_fallback(message)
        # Randomized nearest-neighbor matching width. Keeps candidates close in metric space
        # while allowing independent random control sets across repeats.
        neighbor_width = 16

        for feature_idx in binding_features[:target_count]:
            target_value = AblationExperiment._extract_metric_value(
                feature_stats.get(feature_idx, {}), matched_metric
            )
            if target_value is None or not valid_metric_pool:
                if not available:
                    break
                choice = rng.choice(sorted(available))
            else:
                scored = sorted(
                    (abs(candidate_stats[idx] - target_value), idx)
                    for idx in valid_metric_pool
                )
                window = scored[: min(neighbor_width, len(scored))]
                if len(window) == 1:
                    choice = window[0][1]
                else:
                    # Prefer closer matches but sample stochastically.
                    max_dist = window[-1][0]
                    if max_dist <= 0.0:
                        choice = rng.choice([idx for _, idx in window])
                    else:
                        weights = [(max_dist - dist) + 1e-12 for dist, _ in window]
                        choice = rng.choices(
                            [idx for _, idx in window],
                            weights=weights,
                            k=1,
                        )[0]
                valid_metric_pool.discard(choice)
            if choice in available:
                available.remove(choice)
                selected.append(choice)

        if len(selected) < target_count and available:
            remainder = rng.sample(list(available), k=min(target_count - len(selected), len(available)))
            selected.extend(remainder)

        if len(selected) < target_count:
            # Should rarely happen; fallback to pure uniform sampling from entire non-binding pool.
            fallback_pool = list(pool)
            extra = [idx for idx in fallback_pool if idx not in selected]
            if extra:
                selected.extend(rng.sample(extra, k=min(target_count - len(selected), len(extra))))
        return selected[:target_count]

    @staticmethod
    def _normalize_feature_stats(feature_stats: Optional[Dict[int, dict]]) -> Dict[int, dict]:
        if not feature_stats:
            return {}
        normalized = {}
        for key, value in feature_stats.items():
            try:
                feature_idx = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                normalized[feature_idx] = value
        return normalized

    @staticmethod
    def _extract_metric_value(stats: Dict, metric: str) -> Optional[float]:
        if not isinstance(stats, dict):
            return None
        if metric in stats and stats[metric] is not None:
            try:
                return float(stats[metric])
            except (TypeError, ValueError):
                return None
        for fallback in ("correct_mean", "ratio", "diff", "incorrect_mean"):
            if fallback in stats and stats[fallback] is not None:
                try:
                    return float(stats[fallback])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _aggregate_random_summaries(set_summaries: List[dict]) -> Dict[str, Optional[float]]:
        if not set_summaries:
            return {}
        if len(set_summaries) == 1:
            summary = dict(set_summaries[0])
            summary.pop("set_index", None)
            summary["n_sets"] = 1
            return summary

        aggregate = dict(set_summaries[0])
        aggregate.pop("set_index", None)
        aggregate["n_sets"] = len(set_summaries)
        metric_keys = [
            "baseline_accuracy",
            "ablated_accuracy",
            "accuracy_drop",
            "mean_probability_drop",
            "baseline_gt_probability",
            "ablated_gt_probability",
            "mean_gt_probability_drop",
            "baseline_margin",
            "ablated_margin",
            "mean_margin_drop",
            "mean_relative_perturbation",
        ]
        for metric in metric_keys:
            values = [float(item[metric]) for item in set_summaries if item.get(metric) is not None]
            if not values:
                aggregate[metric] = None
                aggregate[f"{metric}_std"] = None
                aggregate[f"{metric}_ci95_low"] = None
                aggregate[f"{metric}_ci95_high"] = None
                continue
            aggregate[metric] = float(statistics.fmean(values))
            aggregate[f"{metric}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
            if len(values) > 1:
                ci_low, ci_high = np.percentile(values, [2.5, 97.5])
                aggregate[f"{metric}_ci95_low"] = float(ci_low)
                aggregate[f"{metric}_ci95_high"] = float(ci_high)
            else:
                aggregate[f"{metric}_ci95_low"] = float(values[0])
                aggregate[f"{metric}_ci95_high"] = float(values[0])
        return aggregate

    @staticmethod
    def _compare_binding_vs_random(binding_summary: dict, random_set_summaries: List[dict]) -> Dict[str, dict]:
        if not random_set_summaries:
            return {}
        metrics = [
            "accuracy_drop",
            "mean_probability_drop",
            "mean_gt_probability_drop",
            "mean_margin_drop",
        ]
        out: Dict[str, dict] = {}
        for metric in metrics:
            binding_value = binding_summary.get(metric)
            random_values = [item.get(metric) for item in random_set_summaries if item.get(metric) is not None]
            if binding_value is None or not random_values:
                continue
            random_values = [float(v) for v in random_values]
            rand_mean = float(statistics.fmean(random_values))
            rand_std = float(statistics.pstdev(random_values)) if len(random_values) > 1 else 0.0
            p_empirical = AblationExperiment._empirical_p_value(
                observed=float(binding_value),
                distribution=random_values,
                greater_is_more_extreme=True,
            )
            z_score = None
            if rand_std > 0.0:
                z_score = float((float(binding_value) - rand_mean) / rand_std)
            out[metric] = {
                "binding": float(binding_value),
                "random_mean": rand_mean,
                "random_std": rand_std,
                "empirical_p_value": p_empirical,
                "z_score": z_score,
            }
        return out

    @staticmethod
    def _empirical_p_value(
        observed: float,
        distribution: List[float],
        greater_is_more_extreme: bool = True,
    ) -> float:
        if not distribution:
            return 1.0
        if greater_is_more_extreme:
            extreme = sum(1 for x in distribution if x >= observed)
        else:
            extreme = sum(1 for x in distribution if x <= observed)
        # +1 smoothing avoids zero p-values for small random-set counts.
        return float((extreme + 1) / (len(distribution) + 1))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        total = int(seconds)
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
