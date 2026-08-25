"""Conditions and controls for the multi-layer ablation study.

A ``Condition`` is one intervention evaluated on the shared sample set: SAE features zeroed
at a set of layers, attention knocked out at a set of layers, both together, or neither.
``MultiLayerAblationExperiment`` runs them and draws the joint random controls.

Two things here differ from the single-layer path and are the reason this module exists.

**Joint controls.** A control for a multi-layer condition has to be a control at every layer
at once: the same per-layer feature counts as the binding set, each layer matched against its
own dictionary's statistics, and every one of the N sets an independent draw across all
layers. Sharing one per-layer set across layers, or matching against a pooled statistic,
would not be a control for the intervention actually performed.

**Matching that fails loudly.** Every published v2 run requested matched controls and
silently got uniform ones, because the configs matched on ``correct_mean`` and the v2 stats
files carry only ``causal_score`` / ``activation_mean`` / ``gradient_mean``. The controls
ended up being near-dead features and the z-scores were inflated. Here matching defaults to
``activation_mean`` and ``strict=True`` raises rather than falling back.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vlmflowprobe.ablation.ablation_experiments import AblationExperiment
from vlmflowprobe.ablation.multilayer_ablator import MultiLayerFeatureAblator
from vlmflowprobe.ablation.sample_cache import baseline_cache, positions_cache
from vlmflowprobe.knockout.knockout_utils import (
    build_block_config_for_layers,
    estimate_inputs_embeds_shape,
    resolve_flow_ranges,
)

# Condition kinds.
KIND_SAE = "sae"
KIND_KNOCKOUT = "knockout"
KIND_COMBINED = "combined"
KIND_PASSTHROUGH = "passthrough"
KIND_NONE = "none"


@dataclass(frozen=True)
class Condition:
    """One intervention to evaluate.

    ``features`` maps layer -> feature ids. The distinction the ablator enforces matters
    here too: a layer absent from the mapping is not hooked at all, while a layer mapped to
    ``[]`` is hooked but ablates nothing (a pass-through, which in ``replace`` mode still
    injects SAE reconstruction error).
    """

    condition_id: str
    kind: str
    features: Dict[int, List[int]] = field(default_factory=dict)
    knockout_layers: Tuple[int, ...] = ()
    flow: str = "Image->Question"
    mode: str = "replace"
    label: str = ""
    # Joint-control policy, set explicitly at condition-construction time.
    # "full" -> random_control.n_random_sets, "single" -> 1, "none" -> 0.
    # (Replaces the old runner's string-prefix dispatch on condition_id.)
    control_kind: str = "none"

    def n_controls(self, n_full: int) -> int:
        if self.kind not in (KIND_SAE, KIND_COMBINED):
            return 0
        return {"full": n_full, "single": 1, "none": 0}[self.control_kind]

    @property
    def layers(self) -> List[int]:
        return sorted(self.features)

    @property
    def total_features(self) -> int:
        return sum(len(v) for v in self.features.values())

    def describe(self) -> str:
        parts = [f"{self.condition_id} [{self.kind}]"]
        if self.features:
            counts = ", ".join(f"L{l}:{len(self.features[l])}" for l in self.layers)
            parts.append(f"features {{{counts}}} = {self.total_features} total")
        if self.knockout_layers:
            parts.append(
                f"knockout {self.flow} at layers "
                f"{_compact_layer_range(self.knockout_layers)}"
            )
        if self.kind in (KIND_SAE, KIND_COMBINED, KIND_PASSTHROUGH):
            parts.append(f"mode={self.mode}")
        return " | ".join(parts)


class MultiLayerAblationExperiment:
    """Runs multi-layer conditions and their joint random controls."""

    def __init__(self, model, saes, catalogs, feature_stats, config):
        self.model = model
        self.saes = {int(k): v for k, v in saes.items()}
        self.catalogs = {int(k): [int(f) for f in v] for k, v in catalogs.items()}
        self.feature_stats = {
            int(layer): {int(f): s for f, s in stats.items()}
            for layer, stats in (feature_stats or {}).items()
        }
        self.config = config

        ml_cfg = config.get("multilayer", {})
        ablation_cfg = config.get("ablation", {})
        self.activation_site = config.get("model", {}).get("activation_site", "attn_out")
        self.position_type = ablation_cfg.get("position_type", "question")
        self.mode = ablation_cfg.get("mode", "replace")
        self.delta_scale = float(ablation_cfg.get("delta_scale", 1.0))
        self.operation = str(ablation_cfg.get("operation", "zero")).lower()
        self.operation_scale = float(ablation_cfg.get("operation_scale", 1.0))
        self.logprob_normalize = bool(
            config.get("evaluation", {}).get("logprob_normalize", True)
        )

        self.ablator = MultiLayerFeatureAblator(
            model,
            self.saes,
            activation_site=self.activation_site,
            encode_positions_only=bool(ml_cfg.get("encode_positions_only", True)),
        )

    # ------------------------------------------------------------------ features

    def top_k(self, layer: int, k: int) -> List[int]:
        """The ``k`` highest-scoring causal features at a layer.

        The exported catalogs hold only the top 200, but the budget-matched arm needs a
        concentrated count curve well past that. When ``k`` exceeds the catalog we re-derive
        the ranking from the full stats file, which scores every feature -- the catalog is
        just its head, so the two agree on the overlap.
        """
        layer = int(layer)
        catalog = self.catalogs[layer]
        if k <= len(catalog):
            return list(catalog[:k])

        stats = self.feature_stats.get(layer, {})
        if len(stats) < k:
            raise ValueError(
                f"layer {layer}: {k} features requested but the catalog holds "
                f"{len(catalog)} and the stats file {len(stats)}. Point stats_paths at the "
                f"full causal_feature_stats.json, or lower the requested k"
            )
        ranked = sorted(
            stats.items(),
            key=lambda item: (-float(item[1].get("causal_score", 0.0)), item[0]),
        )
        return [int(feature) for feature, _ in ranked[:k]]

    def features_for(self, layers: Sequence[int], k) -> Dict[int, List[int]]:
        """Build a ``{layer: features}`` mapping.

        ``k`` is either an int applied to every layer or a ``{layer: k}`` mapping, which is
        what the budget-matched and mass-matched arms need.
        """
        if isinstance(k, dict):
            per_layer = {int(layer): int(k[int(layer)]) for layer in layers}
        else:
            per_layer = {int(layer): int(k) for layer in layers}
        return {layer: self.top_k(layer, count) for layer, count in sorted(per_layer.items())}

    # ------------------------------------------------------------------ controls

    def sample_joint_random(
        self,
        binding_by_layer: Dict[int, List[int]],
        rng: random.Random,
        sampling: str = "matched",
        matched_metric: str = "activation_mean",
        strict: bool = True,
    ) -> Dict[int, List[int]]:
        """Draw one joint control: an independent matched set at every layer.

        Per-layer counts mirror the binding set, so the control matches the total
        perturbation budget *and* how it is distributed across layers.
        """
        control: Dict[int, List[int]] = {}
        for layer, binding in sorted(binding_by_layer.items()):
            if not binding:
                control[layer] = []
                continue

            sae = self.saes[layer]
            n_total = int(sae.n_features)
            binding_set = {f for f in binding if 0 <= f < n_total}
            pool = [i for i in range(n_total) if i not in binding_set]
            target_count = min(len(binding), len(pool))
            stats = self.feature_stats.get(layer, {})

            if sampling in ("matched", "matched_activation", "matched_metric"):
                if strict and not stats:
                    raise ValueError(
                        f"layer {layer}: matched sampling requires feature stats, none loaded"
                    )
                control[layer] = AblationExperiment._sample_matched_random_features(
                    binding_features=binding,
                    pool=pool,
                    target_count=target_count,
                    feature_stats=stats,
                    matched_metric=matched_metric,
                    rng=rng,
                    strict_matching=strict,
                )
            else:
                control[layer] = rng.sample(pool, k=target_count)
        return control

    def effective_sampling(self, sampling: str, matched_metric: str) -> Dict[str, str]:
        """Per-layer report of what the sampler will really do, for the result metadata.

        Keyed off the loaded stats rather than the loaded SAEs so it also works in a dry
        run, where no model or dictionary has been loaded.
        """
        layers = sorted(set(self.feature_stats) | set(self.catalogs))
        return {
            str(layer): AblationExperiment._effective_sampling(
                sampling, self.feature_stats.get(layer), matched_metric
            )
            for layer in layers
        }

    # ------------------------------------------------------------------ running

    def run_condition(
        self,
        dataset,
        condition: Condition,
        sample_records,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> Tuple[List[dict], dict]:
        """Evaluate one condition, returning its per-sample rows and its summary."""
        resolver = None
        if condition.knockout_layers:
            resolver = self._make_block_resolver(
                condition.flow, condition.knockout_layers
            )

        started = time.time()
        rows = self.ablator.batch_ablation_experiment(
            dataset,
            feature_indices=condition.features,
            position_type=self.position_type,
            mode=condition.mode,
            delta_scale=self.delta_scale,
            operation=self.operation,
            operation_scale=self.operation_scale,
            logprob_normalize=self.logprob_normalize,
            apply_sae=bool(condition.features),
            attn_block_resolver=resolver,
            show_progress=show_progress,
            max_samples=max_samples,
            baseline_cache=baseline_cache(sample_records),
            positions_cache=positions_cache(sample_records),
            strict_cache=True,
            progress_desc=progress_desc or f"  {condition.condition_id}",
        )
        summary = self.ablator.compute_ablation_effect(rows)
        summary.update(
            {
                "condition_id": condition.condition_id,
                "kind": condition.kind,
                "layers": condition.layers,
                "features_per_layer": {
                    str(l): len(condition.features[l]) for l in condition.layers
                },
                "total_features": condition.total_features,
                "knockout_layers": list(condition.knockout_layers),
                "flow": condition.flow if condition.knockout_layers else None,
                "mode": condition.mode,
                "n_samples": len(rows),
                "runtime_seconds": round(time.time() - started, 2),
            }
        )
        return rows, summary

    def _make_block_resolver(self, flow: str, layers: Sequence[int]):
        """Per-sample attention block config for an explicit layer set.

        Mirrors ``tools/knockout_sae_pipeline._make_attn_block_resolver`` but takes the
        layer set directly instead of a symmetric window around a centre layer.
        """
        model = self.model
        block_layers = tuple(int(l) for l in layers)

        def resolver(input_ids, image_tensor, image_sizes, dataset, line):
            inputs_embeds_shape = estimate_inputs_embeds_shape(
                model, input_ids, image_tensor, image_sizes
            )
            if inputs_embeds_shape is None:
                return None
            question_text = dataset.dataset_dict[line["q_id"]].get("question", "")
            source_range, target_range = resolve_flow_ranges(
                flow,
                input_ids,
                inputs_embeds_shape,
                question_text,
                dataset.tokenizer,
            )
            if not source_range or not target_range:
                return None
            pairs = [(tgt, src) for src in source_range for tgt in target_range]
            return build_block_config_for_layers(block_layers, pairs)

        return resolver

    # ------------------------------------------------------------------ statistics

    @staticmethod
    def compare_to_controls(
        binding_summary: dict, control_summaries: List[dict]
    ) -> Dict[str, dict]:
        """Binding vs. control significance, reusing the single-layer implementation.

        Same output schema as the single-layer runs -- ``{metric: {binding, random_mean,
        random_std, empirical_p_value, z_score}}`` -- so a multi-layer condition is directly
        comparable to a published one.
        """
        return AblationExperiment._compare_binding_vs_random(
            binding_summary, control_summaries
        )


def _compact_layer_range(layers: Sequence[int]) -> str:
    """Render a layer set as '10-14' when contiguous, else '10,12,14'."""
    ordered = sorted(set(int(l) for l in layers))
    if not ordered:
        return "none"
    if len(ordered) > 1 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ",".join(str(l) for l in ordered)
