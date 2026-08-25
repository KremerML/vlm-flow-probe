"""Gradient-based causal feature identification for SAE features.

Instead of correlating feature activations with prediction correctness (v1),
this computes d(output_logit) / d(feature_activation) via backpropagation.
Features with high |gradient * activation| are causally important — changing
their activation directly affects the model's output.

Theoretical grounding:
  - Marks et al. (2024), "Sparse Feature Circuits" (arXiv:2403.19647)
  - Agrawal et al. (2025), "SAEs Are Good for Steering" (arXiv:2505.20063)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vlmflowprobe.adapters.base import ModelAdapter
from vlmflowprobe.data.loading import iter_batches
from vlmflowprobe.positions import COLLECTION_POLICY, resolve_positions


class CausalFeatureIdentifier:
    """Identify causally important SAE features via gradient attribution."""

    def __init__(
        self,
        sae: torch.nn.Module,
        adapter: ModelAdapter,
        dataset,
        layer_idx: int,
        activation_site: str = "attn_out",
    ):
        self.sae = sae
        self.adapter = adapter
        self.dataset = dataset
        self.layer_idx = layer_idx
        self.activation_site = activation_site
        self.feature_stats: Dict[int, Dict[str, float]] = {}
        self.summary: Optional[Dict[str, Any]] = None

    def compute_causal_scores(
        self,
        position_type: str = "question",
        max_samples: Optional[int] = None,
        target: str = "margin",
        show_progress: bool = False,
    ) -> Dict[int, Dict[str, float]]:
        """Compute gradient-based causal scores for all SAE features.

        For each sample, inserts the SAE into the forward pass at the target
        layer, computes the output logit (or margin), and backpropagates to
        get d(output)/d(feature_activation). The causal score for each feature
        is |gradient| * |activation|, averaged across samples.

        Args:
            position_type: Which token positions to aggregate over
                ("question", "all", "image", "last").
            max_samples: Cap on number of samples to process.
            target: What to differentiate — "margin" (true_logit - false_logit)
                or "correct_logit" (true_logit only).
            show_progress: Show tqdm progress bar.

        Returns:
            Dict mapping feature index to stats dict with keys:
            causal_score, activation_mean, gradient_mean.
        """
        tokenizer = self.adapter.tokenizer
        model = self.adapter.model

        model.requires_grad_(False)
        self.sae.requires_grad_(False)
        model.eval()
        self.sae.eval()

        n_features = self.sae.n_features
        causal_accum = torch.zeros(n_features, dtype=torch.float64)
        act_accum = torch.zeros(n_features, dtype=torch.float64)
        grad_accum = torch.zeros(n_features, dtype=torch.float64)
        n_processed = 0
        n_skipped = 0

        for batch, line in iter_batches(
            self.dataset,
            self.adapter,
            max_samples=max_samples,
            show_progress=show_progress,
            progress_desc="Causal scoring",
        ):
            detail = self.dataset.dataset_dict[line["q_id"]]
            true_answer = detail.get("true option", detail.get("answer", "")).strip()
            false_answer = detail.get("false option", "").strip()

            true_ids = tokenizer.encode(true_answer, add_special_tokens=False)
            if not true_ids:
                n_skipped += 1
                continue
            if target == "margin" and not false_answer:
                n_skipped += 1
                continue
            false_ids = tokenizer.encode(false_answer, add_special_tokens=False) if false_answer else []

            true_target_id = true_ids[0]
            false_target_id = false_ids[0] if false_ids else None

            positions = resolve_positions(
                position_type, batch, self.adapter, policy=COLLECTION_POLICY, line=line
            )

            features_buffer: Dict[str, Any] = {}
            sae_ref = self.sae

            def sae_hook(module, inp, output):
                acts = output[0] if isinstance(output, (tuple, list)) else output
                acts_sae = acts.to(
                    sae_ref.encoder.weight.device,
                    dtype=sae_ref.encoder.weight.dtype,
                )
                z_raw = sae_ref.encode(acts_sae)
                z = z_raw.detach().requires_grad_(True)
                z.retain_grad()
                features_buffer["z"] = z
                recon = sae_ref.decode(z)
                recon = recon.to(acts.device, dtype=acts.dtype)
                if recon.shape != acts.shape:
                    recon = recon.view_as(acts)
                if isinstance(output, (tuple, list)):
                    return (recon,) + output[1:]
                return recon

            layer_module = self.adapter.layer_module(self.layer_idx, self.activation_site)
            handle = layer_module.register_forward_hook(sae_hook)

            try:
                with torch.enable_grad():
                    logits = self.adapter.forward(batch)[0]
                    true_logit = logits[-1, true_target_id]

                    if target == "margin" and false_target_id is not None:
                        false_logit = logits[-1, false_target_id]
                        objective = true_logit - false_logit
                    else:
                        objective = true_logit

                    objective.backward()

                z = features_buffer.get("z")
                if z is None or z.grad is None:
                    n_skipped += 1
                    continue

                grad = z.grad
                z_val = z.detach()
                while grad.ndim > 2:
                    grad = grad[0]
                while z_val.ndim > 2:
                    z_val = z_val[0]

                if positions:
                    valid_pos = [p for p in positions if p < grad.shape[0]]
                    if valid_pos:
                        grad_sel = grad[valid_pos, :]
                        z_sel = z_val[valid_pos, :]
                    else:
                        n_skipped += 1
                        continue
                else:
                    grad_sel = grad
                    z_sel = z_val

                causal = (grad_sel.abs() * z_sel.abs()).mean(dim=0)
                act_mean = z_sel.abs().mean(dim=0)
                grad_mean = grad_sel.abs().mean(dim=0)

                causal_accum += causal.cpu().to(torch.float64)
                act_accum += act_mean.cpu().to(torch.float64)
                grad_accum += grad_mean.cpu().to(torch.float64)
                n_processed += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    n_skipped += 1
                    continue
                raise
            finally:
                handle.remove()
                model.zero_grad(set_to_none=True)
                features_buffer.clear()
                torch.cuda.empty_cache()

        if n_processed == 0:
            self.summary = {"n_processed": 0, "n_skipped": n_skipped}
            return {}

        causal_scores = (causal_accum / n_processed).numpy()
        act_means = (act_accum / n_processed).numpy()
        grad_means = (grad_accum / n_processed).numpy()

        self.feature_stats = {}
        for i in range(n_features):
            self.feature_stats[i] = {
                "causal_score": float(causal_scores[i]),
                "activation_mean": float(act_means[i]),
                "gradient_mean": float(grad_means[i]),
            }

        self.summary = {
            "n_processed": n_processed,
            "n_skipped": n_skipped,
            "n_features": n_features,
            "target": target,
            "position_type": position_type,
            "causal_score": _percentile_stats(causal_scores),
            "activation_mean": _percentile_stats(act_means),
            "gradient_mean": _percentile_stats(grad_means),
        }
        return self.feature_stats

    def get_top_k_features(
        self, k: int, score_key: str = "causal_score"
    ) -> List[int]:
        ranked = sorted(
            self.feature_stats.items(),
            key=lambda x: x[1].get(score_key, 0.0),
            reverse=True,
        )
        return [idx for idx, _ in ranked[:k]]

    def save_results(self, output_dir: str) -> None:
        import json, os
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "causal_feature_stats.json"), "w") as f:
            json.dump(self.feature_stats, f, indent=2)
        if self.summary:
            with open(os.path.join(output_dir, "causal_summary.json"), "w") as f:
                json.dump(self.summary, f, indent=2)

    def export_catalog(self, output_dir: str, top_k: int = 200) -> str:
        """Export a feature catalog compatible with the ablation pipeline."""
        import json, os
        from vlmflowprobe.features.feature_catalog import FeatureCatalog

        top_features = self.get_top_k_features(top_k)
        catalog = FeatureCatalog()
        for feat_idx in top_features:
            stats = self.feature_stats.get(feat_idx, {})
            catalog.add_feature(feat_idx, {
                "name": f"feature_{feat_idx}",
                "type": "causal",
                "causal_score": stats.get("causal_score", 0.0),
                "activation_mean": stats.get("activation_mean", 0.0),
                "gradient_mean": stats.get("gradient_mean", 0.0),
            })
        path = os.path.join(output_dir, "causal_feature_catalog.json")
        catalog.export_to_json(path)
        return path


def _percentile_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
