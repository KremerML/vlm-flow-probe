"""Ablate SAE features at several layers in a single forward pass.

Single-layer ablation recovers only 50-68% of the attention-knockout ceiling at the same
layer. The leading explanation is that the model re-reads the image at layers L+1..31, so
whatever a single ablation removes is partly restored downstream. Testing that needs the
intervention applied at every layer of a span at once, which is what this class does.

It reuses ``FeatureAblator`` wholesale -- the sampling loop, the baseline handling, the
margin scoring and the result schema are all inherited. Only two things are overridden:
which hooks get registered (``_register_sae_hooks``) and how the per-sample diagnostics are
reduced (``_summarize_diagnostics``, which gains a per-layer breakdown).

Encoding semantics are **live**: the hook at layer L+1 encodes a residual stream that the
hook at layer L has already perturbed. That is deliberate. The hypothesis is about
downstream compensation, so if layer 12 re-derives a feature that layer 11's ablation
removed, the layer-12 hook must see and remove the re-derived magnitude. Encoding from
clean cached activations instead would subtract that feature at its unperturbed magnitude,
systematically under-removing precisely the effect under test and biasing the result toward
"no redundancy". It also matches the multi-layer attention knockout this is compared
against, where every blocked layer likewise operates on an already-perturbed stream.
"""

from typing import Any, Dict, List, Optional

from vlmflowprobe.ablation.feature_ablator import FeatureAblator
from vlmflowprobe.hooks import get_target_module


class MultiLayerFeatureAblator(FeatureAblator):
    """Ablates SAE features at a set of layers simultaneously, one SAE per layer."""

    def __init__(
        self,
        model,
        saes: Dict[int, Any],
        activation_site: str = "attn_out",
        encode_positions_only: bool = True,
    ):
        if not saes:
            raise ValueError("saes is empty; pass at least one {layer: SparseAutoencoder}")
        saes = {int(layer): sae for layer, sae in saes.items()}
        # layer_idx is inherited but unused by this subclass; point it at the first layer so
        # anything reading it sees something coherent rather than a stale default.
        super().__init__(
            model,
            sae=saes[min(saes)],
            layer_idx=min(saes),
            activation_site=activation_site,
        )
        self.saes = dict(sorted(saes.items()))
        self.layers: List[int] = list(self.saes)
        self.encode_positions_only = encode_positions_only

    def _register_sae_hooks(
        self,
        feature_indices,
        positions: Optional[List[int]],
        mode: str,
        delta_scale: float,
        operation: str,
        operation_scale: float,
        diagnostics_buffer: List[Dict[str, float]],
    ) -> List[Any]:
        """Register one hook per layer named in ``feature_indices``.

        ``feature_indices`` is a ``{layer: [feature ids]}`` mapping, and the distinction
        between the two ways a layer can contribute nothing is load-bearing:

        * a layer **absent** from the mapping is not hooked at all;
        * a layer mapped to ``[]`` is hooked but ablates nothing -- a pass-through.

        In ``replace`` mode those differ, because a pass-through hook still swaps the
        hidden state for the SAE reconstruction and so injects reconstruction error. The
        pass-through form is what measures that error; the absent form is what excludes a
        layer from the experiment.
        """
        features_by_layer = self._normalize_feature_indices(feature_indices)

        unknown = sorted(set(features_by_layer) - set(self.saes))
        if unknown:
            raise KeyError(
                f"no SAE loaded for layer(s) {unknown}; "
                f"available layers are {sorted(self.saes)}"
            )

        handles = []
        for layer in sorted(features_by_layer):
            module = get_target_module(self.model, layer, self.activation_site)
            handles.append(
                module.register_forward_hook(
                    self.create_ablation_hook(
                        features_by_layer[layer],
                        positions=positions,
                        mode=mode,
                        delta_scale=delta_scale,
                        operation=operation,
                        operation_scale=operation_scale,
                        diagnostics_buffer=diagnostics_buffer,
                        sae=self.saes[layer],
                        diagnostics_tag=layer,
                        encode_positions_only=self.encode_positions_only,
                    )
                )
            )
        return handles

    def _summarize_diagnostics(
        self, sample_diagnostics: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Base fields plus a per-layer breakdown.

        The inherited fields average over every hook call, which with N hooked layers
        blends N layers together and makes ``perturb_relative_norm`` incomparable to a
        single-layer run. ``perturb_by_layer`` keeps the per-layer values, and
        ``perturb_total_relative_norm`` sums them so the total perturbation budget of a
        condition is directly readable.
        """
        summary = super()._summarize_diagnostics(sample_diagnostics)

        by_layer: Dict[int, List[Dict[str, float]]] = {}
        for entry in sample_diagnostics:
            if "layer" in entry:
                by_layer.setdefault(int(entry["layer"]), []).append(entry)

        per_layer = {}
        for layer, entries in sorted(by_layer.items()):
            n = len(entries)
            per_layer[str(layer)] = {
                "mean_delta_norm": sum(e["delta_norm"] for e in entries) / n,
                "mean_acts_norm": sum(e["acts_norm"] for e in entries) / n,
                "relative_norm": sum(e["relative_norm"] for e in entries) / n,
                "calls": n,
            }

        summary["perturb_by_layer"] = per_layer
        summary["perturb_total_relative_norm"] = (
            sum(v["relative_norm"] for v in per_layer.values()) if per_layer else None
        )
        summary["perturb_layers"] = sorted(by_layer)
        return summary

    @staticmethod
    def _normalize_feature_indices(feature_indices) -> Dict[int, List[int]]:
        """Accept ``{layer: [ids]}`` and coerce keys and ids to int."""
        if not isinstance(feature_indices, dict):
            raise TypeError(
                "MultiLayerFeatureAblator expects feature_indices as a {layer: [feature ids]} "
                f"mapping, got {type(feature_indices).__name__}"
            )
        return {
            int(layer): [int(f) for f in features]
            for layer, features in feature_indices.items()
        }
