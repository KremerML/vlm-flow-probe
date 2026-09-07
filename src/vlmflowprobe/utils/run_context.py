"""Where an analysis gets its layer set and its model depth.

The analysis CLIs used to carry the LLaVA layer numbers as literals -- ``LAYERS
= (10, 11, 12, 13, 14)``, ``for layer in (11, 14)``, a hand-written table of
span tuples. That is fine for exactly one 32-layer model and wrong for the
second one. Layers now come from the run being analysed, in this order:

1. ``--config``, when the caller names one;
2. ``provenance.json`` in the run directory, which carries the fully-resolved
   config every stage stamps beside its outputs;
3. the package defaults, announced as such.

Depth fractions (``layer / n_layers``) make layer numbers comparable across
models of different depths: layer 11 of 32 and layer 14 of 42 are the same
place in the stack. ``n_layers`` is read from the ``model_geometry`` block that
the runners stamp into provenance; runs made before that existed report no
depth rather than a guessed one.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy
import json
import os


def span_label(layers: Sequence[int]) -> str:
    """How a span is named in a report: ``{14}``, ``{10,11,12}``, ``{11-14}``.

    Short spans are listed member by member; from four layers up a contiguous
    span is compacted, which is where the literal spelling stops being readable.
    Distinct from ``_compact_layer_range``, which spells *condition ids* and
    compacts from two layers up.
    """
    ordered = sorted(int(layer) for layer in layers)
    if not ordered:
        return "{}"
    contiguous = ordered == list(range(ordered[0], ordered[-1] + 1))
    if len(ordered) >= 4 and contiguous:
        return "{" + f"{ordered[0]}-{ordered[-1]}" + "}"
    return "{" + ",".join(str(layer) for layer in ordered) + "}"


@dataclass(frozen=True)
class SpanPair:
    """One span with the condition ids of both arms measured over it.

    The runner's ids are irregular by design -- the full span is
    ``joint_``/``span_knockout_`` while sub-spans are
    ``nested_``/``nested_knockout_`` -- so the mapping is built here once and
    both analysis CLIs read it, instead of each carrying its own table.
    """

    kind: str
    label: str
    ablation_id: str
    knockout_id: str
    control_id: str
    layers: Tuple[int, ...]

    @property
    def span_size(self) -> int:
        return len(self.layers)


@dataclass
class RunContext:
    """The resolved config behind a run, plus the model geometry it used."""

    config: Dict[str, Any] = field(default_factory=dict)
    n_layers: Optional[int] = None
    #: Human-readable provenance of the config, for the report header.
    source: str = "package defaults"

    # ------------------------------------------------------------------ layers
    @property
    def layers(self) -> List[int]:
        """The layer set the run intervened on."""
        multilayer = self.config.get("multilayer", {}) or {}
        layers = multilayer.get("layers")
        if not layers:
            # run_controls takes its layers from whichever layers have statistics.
            layers = list((multilayer.get("stats_paths") or {}).keys())
        return sorted(int(layer) for layer in layers)

    @property
    def nested_spans(self) -> List[List[int]]:
        return self._spans("nested")

    @property
    def non_nested_spans(self) -> List[List[int]]:
        return self._spans("non_nested")

    @property
    def sensitivity_span(self) -> List[int]:
        span = (self.config.get("conditions", {}) or {}).get("sensitivity_span") or []
        return [int(layer) for layer in span]

    @property
    def a0_regression_layer(self) -> Optional[int]:
        """Layer of the single-layer regression condition, when the run has one."""
        value = (self.config.get("conditions", {}) or {}).get("a0_regression_layer")
        return None if value is None else int(value)

    def _spans(self, key: str) -> List[List[int]]:
        groups = (self.config.get("conditions", {}) or {}).get(key) or []
        spans = [sorted(int(layer) for layer in group) for group in groups]
        # Sorted so report row order does not depend on how the config listed them.
        return sorted(spans, key=lambda group: (len(group), group))

    # ------------------------------------------------------------------ depth
    def depth(self, layer: int) -> Optional[float]:
        """``layer / n_layers``, or ``None`` when the run recorded no depth."""
        if not self.n_layers:
            return None
        return int(layer) / int(self.n_layers)

    def depth_of(self, layers: Sequence[int]) -> Optional[float]:
        """Mean depth fraction of a span."""
        depths = [self.depth(layer) for layer in layers]
        if not depths or any(d is None for d in depths):
            return None
        return sum(depths) / len(depths)

    def depth_label(self, layer: int) -> str:
        """``'11 (depth 0.34)'``, or just ``'11'`` when the depth is unknown."""
        depth = self.depth(layer)
        return f"{layer}" if depth is None else f"{layer} (depth {depth:.2f})"

    # ------------------------------------------------------------------ spans
    def span_pairs(self) -> List[SpanPair]:
        """Every span the run measures both an ablation and a knockout arm over."""
        from vlmflowprobe.ablation.multilayer_experiments import _compact_layer_range

        span = self.layers
        pairs: List[SpanPair] = []
        for layers in self.nested_spans:
            tag = _compact_layer_range(layers)
            if layers == span:
                # The full span is the primary condition, under its own ids.
                pairs.append(SpanPair("nested", span_label(layers), f"joint_L{tag}",
                                      f"span_knockout_L{tag}", f"ctl_joint_L{tag}",
                                      tuple(layers)))
            else:
                pairs.append(SpanPair("nested", span_label(layers), f"nested_L{tag}",
                                      f"nested_knockout_L{tag}", f"ctl_nested_L{tag}",
                                      tuple(layers)))
        a0 = self.a0_regression_layer
        if a0 is not None:
            pairs.append(SpanPair("single", span_label([a0]), f"A0_regression_L{a0}",
                                  f"knockout_L{a0}", f"ctl_single_L{a0}", (a0,)))
        for layers in self.non_nested_spans:
            tag = _compact_layer_range(layers)
            pairs.append(SpanPair("non-nested", span_label(layers), f"nonnested_L{tag}",
                                  f"nonnested_knockout_L{tag}", f"ctl_nonnested_L{tag}",
                                  tuple(layers)))
        sensitivity = self.sensitivity_span
        if sensitivity:
            tag = _compact_layer_range(sensitivity)
            pairs.append(SpanPair("sensitivity", span_label(sensitivity),
                                  f"sensitivity_joint_L{tag}", f"sensitivity_knockout_L{tag}",
                                  f"ctl_sensitivity_L{tag}", tuple(sensitivity)))
        return pairs


def _provenance_path(experiment_dir: str) -> str:
    return os.path.join(experiment_dir, "provenance.json")


def load_run_context(
    experiment_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    n_layers: Optional[int] = None,
) -> RunContext:
    """Resolve the config and model depth behind ``experiment_dir``.

    ``config_path`` wins over the run's own provenance, which is what re-running
    an analysis with different layers is for. ``n_layers`` overrides whatever
    provenance recorded, for runs made before geometry was stamped.
    """
    from vlmflowprobe.core.config import DEFAULT_CONFIG, load_config

    if config_path:
        config = load_config(config_path, overrides=overrides).to_dict()
        context = RunContext(config=config, source=config_path)
    else:
        config, source, geometry = None, None, {}
        if experiment_dir:
            path = _provenance_path(experiment_dir)
            if os.path.exists(path):
                with open(path) as handle:
                    provenance = json.load(handle)
                config = provenance.get("resolved_config")
                geometry = provenance.get("model_geometry") or {}
                source = path
        if config is None:
            config = copy.deepcopy(DEFAULT_CONFIG)
            source = "package defaults (no provenance.json in the run directory)"
        context = RunContext(config=config, source=source)
        if geometry.get("n_layers"):
            context.n_layers = int(geometry["n_layers"])

    if n_layers:
        context.n_layers = int(n_layers)
    return context
