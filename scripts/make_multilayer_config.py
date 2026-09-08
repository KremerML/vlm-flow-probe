"""Write a multi-layer experiment config for a model tag and a layer span.

The LLaVA multilayer config was written by hand for layers 10-14. The Gemma
span is chosen from its own knockout sweep, so the config is generated from
the span rather than edited: nested spans grown downward from the top layer,
two non-nested spans of size 3, the budget arm concentrated at the strongest
layer, downstream knockout anchored at the strongest and the top layer, and a
sensitivity span dropping one layer.

    python scripts/make_multilayer_config.py --tag gemma3_4b --site attn_z \
        --ablation_fragment ../../fragments/ablation_gemma_delta.yaml \
        --span 10 11 12 13 14 --concentrated 11 --drop 13 --n_layers 34
"""

import argparse
import os

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--site", default="attn_z")
    parser.add_argument("--span", type=int, nargs="+", required=True)
    parser.add_argument("--concentrated", type=int, required=True, help="layer for the budget curve")
    parser.add_argument("--drop", type=int, default=None, help="layer left out of the sensitivity span")
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument(
        "--anchors",
        type=int,
        nargs="*",
        default=None,
        help="downstream-knockout anchor layers (default: concentrated + top of span)",
    )
    parser.add_argument("--spread_per_layer", type=int, default=40)
    parser.add_argument(
        "--concentrated_k",
        type=int,
        nargs="*",
        default=[40, 100, 200, 400, 800],
        help="budget curve; the equal-budget point spread_per_layer*len(span) is added",
    )
    parser.add_argument("--model_fragment", default=None)
    parser.add_argument("--sae_fragment", default=None)
    parser.add_argument(
        "--ablation_fragment",
        default=None,
        help="extra ablation fragment included after ablation_default (e.g. "
        "../../fragments/ablation_gemma_delta.yaml for a delta-mode run). Omit to "
        "keep the published replace-mode protocol.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    span = sorted(args.span)
    tag, site = args.tag, args.site
    exp = lambda layer: f"output/experiments/{tag}_sae_clevr_lite_layer{layer}_{site}_question"  # noqa: E731
    nested = [span[i:] for i in range(len(span) - 1, -1, -1)]
    non_nested = []
    if len(span) >= 5:
        non_nested = [[span[0], span[2], span[4]], span[:3]]
    elif len(span) >= 3:
        non_nested = [span[:3]]
    sensitivity = [layer for layer in span if layer != args.drop] if args.drop is not None else []
    anchors = sorted(set(args.anchors)) if args.anchors else sorted({args.concentrated, span[-1]})
    concentrated_k = sorted(set(args.concentrated_k) | {args.spread_per_layer * len(span)})

    config = {
        "include": [
            args.model_fragment or f"../../fragments/model_{tag}_it.yaml",
            "../../fragments/dataset_clevr_lite.yaml",
            args.sae_fragment or "../../fragments/sae_gemma_scope2_attn.yaml",
            "../../fragments/ablation_default.yaml",
            *([args.ablation_fragment] if args.ablation_fragment else []),
        ],
        "model": {"target_layer": args.concentrated},
        "dataset": {"split": "val"},
        "knockout": {"flows": ["Image->Question"], "filter_correct": False},
        "multilayer": {
            "layers": span,
            # Layer 0 is excluded from the span choice by the paper's rule: it is
            # where visual information enters, not part of the mid-stack band.
            "excluded_layers": [0],
            "encode_mode": "live",
            "encode_positions_only": True,
            "features_per_layer": 200,
            "sae_dtype": "float32",
            "sae_paths": {layer: f"{exp(layer)}/sae_checkpoint.pt" for layer in span},
            "catalog_paths": {layer: f"{exp(layer)}_causal/causal_feature_catalog.json" for layer in span},
            "stats_paths": {layer: f"{exp(layer)}_causal/causal_feature_stats.json" for layer in span},
        },
        "conditions": {
            "gate": True,
            "primary": True,
            # No published number to regress against for a new model.
            "a0_regression_layer": None,
            "nested": nested,
            "non_nested": non_nested,
            "leave_one_out": True,
            "span_knockout": True,
            "budget_matched": {
                "spread_per_layer": args.spread_per_layer,
                "concentrated_layer": args.concentrated,
                "concentrated_k": concentrated_k,
            },
            "downstream_knockout": {"anchor_layers": anchors, "downstream_to": args.n_layers - 1},
            "sensitivity_span": sensitivity,
        },
        "experiment": {
            "output_base": "output/experiments",
            "name": f"{tag}_multilayer_clevr_lite_l{span[0]}-{span[-1]}_{site}_question",
            "use_timestamp": False,
        },
    }
    out = args.out or f"configs/experiments/{tag}/multilayer_l{span[0]}-{span[-1]}_{site}_question.yaml"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(
            f"# Generated by scripts/make_multilayer_config.py for span {span} "
            f"(concentrated layer {args.concentrated}).\n"
        )
        yaml.safe_dump(config, handle, sort_keys=False)
    print(out)


if __name__ == "__main__":
    main()
