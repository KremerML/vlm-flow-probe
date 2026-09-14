"""Single-layer ablation against the knockout ceiling, layer by layer.

A is the margin drop from ablating a layer's top-k causal features; K is the
drop from severing that layer's Image->Question attention entirely. A/K is the
share of the model-centric ceiling the dictionary recovers at that layer, and
whether single-layer A saturates well below K is the calibration's second
precondition -- the one that says ablation is acting inside the pathway rather
than around it.

Reads only distilled artifacts (``knockout_summary.json`` and each layer's
``ablation_v2_results.summary.json``), so it runs on a login node without the
model venv.

**Where K comes from matters.** The sweep measures every layer but on all
correctly answered validation items; the multi-layer matrix measures the span's
layers on the same 256 items the ablation used. Same-sample K is the honest
comparison, so ``--multilayer_dir`` is used for the layers it covers and the
sweep fills in the rest; every row says which source it used.

    python scripts/single_layer_ak_table.py --tag llava16 --site attn_out \
        --multilayer_dir output/experiments/llava16/llava16_multilayer_clevr_lite_l10-14_attn_out_question
"""

import argparse
import json
import os

from _paths import resolve_root


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--site", default="attn_out")
    parser.add_argument("--root", default="output/experiments")
    parser.add_argument("--flow_dir", default=None, help="knockout run dir (default: <tag>_knockout_clevr_lite_iq)")
    parser.add_argument("--n_layers", type=int, default=32)
    parser.add_argument("--multilayer_dir", default=None,
                        help="matrix run dir; its knockout_L<n> conditions give K on the "
                             "ablation's own 256 samples")
    parser.add_argument("--span", type=int, nargs="*", default=None)
    parser.add_argument("--min_k", type=float, default=0.02,
                        help="K below this is too small for A/K to mean anything")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = resolve_root(args.root, args.tag)
    print("root:", root)
    flow_dir = args.flow_dir or os.path.join(root, f"{args.tag}_knockout_clevr_lite_iq")
    knockout = {
        int(row["layer"]): float(row["mean_margin_drop"])
        for row in load_json(os.path.join(flow_dir, "knockout", "knockout_summary.json"))
        if row.get("flow") == "Image->Question"
    }

    # Same-sample K, where the matrix measured it.
    paired = {}
    if args.multilayer_dir:
        for layer in range(args.n_layers):
            path = os.path.join(args.multilayer_dir, "conditions", f"knockout_L{layer}", "summary.json")
            if os.path.exists(path):
                paired[layer] = float(load_json(path)["summary"]["mean_margin_drop"])

    rows = []
    for layer in range(args.n_layers):
        path = os.path.join(
            root,
            f"{args.tag}_sae_clevr_lite_layer{layer}_{args.site}_question_causal",
            "results",
            "ablation_v2_results.summary.json",
        )
        if not os.path.exists(path) or layer not in knockout:
            continue
        payload = load_json(path)
        ablation = float(payload["binding"]["mean_margin_drop"])
        control = float(payload["random"]["mean_margin_drop"])
        ceiling = paired.get(layer, knockout[layer])
        source = "matrix" if layer in paired else "sweep"
        rows.append(
            {
                "layer": layer,
                "ablation": ablation,
                "knockout": ceiling,
                "knockout_source": source,
                "knockout_sweep": knockout[layer],
                "recovered": ablation / ceiling if ceiling > args.min_k else None,
                "control": control,
                "baseline_margin": float(payload["binding"]["baseline_margin"]),
            }
        )
    if not rows:
        raise SystemExit(f"no per-layer ablation summaries found under {root} for tag {args.tag}")

    out = args.out or os.path.join(root, f"{args.tag}_single_layer_ak.json")
    with open(out, "w") as handle:
        json.dump(rows, handle, indent=1)

    print(f"{'layer':>5} {'A':>9} {'K':>9} {'A/K':>7} {'control':>9}  K from")
    for row in rows:
        share = "" if row["recovered"] is None else f"{row['recovered']:7.3f}"
        print(f"{row['layer']:>5} {row['ablation']:>9.4f} {row['knockout']:>9.4f} "
              f"{share:>7} {row['control']:>9.4f}  {row['knockout_source']}")

    span = args.span or []
    if span:
        picked = [r for r in rows if r["layer"] in span]
        a_sum = sum(r["ablation"] for r in picked)
        k_sum = sum(r["knockout"] for r in picked)
        print(f"\nspan {span}: sum A = {a_sum:.4f}, sum K = {k_sum:.4f}, "
              f"sum A / sum K = {a_sum / k_sum:.3f}")
        print("(the additive comparison; the matrix measures the joint arms directly)")
    print("wrote", out)


if __name__ == "__main__":
    main()
