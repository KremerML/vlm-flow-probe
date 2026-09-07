"""Decompose margin drops into their two components, per condition and per layer.

margin = log P(true) - log P(false). A drop can come from the true option losing
probability (the answer is gone) or from the false option gaining it (the
distribution has spread). At Gemma's confidence -- true near 0 nats, false near
-33 -- the two behave very differently under ablation and knockout, so every
effect is reported as (true drop, false rise, margin drop, forced-choice and
generation accuracy). Stdlib only; reads the raw per-sample files, so it runs
where they live.

    python scripts/gemma3_decompose.py --root output/experiments --tag gemma3_4b \\
        --sweep output/experiments/gemma3_4b_knockout_clevr_lite_iq/knockout/checkpoint.jsonl \\
        --out output/experiments/gemma3_4b_metric_decomposition.json
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def decompose(
    rows,
    b_true="baseline_true_logprob",
    a_true="ablated_true_logprob",
    b_false="baseline_false_logprob",
    a_false="ablated_false_logprob",
):
    out = {
        "n": len(rows),
        "true_drop": mean(r[b_true] - r[a_true] for r in rows),
        "false_rise": mean(r[a_false] - r[b_false] for r in rows),
        "margin_drop": mean((r[b_true] - r[b_false]) - (r[a_true] - r[a_false]) for r in rows),
        "baseline_true": mean(r[b_true] for r in rows),
        "baseline_false": mean(r[b_false] for r in rows),
        "after_true": mean(r[a_true] for r in rows),
        "after_false": mean(r[a_false] for r in rows),
        "fc_acc_after": mean(1.0 if r[a_true] > r[a_false] else 0.0 for r in rows),
    }
    if "ablated_pred" in rows[0]:
        out["gen_acc_after"] = mean(1.0 if r["ablated_pred"] == r["answer"] else 0.0 for r in rows)
        out["gen_acc_before"] = mean(1.0 if r["baseline_pred"] == r["answer"] else 0.0 for r in rows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/experiments")
    parser.add_argument("--tag", default="gemma3_4b")
    parser.add_argument("--site", default="attn_z")
    parser.add_argument("--sweep", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = {"single_layer": {}, "multilayer": {}, "sweep": {}}

    for path in sorted(
        glob.glob(
            os.path.join(
                args.root,
                f"{args.tag}_sae_clevr_lite_layer*_{args.site}_question_causal",
                "results",
                "ablation_v2_results.json",
            )
        )
    ):
        layer = int(path.split("_layer")[1].split("_")[0])
        data = json.load(open(path))
        entry = {"binding": decompose(data["binding_results"])}
        if data.get("random_results"):
            entry["random_first_set"] = decompose(data["random_results"])
        report["single_layer"][layer] = entry

    for ml in sorted(glob.glob(os.path.join(args.root, f"{args.tag}_multilayer_*"))):
        conds = {}
        for path in sorted(glob.glob(os.path.join(ml, "conditions", "*", "results.json"))):
            rows = json.load(open(path))["per_sample"]
            conds[os.path.basename(os.path.dirname(path))] = decompose(rows)
        if conds:
            report["multilayer"][os.path.basename(ml)] = conds

    by_layer = defaultdict(list)
    n_items = 0
    for path in args.sweep:
        for line in open(path):
            if line.strip():
                n_items += 1
                for r in json.loads(line)["rows"]:
                    by_layer[(r["flow"], r["layer"])].append(r)
    for (flow, layer), rows in sorted(by_layer.items()):
        report["sweep"].setdefault(flow, {})[layer] = decompose(
            rows, "base_true_logprob", "new_true_logprob", "base_false_logprob", "new_false_logprob"
        )
    report["sweep_items"] = n_items

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote", args.out)
    for layer, e in sorted(report["single_layer"].items()):
        b = e["binding"]
        print(
            "L%2d ablation: margin %+7.3f = true drop %+7.3f + false rise %+7.3f | gen acc %.3f -> %.3f"
            % (
                layer,
                b["margin_drop"],
                b["true_drop"],
                b["false_rise"],
                b["gen_acc_before"],
                b["gen_acc_after"],
            )
        )


if __name__ == "__main__":
    main()
