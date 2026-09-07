"""Per-layer knockout means from a sweep's checkpoint.jsonl, complete or not.

The sweep writes one JSONL line per finished sample, so a running job can be
read out at any time. Prints the per-layer mean margin drop and Cohen's d per
flow, the top layers, and the sample count so a preliminary reading is never
mistaken for the final one.

    python scripts/knockout_partial_summary.py \
        output/experiments/gemma3_4b_knockout_clevr_lite_iq/knockout/checkpoint.jsonl
"""

import argparse
import json
import math
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    drops = defaultdict(list)
    n_samples = 0
    base_margins = []
    for path in args.checkpoints:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                n_samples += 1
                for row in entry["rows"]:
                    drops[(row["flow"], row["layer"])].append(row["margin_drop"])
                if entry["rows"]:
                    base_margins.append(entry["rows"][0]["base_margin"])

    summary = []
    for (flow, layer), values in sorted(drops.items()):
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        sd = math.sqrt(var)
        summary.append(
            {
                "flow": flow,
                "layer": layer,
                "samples": len(values),
                "mean_margin_drop": mean,
                "effect_size": mean / sd if sd else 0.0,
                "se": sd / math.sqrt(len(values)),
            }
        )

    mean_base = sum(base_margins) / max(1, len(base_margins))
    print(f"{n_samples} samples; mean baseline margin {mean_base:.3f}")
    for flow in sorted({s["flow"] for s in summary}):
        rows = [s for s in summary if s["flow"] == flow]
        print(f"\n{flow} (n = {rows[0]['samples']})")
        print("  layer   drop      se     d")
        for s in rows:
            print(f"  {s['layer']:5d} {s['mean_margin_drop']:+8.4f} {s['se']:7.4f} {s['effect_size']:+6.3f}")
        top = sorted(rows, key=lambda s: -s["mean_margin_drop"])[: args.top]
        print("  top:", [(s["layer"], round(s["mean_margin_drop"], 3)) for s in top])
        print("  negative:", [s["layer"] for s in rows if s["mean_margin_drop"] < 0])
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"n_samples": n_samples, "summary": summary}, handle, indent=1)


if __name__ == "__main__":
    main()
