"""Per-layer dictionary-fit table for a model tag: the appendix table, as an artifact.

How well a dictionary reconstructs *this task's* activations is the first of the
three preconditions the calibration needs, and it is a measurement, not an
assumption -- the Gemma replication failed on exactly this. Every stage-01 run
writes ``reconstruction_eval.json``; this collects them into one table so the
paper's appendix and the replication record cite a file rather than a scrollback.

    python scripts/sae_fit_table.py --tag llava16 --site attn_out
    python scripts/sae_fit_table.py --tag gemma3_4b --site attn_z

Stdlib only: it runs on a login node without the model venv.
"""

import argparse
import glob
import json
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--site", default="attn_out")
    parser.add_argument("--root", default="output/experiments")
    parser.add_argument("--split", default="train", help="key inside reconstruction_eval.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pattern = os.path.join(
        args.root, f"{args.tag}_sae_clevr_lite_layer*_{args.site}_question"
    )
    rows = []
    for directory in glob.glob(pattern):
        match = re.search(r"layer(\d+)_", os.path.basename(directory))
        path = os.path.join(directory, "reconstruction_eval.json")
        if not match or not os.path.exists(path):
            continue
        with open(path) as handle:
            payload = json.load(handle)
        # Older runs wrote the metrics at the top level; newer ones key by split.
        metrics = payload.get(args.split, payload)
        rows.append(
            {
                "layer": int(match.group(1)),
                "rows": metrics.get("rows"),
                "explained_variance": metrics.get("explained_variance"),
                "normalized_mse": metrics.get("normalized_mse"),
                "mean_l0": metrics.get("mean_l0"),
                "dead_fraction": metrics.get("dead_feature_fraction"),
            }
        )
    if not rows:
        raise SystemExit(f"no reconstruction_eval.json found under {pattern}")
    rows.sort(key=lambda r: r["layer"])

    out = args.out or os.path.join(args.root, f"{args.tag}_sae_fit_table.json")
    with open(out, "w") as handle:
        json.dump(rows, handle, indent=1)

    print(f"{'layer':>5} {'EV':>9} {'mean L0':>9} {'dead':>8}")
    for row in rows:
        ev, l0, dead = row["explained_variance"], row["mean_l0"], row["dead_fraction"]
        print(f"{row['layer']:>5} {ev:>9.5f} {l0:>9.1f} {dead:>8.4f}")
    evs = [r["explained_variance"] for r in rows if r["explained_variance"] is not None]
    deads = [r["dead_fraction"] for r in rows if r["dead_fraction"] is not None]
    print(f"\n{len(rows)} layers | EV {min(evs):.5f}-{max(evs):.5f} | "
          f"dead {min(deads):.4f}-{max(deads):.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
