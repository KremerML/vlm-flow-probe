"""The equivalence gate: does the new harness reproduce the archived numbers?

Four stages, each with explicit pass criteria (see gate/README.md):

  0  geometry parity   — prompt strings, 576 image tokens, question spans
                         exactly equal the archived sample_cache positions
  1  baseline margins  — 256 per-sample baselines vs the archived cache
  2  knockout parity   — per-layer Image->Question / Image->Last sweep vs the
                         archived summaries (n=10 config; n=7084 as secondary)
  3  A0 regression     — archived layer-11 SAE + catalog + archived control
                         feature sets; mean drop 0.21307 within tolerance

Writes output/gate/<timestamp>/gate_report.json and exits nonzero on failure.
Stage-0 failures are never acceptable; other stages may only be accepted via
an entry in docs/deviation_ledger.md.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REFERENCES = Path(__file__).resolve().parents[3] / "gate" / "references"

EXPECTED_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: <image>\n{question} \nAnswer the question using a single word or phrase. ASSISTANT:"
)

N_SAMPLES = 256


# ---------------------------------------------------------------------- helpers
def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def _spearman(xs, ys):
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    return _pearson(ranks(xs), ranks(ys))


class _Check:
    """One named criterion; collects into the stage report."""

    def __init__(self, stage_report):
        self.report = stage_report

    def add(self, name, passed, detail):
        self.report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        marker = "PASS" if passed else "FAIL"
        print(f"    [{marker}] {name}: {detail}")


def _stage_report(report, name):
    stage = {"stage": name, "checks": []}
    report["stages"].append(stage)
    print(f"\n== Stage {name} ==")
    return stage


def _load_reference(name):
    return json.loads((REFERENCES / name).read_text())


# ---------------------------------------------------------------------- setup
def _load_adapter():
    from vlmflowprobe.adapters.registry import create_adapter

    adapter = create_adapter({"adapter": "hf-llava", "name": "llava-hf/llava-1.5-7b-hf"})
    adapter.load()
    return adapter


def _load_dataset(archive_root):
    from vlmflowprobe.data.datasets import CLEVRLiteVQADataset

    data_dir = Path(archive_root) / "datasets" / "clevr_lite"
    if not data_dir.exists():
        sys.exit(f"CLEVR-Lite not found at {data_dir}; pass --archive-root")
    return CLEVRLiteVQADataset(data_dir=str(data_dir), split="val")


def _cache_records():
    return _load_reference("sample_cache.json")["records"]


# ---------------------------------------------------------------------- stage 0
def stage0(adapter, dataset, report):
    stage = _stage_report(report, "0-geometry")
    check = _Check(stage)
    records = _cache_records()

    qid_mismatches, span_mismatches, prompt_mismatches, expansion_bad = [], [], [], []
    for idx, rec in enumerate(records):
        line = dataset.questions[idx]
        if str(line["q_id"]) != str(rec["question_id"]):
            qid_mismatches.append((idx, line["q_id"], rec["question_id"]))
            continue
        detail = dataset.dataset_dict[line["q_id"]]
        batch = adapter.build_inputs(detail["question"], dataset.load_image(line))
        if batch.prompt != EXPECTED_PROMPT.format(question=detail["question"]):
            prompt_mismatches.append(line["q_id"])
        if adapter.n_image_tokens(batch) != 576:
            expansion_bad.append((line["q_id"], adapter.n_image_tokens(batch)))
        span = adapter.question_token_span(batch)
        if span != rec["positions"]:
            span_mismatches.append((line["q_id"], span[:3], rec["positions"][:3]))

    check.add("sample order matches cache", not qid_mismatches,
              f"{len(records) - len(qid_mismatches)}/{len(records)} in order" +
              (f"; first mismatch {qid_mismatches[0]}" if qid_mismatches else ""))
    check.add("prompt strings byte-exact", not prompt_mismatches,
              f"{len(records) - len(prompt_mismatches)}/{len(records)}")
    check.add("576 image tokens everywhere", not expansion_bad,
              f"{len(records) - len(expansion_bad)}/{len(records)}" +
              (f"; first bad {expansion_bad[0]}" if expansion_bad else ""))
    check.add("question spans exactly equal archived positions", not span_mismatches,
              f"{len(records) - len(span_mismatches)}/{len(records)}" +
              (f"; first mismatch {span_mismatches[0]}" if span_mismatches else ""))
    return stage


# ---------------------------------------------------------------------- stage 1
def stage1(adapter, dataset, report, shared):
    from vlmflowprobe.ablation.feature_ablator import FeatureAblator
    from vlmflowprobe.ablation.sample_cache import build_sample_cache

    stage = _stage_report(report, "1-baselines")
    check = _Check(stage)
    records = _cache_records()

    ablator = FeatureAblator(adapter, sae=None, layer_idx=11, activation_site="attn_out")
    t0 = time.time()
    new_records = build_sample_cache(
        ablator, dataset, position_type="question",
        max_samples=len(records), show_progress=True,
    )
    stage["runtime_seconds"] = round(time.time() - t0, 1)
    shared["sample_records"] = new_records
    shared["ablator"] = ablator

    old_m = [r["baseline"]["baseline_margin"] for r in records]
    new_m = [r.baseline["baseline_margin"] for r in new_records]
    deltas = sorted(abs(a - b) for a, b in zip(old_m, new_m))
    r = _pearson(old_m, new_m)
    mean_d = sum(deltas) / len(deltas)
    p99 = deltas[int(0.99 * (len(deltas) - 1))]

    preds_old = [r["baseline"]["baseline_pred"] for r in records]
    preds_new = [r.baseline["baseline_pred"] for r in new_records]
    agree = sum(a == b for a, b in zip(preds_old, preds_new))
    bad_flips = [
        (records[i]["question_id"], old_m[i])
        for i in range(len(records))
        if preds_old[i] != preds_new[i] and abs(old_m[i]) >= 0.1
    ]
    answers = [dataset.dataset_dict[r.question_id]["answer"].strip().lower() for r in new_records]
    acc = sum(p == a for p, a in zip(preds_new, answers)) / len(new_records)

    check.add("margin correlation r >= 0.999", r >= 0.999, f"r = {r:.6f}")
    check.add("mean |dmargin| <= 0.05", mean_d <= 0.05, f"mean = {mean_d:.4f}")
    check.add("p99 |dmargin| <= 0.25", p99 <= 0.25, f"p99 = {p99:.4f}, max = {deltas[-1]:.4f}")
    check.add("pred agreement >= 254/256, flips only near zero margin",
              agree >= 254 and not bad_flips,
              f"agree = {agree}/{len(records)}, confident flips = {bad_flips[:3]}")
    check.add("accuracy within 1/256 of 0.6484", abs(acc - 0.6484375) <= 1 / 256 + 1e-9,
              f"accuracy = {acc:.6f}")

    # self-consistency: our own jitter on 8 samples
    rerun = build_sample_cache(ablator, dataset, position_type="question", max_samples=8)
    jitter = max(
        abs(a.baseline["baseline_margin"] - b.baseline["baseline_margin"])
        for a, b in zip(new_records[:8], rerun)
    )
    check.add("self-consistency max |dmargin| < 1e-3", jitter < 1e-3, f"jitter = {jitter:.2e}")
    stage["stats"] = {"pearson_r": r, "mean_abs_delta": mean_d, "p99_abs_delta": p99,
                      "max_abs_delta": deltas[-1], "pred_agreement": agree,
                      "accuracy": acc, "self_jitter": jitter}
    return stage


# ---------------------------------------------------------------------- stage 2
def stage2(adapter, dataset, report):
    from vlmflowprobe.knockout.runner import run_knockout_sweep

    stage = _stage_report(report, "2-knockout")
    check = _Check(stage)
    ref10 = _load_reference("knockout_summary_n10.json")
    ref_full = _load_reference("knockout_summary_n7084.json")

    t0 = time.time()
    _, summaries = run_knockout_sweep(
        adapter, dataset, flows=["Image->Question", "Image->Last"],
        window=1, max_samples=10, filter_correct=True,
    )
    stage["runtime_seconds"] = round(time.time() - t0, 1)
    stage["summaries"] = summaries

    for flow in ("Image->Question", "Image->Last"):
        ref_rows = {r["layer"]: r for r in ref10 if r["flow"] == flow}
        new_rows = {r["layer"]: r for r in summaries if r["flow"] == flow}
        layers = sorted(set(ref_rows) & set(new_rows))
        ref_drops = [ref_rows[l]["mean_margin_drop"] for l in layers]
        new_drops = [new_rows[l]["mean_margin_drop"] for l in layers]

        rho = _spearman(ref_drops, new_drops)
        top3_ref = {l for l, _ in sorted(ref_rows.items(), key=lambda kv: -kv[1]["mean_margin_drop"])[:3]}
        top3_new = {l for l, _ in sorted(new_rows.items(), key=lambda kv: -kv[1]["mean_margin_drop"])[:3]}
        top5 = [l for l, _ in sorted(ref_rows.items(), key=lambda kv: -kv[1]["mean_margin_drop"])[:5]]
        drift = {
            l: (ref_rows[l]["mean_margin_drop"], new_rows[l]["mean_margin_drop"])
            for l in top5
            if abs(ref_rows[l]["mean_margin_drop"] - new_rows[l]["mean_margin_drop"])
            > max(0.05, 0.15 * abs(ref_rows[l]["mean_margin_drop"]))
        }
        check.add(f"{flow}: Spearman >= 0.95 vs n=10 reference", rho >= 0.95, f"rho = {rho:.4f}")
        check.add(f"{flow}: top-3 layer set identical", top3_ref == top3_new,
                  f"ref {sorted(top3_ref)} vs new {sorted(top3_new)}")
        check.add(f"{flow}: top-5 drops within max(0.05, 15%)", not drift, f"drift = {drift}")

        # secondary, report-only: the layer profile against the full n=7084 sweep
        full_rows = {r["layer"]: r for r in ref_full if r["flow"] == flow}
        common = sorted(set(full_rows) & set(new_rows))
        rho_full = _spearman(
            [full_rows[l]["mean_margin_drop"] for l in common],
            [new_rows[l]["mean_margin_drop"] for l in common],
        )
        stage.setdefault("secondary", {})[flow] = {"spearman_vs_n7084": rho_full}
        print(f"    [info] {flow}: Spearman vs full n=7084 profile = {rho_full:.4f}")
    return stage


# ---------------------------------------------------------------------- stage 3
def stage3(adapter, dataset, report, shared, archive_root, n_controls):
    import torch

    from vlmflowprobe.ablation.multilayer_ablator import MultiLayerFeatureAblator
    from vlmflowprobe.ablation.sample_cache import baseline_cache, positions_cache
    from vlmflowprobe.utils.runtime import load_sae

    stage = _stage_report(report, "3-A0-regression")
    check = _Check(stage)
    ref = _load_reference("A0_regression_L11.summary.json")

    archive = Path(archive_root)
    ckpt = archive / "output/sae_experiments/sae_clevr_lite_layer11_attn_out_question/sae_checkpoint.pt"
    catalog_path = archive / "output/sae_experiments/sae_clevr_lite_layer11_attn_out_question_causal/causal_feature_catalog.json"
    if not ckpt.exists():
        check.add("archived SAE checkpoint present", False, str(ckpt))
        return stage

    class _Cfg:
        def get(self, section, default=None):
            return {"sae": {"l1_coeff": 0.0005}, "training": {"dtype": "float32"}}.get(section, default)

    sae = load_sae(_Cfg(), adapter, str(ckpt))
    catalog = json.loads(catalog_path.read_text())
    features = catalog.get("features", catalog)
    binding = [int(f) for f in list(features)[:200]]

    ablator = MultiLayerFeatureAblator(
        adapter, {11: sae}, activation_site="attn_out", encode_positions_only=True
    )
    records = shared.get("sample_records")
    if records is None:
        from vlmflowprobe.ablation.sample_cache import build_sample_cache

        records = build_sample_cache(
            ablator, dataset, position_type="question", max_samples=N_SAMPLES, show_progress=True
        )
        shared["sample_records"] = records

    def run_condition(feature_map, desc):
        t0 = time.time()
        rows = ablator.batch_ablation_experiment(
            dataset,
            feature_indices=feature_map,
            position_type="question",
            mode="replace",
            max_samples=N_SAMPLES,
            baseline_cache=baseline_cache(records),
            positions_cache=positions_cache(records),
            strict_cache=True,
            show_progress=True,
            progress_desc=desc,
        )
        summary = ablator.compute_ablation_effect(rows)
        summary["runtime_seconds"] = round(time.time() - t0, 1)
        return rows, summary

    rows, summary = run_condition({11: binding}, "A0 binding")
    stage["binding_summary"] = summary

    ref_drop = ref["summary"]["mean_margin_drop"]
    new_drop = summary["mean_margin_drop"]
    check.add("mean_margin_drop within 0.03 of archived",
              abs(new_drop - ref_drop) <= 0.03,
              f"archived {ref_drop:.5f} vs new {new_drop:.5f} (d = {new_drop - ref_drop:+.5f})")

    ref_persample = ref["per_sample_distilled"]["margin_drops"]
    new_persample = [
        r["baseline_margin"] - r["ablated_margin"]
        for r in rows
        if r["baseline_margin"] is not None and r["ablated_margin"] is not None
    ]
    if len(new_persample) == len(ref_persample):
        r_ps = _pearson(ref_persample, new_persample)
        check.add("per-sample drop correlation r >= 0.98", r_ps >= 0.98, f"r = {r_ps:.5f}")
        stage["per_sample_r"] = r_ps
    else:
        check.add("per-sample drop correlation r >= 0.98", False,
                  f"length mismatch {len(new_persample)} vs {len(ref_persample)}")

    # archived control feature sets, rerun verbatim
    control_sets = ref.get("control_feature_sets", [])[:n_controls]
    control_drops = []
    for i, cset in enumerate(control_sets):
        feature_map = {int(l): [int(f) for f in fs] for l, fs in cset.items()}
        _, csummary = run_condition(feature_map, f"control {i + 1}/{len(control_sets)}")
        control_drops.append(csummary["mean_margin_drop"])
    if control_drops:
        mean_control = sum(control_drops) / len(control_drops)
        ref_controls = [c["mean_margin_drop"] for c in ref.get("control_summaries", [])][:n_controls]
        check.add("controls: mean drop < 50% of binding drop",
                  abs(mean_control) < 0.5 * abs(new_drop),
                  f"mean control {mean_control:.5f} vs binding {new_drop:.5f}")
        r_ctl = _pearson(ref_controls, control_drops) if len(ref_controls) == len(control_drops) and len(control_drops) > 2 else None
        stage["controls"] = {"new": control_drops, "archived": ref_controls, "pearson_r": r_ctl}
        if r_ctl is not None:
            print(f"    [info] control-set drop correlation vs archive: r = {r_ctl:.4f}")
    return stage


# ---------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", action="append", type=int, choices=[0, 1, 2, 3],
                        help="stage(s) to run; repeatable")
    parser.add_argument("--all", action="store_true", help="run every stage")
    parser.add_argument("--archive-root", default=os.path.expanduser(
        "~/Documents/Github/cross-modal-information-flow-in-MLLM"))
    parser.add_argument("--out", default="output/gate")
    parser.add_argument("--controls", type=int, default=15,
                        help="how many archived control sets to rerun in stage 3")
    args = parser.parse_args()

    stages = sorted(set(args.stage or ([] if not args.all else [0, 1, 2, 3])))
    if not stages:
        parser.error("pass --stage N (repeatable) or --all")

    from vlmflowprobe.utils.provenance import provenance_dict

    out_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"stages": [], "requested": stages, "provenance": provenance_dict(argv=sys.argv)}

    print("Loading adapter + dataset ...")
    adapter = _load_adapter()
    dataset = _load_dataset(args.archive_root)
    shared = {}

    if 0 in stages:
        stage0(adapter, dataset, report)
    if 1 in stages:
        stage1(adapter, dataset, report, shared)
    if 2 in stages:
        stage2(adapter, dataset, report)
    if 3 in stages:
        stage3(adapter, dataset, report, shared, args.archive_root, args.controls)

    all_checks = [c for s in report["stages"] for c in s["checks"]]
    passed = all(c["passed"] for c in all_checks)
    report["passed"] = passed
    report_path = out_dir / "gate_report.json"
    report_path.write_text(json.dumps(report, indent=1))
    print(f"\n{'GATE PASSED' if passed else 'GATE FAILED'} "
          f"({sum(c['passed'] for c in all_checks)}/{len(all_checks)} checks) -> {report_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
