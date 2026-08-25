"""Analyse the redesigned control conditions.

Every condition -- published and new -- is evaluated on the same 256 questions
in the same order, so per-question values pair across conditions. All inference
here is over questions, using ``paired_bootstrap_ci``, which resamples the
question indices once per iteration and applies them to every condition.

No z-scores. The z over N control sets is not estimable for this experiment
(see run_control_conditions.py) and is replaced throughout by paired contrasts
with bootstrap intervals.

    vfp-analyze-controls --control_dir output/experiments/controls_v3 \
                         --published <archived multilayer run>
"""

import argparse
import json
import os
import sys

import numpy as np

from vlmflowprobe.ablation.statistical_analysis import paired_bootstrap_ci

#: Reference run to pair against; overridable with --published.
PUBLISHED = os.environ.get("VFP_PUBLISHED_RUN", "")
LAYERS = (10, 11, 12, 13, 14)


# --------------------------------------------------------------------------- loading


def load_condition(directory, condition_id):
    """Per-sample drops and summary for one condition, from either run."""
    path = os.path.join(directory, "conditions", condition_id, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        data = json.load(handle)
    distilled = data.get("per_sample_distilled") or {}
    drops = distilled.get("margin_drops")
    if drops is None:
        rows = data.get("per_sample") or []
        drops = [
            r["baseline_margin"] - r["ablated_margin"] for r in rows
            if r.get("baseline_margin") is not None and r.get("ablated_margin") is not None
        ]
    return {
        "id": condition_id,
        "drops": np.array(drops, dtype=float),
        "summary": data.get("summary", {}),
        "sha1": distilled.get("question_ids_sha1"),
        "source": directory,
    }


def load_any(control_dir, condition_id=None, published=None):
    """Look in the control run first, then the reference run."""
    found = load_condition(control_dir, condition_id)
    if found is None and published:
        found = load_condition(published, condition_id)
    return found


# --------------------------------------------------------------------------- statistics


def paired_contrast(treatment, control, n_bootstrap=10000, seed=42):
    """Mean per-question difference (treatment - control) with a bootstrap interval."""
    if treatment is None or control is None:
        return None
    a, b = treatment["drops"], control["drops"]
    if a.size == 0 or a.size != b.size:
        return {"status": "length_mismatch", "n_treatment": int(a.size), "n_control": int(b.size)}
    if treatment["sha1"] and control["sha1"] and treatment["sha1"] != control["sha1"]:
        return {"status": "sha1_mismatch"}

    point, lo, hi = paired_bootstrap_ci(
        {"t": a, "c": b},
        statistic=lambda d: float(np.mean(d["t"]) - np.mean(d["c"])),
        n_bootstrap=n_bootstrap, seed=seed,
    )
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    return {
        "status": "ok",
        "n": int(a.size),
        "treatment_mean": float(np.mean(a)),
        "control_mean": float(np.mean(b)),
        "difference": point,
        "ci_low": lo,
        "ci_high": hi,
        "cohens_dz": float(point / sd) if sd > 0 else None,
        "pct_questions_positive": float(np.mean(diff > 0)),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def adjusted_ratio(num_t, num_c, den_t, den_c, n_bootstrap=10000, seed=42):
    """Ratio of two control-adjusted effects, with a paired interval.

    Used for the headline budget comparison: (spread - its control) divided by
    (concentrated - its control), where all four quantities are measured on the
    same questions.
    """
    if any(x is None for x in (num_t, num_c, den_t, den_c)):
        return None
    data = {"nt": num_t["drops"], "nc": num_c["drops"],
            "dt": den_t["drops"], "dc": den_c["drops"]}
    if len({v.size for v in data.values()}) != 1:
        return {"status": "length_mismatch"}

    def statistic(d):
        denominator = float(np.mean(d["dt"]) - np.mean(d["dc"]))
        if abs(denominator) < 1e-9:
            return float("nan")
        return float((np.mean(d["nt"]) - np.mean(d["nc"])) / denominator)

    point, lo, hi = paired_bootstrap_ci(data, statistic=statistic,
                                        n_bootstrap=n_bootstrap, seed=seed)
    return {"status": "ok", "ratio": point, "ci_low": lo, "ci_high": hi}


def fmt(value, width=8, places=4, signed=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-".rjust(width)
    return f"{value:+.{places}f}".rjust(width) if signed else f"{value:.{places}f}".rjust(width)


# --------------------------------------------------------------------------- sections


def section_single_layer(control_dir, published, report, out):
    """Binding against each control, per layer."""
    out.append("\n## 1. Single-layer: binding against disjoint controls\n")
    out.append("Every value is a mean over the same 256 questions. `difference` is the")
    out.append("paired per-question contrast (binding minus control) with a 95% bootstrap")
    out.append("interval that resamples question indices once and applies them to both arms.\n")
    rows = {}
    for layer in LAYERS:
        binding = load_any(control_dir, published=published, condition_id=f"bind_causal_1_200_L{layer}")
        if binding is None:
            continue
        out.append(f"\n### Layer {layer}\n")
        out.append(f"{'condition':<34}{'drop':>9}{'perturb':>10}{'difference':>11}"
                   f"{'95% interval':>21}{'d_z':>7}{'%q>0':>7}")
        out.append("-" * 99)
        s = binding["summary"]
        out.append(f"{'binding: causal ranks 1-200':<34}{fmt(s.get('mean_margin_drop'),9)}"
                   f"{fmt(s.get('mean_relative_perturbation'),10,5,False)}"
                   f"{'reference':>11}{'':>21}{'':>7}{'':>7}")
        layer_rows = {}
        for cid, label in (
            (f"ctl_passthrough_L{layer}", "pass-through (0 features)"),
            (f"ctl_band_200_400_L{layer}", "control: causal ranks 201-400"),
            (f"ctl_band_400_600_L{layer}", "control: causal ranks 401-600"),
            (f"ctl_acttop_200_L{layer}", "control: top-200 by activation"),
            (f"ctl_actmatched_200_L{layer}", "control: activation-matched"),
        ):
            control = load_any(control_dir, published=published, condition_id=cid)
            if control is None:
                continue
            contrast = paired_contrast(binding, control)
            cs = control["summary"]
            if contrast and contrast.get("status") == "ok":
                interval = f"[{contrast['ci_low']:+.4f}, {contrast['ci_high']:+.4f}]"
                out.append(
                    f"{label:<34}{fmt(cs.get('mean_margin_drop'),9)}"
                    f"{fmt(cs.get('mean_relative_perturbation'),10,5,False)}"
                    f"{fmt(contrast['difference'],11)}"
                    f"{interval:>21}"
                    f"{contrast['cohens_dz']:>7.2f}"
                    f"{100*contrast['pct_questions_positive']:>6.0f}%"
                )
                layer_rows[cid] = contrast
        rows[layer] = layer_rows
    report["single_layer"] = rows


def section_gradient(control_dir, published, report, out):
    """Does the gradient factor contribute beyond activation?"""
    out.append("\n\n## 2. Isolating the gradient term\n")
    out.append("causal_score = |gradient| x |activation|, so the causal and activation")
    out.append("rankings overlap heavily. These conditions ablate only the features on")
    out.append("which the two rankings disagree, plus the features they share.\n")
    out.append(f"{'condition':<40}{'n feat':>7}{'drop':>10}{'perturb':>10}")
    out.append("-" * 67)
    res = {}
    for layer in (11, 14):
        for cid, label in ((f"grad_shared_L{layer}", f"L{layer}: in both rankings"),
                           (f"grad_causal_only_L{layer}", f"L{layer}: causal-ranking only"),
                           (f"grad_act_only_L{layer}", f"L{layer}: activation-ranking only")):
            c = load_any(control_dir, published=published, condition_id=cid)
            if c is None:
                continue
            s = c["summary"]
            out.append(f"{label:<40}{s.get('total_features',0):>7}"
                       f"{fmt(s.get('mean_margin_drop'),10)}"
                       f"{fmt(s.get('mean_relative_perturbation'),10,5,False)}")
            res[cid] = {"drop": s.get("mean_margin_drop"),
                        "perturbation": s.get("mean_relative_perturbation"),
                        "n_features": s.get("total_features")}
        co = load_any(control_dir, published=published, condition_id=f"grad_causal_only_L{layer}")
        ao = load_any(control_dir, published=published, condition_id=f"grad_act_only_L{layer}")
        contrast = paired_contrast(co, ao)
        if contrast and contrast.get("status") == "ok":
            out.append(f"  L{layer} causal-only minus activation-only: "
                       f"{contrast['difference']:+.4f} "
                       f"[{contrast['ci_low']:+.4f}, {contrast['ci_high']:+.4f}]"
                       f"{'  (interval excludes zero)' if contrast['excludes_zero'] else ''}")
            res[f"contrast_L{layer}"] = contrast
    report["gradient_isolation"] = res


def section_dose(control_dir, published, report, out):
    """Effect against causal rank at fixed feature count."""
    out.append("\n\n## 3. Dose-response over causal rank (40 features per band)\n")
    out.append("Bands are disjoint and equal in size, so a decline cannot be a count effect.\n")
    res = {}
    for layer in (11, 14):
        bands = []
        for i in range(10):
            lo, hi = i * 40, (i + 1) * 40
            c = load_any(control_dir, published=published, condition_id=f"dose_L{layer}_r{lo}_{hi}")
            if c is None:
                continue
            s = c["summary"]
            bands.append({"lo": lo, "hi": hi, "drop": s.get("mean_margin_drop"),
                          "perturbation": s.get("mean_relative_perturbation")})
        if not bands:
            continue
        out.append(f"\n### Layer {layer}\n")
        out.append(f"{'causal ranks':>16}{'drop':>10}{'perturb':>10}{'drop/perturb':>14}")
        out.append("-" * 50)
        for b in bands:
            eff = b["drop"] / b["perturbation"] if b["perturbation"] else float("nan")
            rank_label = f"{b['lo'] + 1}-{b['hi']}"
            out.append(f"{rank_label:>16}"
                       f"{fmt(b['drop'],10)}{fmt(b['perturbation'],10,5,False)}{eff:>14.1f}")
        drops = [b["drop"] for b in bands]
        idx = list(range(len(bands)))
        rho = float(np.corrcoef(idx, drops)[0, 1]) if len(bands) > 2 else float("nan")
        out.append(f"  correlation of effect with band index: {rho:+.3f} "
                   f"(negative means the ranking orders features by effect)")
        res[f"L{layer}"] = {"bands": bands, "corr_with_rank": rho}
    report["dose_response"] = res


def section_subsets(control_dir, published, report, out):
    """The one place a set-level null is estimable."""
    out.append("\n\n## 4. Random 40-feature subsets of two deep pools\n")
    out.append("Subsets of the top-200 and of ranks 201-1000 overlap by about 20% and 5%,")
    out.append("so between-set variance here is real, unlike the matched-control sets.\n")
    res = {}
    for name, label in (("subset_top200_L11", "random 40 of causal ranks 1-200"),
                        ("subset_tail_L11", "random 40 of causal ranks 201-1000")):
        drops = []
        for i in range(12):
            c = load_any(control_dir, published=published, condition_id=f"{name}_{i:02d}")
            if c is not None:
                drops.append(c["summary"].get("mean_margin_drop"))
        if not drops:
            continue
        arr = np.array([d for d in drops if d is not None], dtype=float)
        res[name] = {"n_sets": int(arr.size), "mean": float(arr.mean()),
                     "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
                     "min": float(arr.min()), "max": float(arr.max()),
                     "drops": arr.tolist()}
        out.append(f"  {label:<38} n={arr.size:<3} mean {arr.mean():+.4f}  "
                   f"sd {arr.std(ddof=1):.4f}  range [{arr.min():+.4f}, {arr.max():+.4f}]")
    if len(res) == 2:
        a = np.array(res["subset_top200_L11"]["drops"])
        b = np.array(res["subset_tail_L11"]["drops"])
        out.append(f"\n  separation: top-200 subsets exceed tail subsets by "
                   f"{a.mean()-b.mean():+.4f} on average; "
                   f"{'no overlap' if a.min() > b.max() else 'ranges overlap'} between the two sets of runs")
        res["separation"] = float(a.mean() - b.mean())
        res["disjoint_ranges"] = bool(a.min() > b.max())
    report["subsets"] = res


def section_multi(control_dir, published, report, out):
    """Multi-layer: symmetric control adjustment and the recomputed index."""
    out.append("\n\n## 5. Multi-layer conditions with symmetric controls\n")

    # -------- budget comparison
    out.append("### 5a. The budget comparison (spreading against concentrating)\n")
    spread = load_any(control_dir, published=published, condition_id="budget_spread40x5")
    spread_ctl = load_any(control_dir, published=published, condition_id="ctl_budget_spread40x5")
    conc = load_any(control_dir, published=published, condition_id="budget_concentrated_L11_k200")
    conc_ctl = load_any(control_dir, published=published, condition_id="ctl_budget_concentrated_L11_k200")
    budget = {}
    if all(x is not None for x in (spread, spread_ctl, conc, conc_ctl)):
        sd_, sc_ = spread["summary"]["mean_margin_drop"], spread_ctl["summary"]["mean_margin_drop"]
        cd_, cc_ = conc["summary"]["mean_margin_drop"], conc_ctl["summary"]["mean_margin_drop"]
        out.append(f"{'arm':<28}{'drop':>10}{'control':>10}{'adjusted':>10}{'perturb':>10}")
        out.append("-" * 68)
        out.append(f"{'spread 40 x 5 layers':<28}{fmt(sd_,10)}{fmt(sc_,10)}{fmt(sd_-sc_,10)}"
                   f"{fmt(spread['summary'].get('mean_relative_perturbation'),10,5,False)}")
        out.append(f"{'concentrated 200 at L11':<28}{fmt(cd_,10)}{fmt(cc_,10)}{fmt(cd_-cc_,10)}"
                   f"{fmt(conc['summary'].get('mean_relative_perturbation'),10,5,False)}")
        ratio = adjusted_ratio(spread, spread_ctl, conc, conc_ctl)
        out.append(f"\n  raw ratio                : {sd_/cd_:.2f}x")
        if ratio and ratio.get("status") == "ok":
            out.append(f"  control-adjusted ratio   : {ratio['ratio']:.2f}x "
                       f"[{ratio['ci_low']:.2f}, {ratio['ci_high']:.2f}]")
        budget = {"spread": sd_, "spread_control": sc_, "concentrated": cd_,
                  "concentrated_control": cc_, "raw_ratio": sd_ / cd_ if cd_ else None,
                  "adjusted_ratio": ratio}
    report["budget"] = budget

    # -------- redundancy index
    out.append("\n### 5b. Redundancy index with control-adjusted ablation\n")
    out.append("R = ablation / knockout. The knockout arm needs no control (it is an")
    out.append("attention mask, not a feature intervention), so only the numerator moves.\n")
    spans = [
        ("{14}", 1, "nested_L14", "nested_knockout_L14", "ctl_nested_L14"),
        ("{13,14}", 2, "nested_L13-14", "nested_knockout_L13-14", "ctl_nested_L13-14"),
        ("{12,13,14}", 3, "nested_L12-14", "nested_knockout_L12-14", "ctl_nested_L12-14"),
        ("{11-14}", 4, "nested_L11-14", "nested_knockout_L11-14", "ctl_nested_L11-14"),
        ("{10-14}", 5, "joint_L10-14", "span_knockout_L10-14", "ctl_joint_L10-14"),
        ("{10,11,12}", 3, "nonnested_L10-12", "nonnested_knockout_L10-12", "ctl_nonnested_L10-12"),
        ("{10,12,14}", 3, "nonnested_L10,12,14", "nonnested_knockout_L10,12,14", "ctl_nonnested_L10,12,14"),
    ]
    out.append(f"{'span':<14}{'size':>5}{'A':>9}{'control':>9}{'A adj':>9}{'K':>9}"
               f"{'R raw':>8}{'R adj':>8}{'95% interval':>20}")
    out.append("-" * 91)
    results = []
    for label, size, a_id, k_id, c_id in spans:
        A, K, C = load_any(control_dir, published=published, condition_id=a_id), load_any(control_dir, published=published, condition_id=k_id), load_any(control_dir, published=published, condition_id=c_id)
        if A is None or K is None:
            continue
        a_mean = A["summary"]["mean_margin_drop"]
        k_mean = K["summary"]["mean_margin_drop"]
        c_mean = C["summary"]["mean_margin_drop"] if C is not None else 0.0
        if k_mean is None or k_mean <= 0:
            continue
        r_raw = a_mean / k_mean
        r_adj = (a_mean - c_mean) / k_mean
        interval = ""
        if C is not None:
            data = {"a": A["drops"], "c": C["drops"], "k": K["drops"]}
            if len({v.size for v in data.values()}) == 1:
                point, lo, hi = paired_bootstrap_ci(
                    data,
                    statistic=lambda d: float((np.mean(d["a"]) - np.mean(d["c"])) / np.mean(d["k"]))
                    if abs(np.mean(d["k"])) > 1e-9 else float("nan"),
                    n_bootstrap=10000, seed=42,
                )
                interval = f"[{100*lo:5.1f}%, {100*hi:5.1f}%]"
                r_adj = point
        out.append(f"{label:<14}{size:>5}{fmt(a_mean,9)}{fmt(c_mean,9)}{fmt(a_mean-c_mean,9)}"
                   f"{fmt(k_mean,9)}{100*r_raw:>7.1f}%{100*r_adj:>7.1f}%{interval:>20}")
        results.append({"span": label, "size": size, "ablation": a_mean, "control": c_mean,
                        "knockout": k_mean, "r_raw": r_raw, "r_adjusted": r_adj})
    if len(results) > 2:
        xs = np.array([r["size"] for r in results], dtype=float)
        for key, name in (("r_raw", "raw"), ("r_adjusted", "adjusted")):
            ys = np.array([r[key] for r in results], dtype=float)
            slope = float(np.polyfit(xs, ys, 1)[0])
            out.append(f"  slope of R against span size, {name:<9}: {slope:+.4f} per layer")
        out.append("  (a redundancy account predicts a POSITIVE slope: wider spans should")
        out.append("   recover more of the knockout ceiling)")
    report["redundancy"] = results

    # -------- other multi-layer controls
    out.append("\n### 5c. Control magnitude across the remaining multi-layer conditions\n")
    out.append(f"{'condition':<30}{'binding':>10}{'control':>10}{'control share':>15}")
    out.append("-" * 65)
    other = {}
    pairs = [(f"loo_drop{l}", f"ctl_loo_drop{l}") for l in LAYERS]
    pairs += [("downstream_ablate_L11", "ctl_downstream_ablate_L11"),
              ("downstream_ablate_L14", "ctl_downstream_ablate_L14")]
    pairs += [(f"budget_concentrated_L11_k{k}", f"ctl_budget_concentrated_L11_k{k}")
              for k in (40, 100, 200, 400, 800)]
    for b_id, c_id in pairs:
        B, C = load_any(control_dir, published=published, condition_id=b_id), load_any(control_dir, published=published, condition_id=c_id)
        if B is None or C is None:
            continue
        bd = B["summary"]["mean_margin_drop"]
        cd = C["summary"]["mean_margin_drop"]
        share = (cd / bd) if bd else float("nan")
        out.append(f"{b_id:<30}{fmt(bd,10)}{fmt(cd,10)}{100*share:>14.1f}%")
        other[b_id] = {"binding": bd, "control": cd, "control_share": share}
    report["other_multilayer"] = other


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control_dir", required=True)
    parser.add_argument("--published", default=PUBLISHED,
                        help="reference run to pair against for conditions not re-run")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = {}
    out = ["# Redesigned random controls: results",
           "",
           "All conditions evaluated on the same 256 questions in the same order.",
           "Intervals are 95% paired bootstrap over question indices, 10000 resamples.",
           "Controls are disjoint from their binding sets by construction, so no",
           "candidate pool is required and no z-score over control sets is reported."]

    section_single_layer(args.control_dir, args.published, report, out)
    section_gradient(args.control_dir, args.published, report, out)
    section_dose(args.control_dir, args.published, report, out)
    section_subsets(args.control_dir, args.published, report, out)
    section_multi(args.control_dir, args.published, report, out)

    text = "\n".join(out)
    print(text)
    analysis_dir = os.path.join(args.control_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    with open(os.path.join(analysis_dir, "control_summary.md"), "w") as handle:
        handle.write(text + "\n")
    with open(os.path.join(analysis_dir, "control_summary.json"), "w") as handle:
        json.dump(report, handle, indent=1, default=float)
    print(f"\nwrote {analysis_dir}/control_summary.{{md,json}}")


if __name__ == "__main__":
    main()
