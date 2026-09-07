"""Paper figures for the Gemma 3 replication, in the archive notebook's style.

Reads only committed artifacts (knockout summaries, condition summaries, the
multilayer analysis JSON) and writes PNG + PDF under --figdir:

  gemma3_fig1_knockout_landscape   margin drop and Cohen's d per layer, both flows
  gemma3_fig6_budget_distribution  spread vs concentrated budget, effect and perturbation
  gemma3_fig8a_ablation_vs_knockout  A against K per span
  gemma3_fig8b_recovered_share     R = A/K per span with paired bootstrap intervals
  gemma3_fig_single_layer          per-layer A (single-layer ablation) against K, all layers run

    python scripts/gemma3_figures.py --root output/experiments --figdir paper/gemma3_figures
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, CRIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, SECOND, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": BASELINE,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def tidy(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(figdir, name))


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- fig 1


def knockout_rows(root, tag):
    rows = []
    for path in glob.glob(
        os.path.join(root, f"{tag}_knockout_clevr_lite*", "knockout", "knockout_summary.json")
    ):
        rows.extend(load_json(path))
    return rows


def fig1(rows, span, figdir, tag, n_layers):
    question = sorted([r for r in rows if r["flow"] == "Image->Question"], key=lambda r: r["layer"])
    last = sorted([r for r in rows if r["flow"] == "Image->Last"], key=lambda r: r["layer"])
    if not question:
        print("fig1: no Image->Question rows yet")
        return
    layers = [r["layer"] for r in question]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True, gridspec_kw={"hspace": 0.18})
    for ax in (ax1, ax2):
        if span:
            ax.axvspan(min(span) - 0.5, max(span) + 0.5, color=BLUE, alpha=0.07, lw=0, zorder=0)
        ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)
        tidy(ax)
    ax1.plot(
        layers,
        [r["mean_margin_drop"] for r in question],
        color=BLUE,
        lw=2,
        marker="o",
        ms=4.5,
        label="Image->Question",
        zorder=3,
    )
    ax2.plot(
        layers,
        [r["effect_size"] for r in question],
        color=BLUE,
        lw=2,
        marker="o",
        ms=4.5,
        label="Image->Question",
        zorder=3,
    )
    if last:
        ax1.plot(
            [r["layer"] for r in last],
            [r["mean_margin_drop"] for r in last],
            color=ORANGE,
            lw=2,
            marker="o",
            ms=4.5,
            label="Image->Last",
            zorder=3,
        )
        ax2.plot(
            [r["layer"] for r in last],
            [r["effect_size"] for r in last],
            color=ORANGE,
            lw=2,
            marker="o",
            ms=4.5,
            label="Image->Last",
            zorder=3,
        )
    for r in question:
        if r["mean_margin_drop"] < 0:
            ax1.plot(
                r["layer"], r["mean_margin_drop"], marker="o", ms=11, mfc="none", mec=CRIT, mew=1.8, zorder=4
            )
    if span:
        ax1.text(
            float(np.mean(span)),
            ax1.get_ylim()[1] * 0.92,
            f"layers {min(span)}-{max(span)}",
            color=BLUE,
            fontsize=9,
            ha="center",
            fontweight="bold",
        )
    ax1.set_ylabel("mean margin drop")
    ax1.legend(loc="upper right", ncol=2)
    ax2.set_ylabel("effect size (Cohen's d)")
    ax2.set_xlabel("layer")
    ax2.set_xticks(range(0, n_layers, 2))
    ax2.set_xlim(-1, n_layers)
    ax2.legend(loc="lower left", ncol=2)
    save(fig, figdir, f"{tag}_fig1_knockout_landscape")


# --------------------------------------------------------------------------- multilayer figures


def cond(ml, name):
    return load_json(os.path.join(ml, "conditions", name, "summary.json"))


def budget_ks(ml, concentrated):
    """The concentrated-curve budgets the run actually has conditions for."""
    found = []
    for path in glob.glob(os.path.join(ml, "conditions", f"budget_concentrated_L{concentrated}_k*")):
        found.append(int(os.path.basename(path).rsplit("_k", 1)[1]))
    return tuple(sorted(found))


def fig6(ml, figdir, tag, concentrated, ks=None, spread_name=None):
    ks = ks or budget_ks(ml, concentrated)
    spread_name = spread_name or next(
        os.path.basename(p) for p in glob.glob(os.path.join(ml, "conditions", "budget_spread*"))
    )
    conc = [cond(ml, f"budget_concentrated_L{concentrated}_k{k}")["summary"] for k in ks]
    spread = cond(ml, spread_name)["summary"]
    conc_drop = [c["mean_margin_drop"] for c in conc]
    conc_pert = [c["mean_relative_perturbation"] for c in conc]
    s_drop, s_pert, s_k = (
        spread["mean_margin_drop"],
        spread["mean_relative_perturbation"],
        spread["total_features"],
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.24})
    ax1.plot(
        ks,
        conc_drop,
        color=ORANGE,
        lw=2,
        marker="o",
        ms=7,
        zorder=3,
        label=f"concentrated at layer {concentrated}",
    )
    ax1.plot(
        [s_k],
        [s_drop],
        color=BLUE,
        marker="D",
        ms=11,
        ls="none",
        zorder=4,
        mec=SURFACE,
        mew=2,
        label=f"spread: {spread['features_per_layer'][next(iter(spread['features_per_layer']))]} features x "
        f"{len(spread['layers'])} layers",
    )
    equal = conc_drop[list(ks).index(s_k)] if s_k in ks else None
    if equal:
        ax1.annotate(
            "", xy=(s_k, s_drop), xytext=(s_k, equal), arrowprops=dict(arrowstyle="<->", color=SECOND, lw=1.3)
        )
        ax1.text(
            s_k * 1.08,
            (s_drop + equal) / 2,
            f"{s_drop / equal:.2f}x\nat equal budget",
            color=SECOND,
            fontsize=9.5,
            fontweight="bold",
            va="center",
        )
    ax1.set_xscale("log")
    ax1.set_xticks(ks)
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlim(min(ks) * 0.8, max(ks) * 1.45)
    ax1.set_ylim(0, max(conc_drop + [s_drop]) * 1.25)
    ax1.set_xlabel("total ablated features (log scale)")
    ax1.set_ylabel("margin drop")
    ax1.set_title("a. Effect against feature budget", loc="left")
    ax1.legend(loc="upper left")
    tidy(ax1)

    ax2.plot(
        conc_pert,
        conc_drop,
        color=ORANGE,
        lw=2,
        marker="o",
        ms=7,
        zorder=3,
        label=f"concentrated at layer {concentrated}",
    )
    for k, p, d in zip(ks, conc_pert, conc_drop):
        ax2.text(p, d - 0.02 * ax1.get_ylim()[1], f"k={k}", color=ORANGE, fontsize=8.5, ha="center", va="top")
    ax2.plot(
        [s_pert],
        [s_drop],
        color=BLUE,
        marker="D",
        ms=11,
        ls="none",
        zorder=4,
        mec=SURFACE,
        mew=2,
        label="spread",
    )
    ax2.set_ylim(0, ax1.get_ylim()[1])
    ax2.set_xlabel("mean relative perturbation at the site")
    ax2.set_ylabel("margin drop")
    ax2.set_title("b. Effect against perturbation size", loc="left")
    ax2.legend(loc="upper left")
    tidy(ax2)
    save(fig, figdir, f"{tag}_fig6_budget_distribution")


def fig8(ml, figdir, tag):
    analysis = load_json(os.path.join(ml, "analysis", "multilayer_summary.json"))
    trend = analysis["redundancy_trend"]
    rows = [r for r in analysis["redundancy_by_span"] if r.get("status") == "ok"]
    nested = sorted([r for r in rows if r["kind"] == "nested"], key=lambda r: r["span_size"])
    other = [r for r in rows if r["kind"] != "nested"]
    if trend.get("status") != "ok" or not nested:
        print("fig8: analysis incomplete", trend.get("status"))
        return
    rc = {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
    }
    lim = max([r["knockout_mean"] for r in rows] + [r["ablation_mean"] for r in rows]) * 1.12
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(3.9, 3.7))
        ax.plot([0, lim], [0, lim], color=BASELINE, lw=1, ls=(0, (4, 3)), zorder=1)
        pooled = trend["pooled_ratio"]
        x_end = min(lim, lim / pooled) if pooled > 0 else lim
        ax.plot([0, x_end], [0, x_end * pooled], color=BLUE, lw=1.3, zorder=2, clip_on=True)
        ax.text(
            x_end * 0.97,
            min(lim * 0.97, x_end * pooled * 0.97),
            "pooled %.1f%%" % (100 * pooled),
            color=BLUE,
            fontsize=8.5,
            fontweight="bold",
            ha="right",
            va="top",
        )
        ax.plot(
            [r["knockout_mean"] for r in nested],
            [r["ablation_mean"] for r in nested],
            color=BLUE,
            lw=0.9,
            alpha=0.4,
            zorder=3,
        )
        for r in nested:
            ax.plot(
                r["knockout_mean"],
                r["ablation_mean"],
                "o",
                ms=4 + 1.4 * r["span_size"],
                color=BLUE,
                mec=SURFACE,
                mew=0.7,
                zorder=5,
                label="nested spans" if r is nested[-1] else None,
            )
        for i, r in enumerate(other):
            ax.plot(
                r["knockout_mean"],
                r["ablation_mean"],
                "s",
                ms=5.5,
                mfc=SURFACE,
                mec=ORANGE,
                mew=1.4,
                zorder=5,
                label="other spans" if i == 0 else None,
            )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("knockout drop $K$")
        ax.set_ylabel("ablation drop $A$")
        ax.legend(loc="lower right", handletextpad=0.3, borderpad=0.15, labelspacing=0.3)
        tidy(ax)
        save(fig, figdir, f"{tag}_fig8a_ablation_vs_knockout")

        pooled = 100 * trend["pooled_ratio"]
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        ax.axhspan(
            100 * trend["pooled_ci_low"],
            100 * trend["pooled_ci_high"],
            color=BLUE,
            alpha=0.10,
            lw=0,
            zorder=1,
        )
        ax.axhline(pooled, color=BLUE, lw=1, ls="--", zorder=2)
        sizes = np.array([r["span_size"] for r in nested], float)
        grid_x = np.array([sizes.min(), sizes.max()])
        ax.plot(
            grid_x,
            pooled + trend["slope_per_layer"] * 100 * (grid_x - sizes.mean()),
            color=SECOND,
            lw=1.3,
            zorder=3,
        )

        def bars(row, x, **kw):
            ax.errorbar(
                x,
                100 * row["ratio"],
                yerr=[[100 * (row["ratio"] - row["ci_low"])], [100 * (row["ci_high"] - row["ratio"])]],
                capsize=2.5,
                capthick=1.1,
                **kw,
            )

        for r in nested:
            bars(
                r,
                r["span_size"] - 0.12,
                fmt="o",
                ms=4.5,
                color=BLUE,
                ecolor=BLUE,
                elinewidth=1.4,
                zorder=5,
                label="nested spans" if r is nested[-1] else None,
            )
        for i, r in enumerate(other):
            bars(
                r,
                r["span_size"] + 0.12 + 0.08 * i,
                fmt="s",
                ms=4,
                mfc=SURFACE,
                mec=ORANGE,
                mew=1.3,
                ecolor=ORANGE,
                elinewidth=1.2,
                zorder=4,
                label="other spans" if i == 0 else None,
            )
        ax.set_xlim(sizes.min() - 0.5, sizes.max() + 0.6)
        ax.set_xticks(sorted({int(s) for s in sizes}))
        ax.set_xlabel("span size (layers)")
        ax.set_ylabel("$R = A/K$   (%)")
        ax.legend(loc="best", handletextpad=0.3, borderpad=0.2, labelspacing=0.3)
        tidy(ax)
        save(fig, figdir, f"{tag}_fig8b_recovered_share")


# --------------------------------------------------------------------------- per-layer A vs K


def single_layer_rows(root, tag, site):
    rows = []
    for path in sorted(
        glob.glob(
            os.path.join(
                root,
                f"{tag}_sae_clevr_lite_layer*_{site}_question_causal",
                "results",
                "ablation_v2_results*.json",
            )
        )
    ):
        if path.endswith(".summary.json") and os.path.exists(path.replace(".summary.json", ".json")):
            continue
        data = load_json(path)
        layer = int(path.split("_layer")[1].split("_")[0])
        rows.append(
            {
                "layer": layer,
                "binding": data["binding"]["mean_margin_drop"],
                "random": data["random"].get("mean_margin_drop"),
                "passthrough": (data.get("passthrough_baseline") or {}).get("mean_margin_drop"),
                "perturb": data["binding"].get("mean_relative_perturbation"),
            }
        )
    return sorted(rows, key=lambda r: r["layer"])


def fig_single(rows, ko_rows, figdir, tag, n_layers, span):
    if not rows:
        print("fig_single: no single-layer ablation results yet")
        return
    ko = {r["layer"]: r["mean_margin_drop"] for r in ko_rows if r["flow"] == "Image->Question"}
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    if span:
        ax.axvspan(min(span) - 0.5, max(span) + 0.5, color=BLUE, alpha=0.07, lw=0, zorder=0)
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)
    layers = [r["layer"] for r in rows]
    if ko:
        ax.plot(
            [layer for layer in layers if layer in ko],
            [ko[layer] for layer in layers if layer in ko],
            color=BLUE,
            lw=2,
            marker="o",
            ms=4.5,
            label="Image->Question knockout $K$ (n = all correct)",
            zorder=3,
        )
    ax.plot(
        layers,
        [r["binding"] for r in rows],
        color=ORANGE,
        lw=2,
        marker="o",
        ms=4.5,
        label="top-200 feature ablation $A$ (n = 256)",
        zorder=4,
    )
    ax.plot(
        layers,
        [r["random"] for r in rows],
        color=MUTED,
        lw=1.2,
        marker="o",
        ms=3,
        ls="--",
        label="matched random controls (mean of 15)",
        zorder=3,
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("mean margin drop")
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xlim(-1, n_layers)
    ax.legend(loc="upper right")
    tidy(ax)
    save(fig, figdir, f"{tag}_fig_single_layer")


# --------------------------------------------------------------------------- decomposition


def fig_decomposition(root, tag, figdir, span, n_layers):
    """Margin drop = true-option loss + false-option rise, per layer, both interventions."""
    path = os.path.join(root, f"{tag}_metric_decomposition.json")
    if not os.path.exists(path):
        print("fig_decomposition: no decomposition file")
        return
    rep = load_json(path)
    sweep = rep["sweep"].get("Image->Question", {})
    single = rep["single_layer"]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True, gridspec_kw={"hspace": 0.2})
    panels = [
        (
            axes[0],
            "a. Image->Question knockout, per layer (n = %d correct items)"
            % (next(iter(sweep.values()))["n"] if sweep else 0),
            {int(k): v for k, v in sweep.items()},
        ),
        (
            axes[1],
            "b. Top-200 feature ablation, per layer (n = 256)",
            {int(k): v["binding"] for k, v in single.items()},
        ),
    ]
    for ax, title, data in panels:
        layers = sorted(data)
        true_drop = np.array([data[l]["true_drop"] for l in layers])
        false_rise = np.array([data[l]["false_rise"] for l in layers])
        if span:
            ax.axvspan(min(span) - 0.5, max(span) + 0.5, color=BLUE, alpha=0.07, lw=0, zorder=0)
        ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)
        ax.bar(layers, false_rise, width=0.8, color=ORANGE, alpha=0.85, label="false option rises", zorder=3)
        ax.bar(
            layers,
            true_drop,
            width=0.8,
            color=BLUE,
            label="true option falls",
            zorder=4,
            bottom=np.where(np.sign(true_drop) == np.sign(false_rise), false_rise, 0),
        )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("margin drop (nats)")
        tidy(ax)
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("layer")
    axes[1].set_xticks(range(0, n_layers, 2))
    axes[1].set_xlim(-1, n_layers)
    save(fig, figdir, f"{tag}_fig_decomposition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/experiments")
    parser.add_argument("--tag", default="gemma3_4b")
    parser.add_argument("--site", default="attn_z")
    parser.add_argument("--n_layers", type=int, default=34)
    parser.add_argument("--span", type=int, nargs="*", default=[])
    parser.add_argument("--concentrated", type=int, default=None)
    parser.add_argument("--multilayer_dir", default=None)
    parser.add_argument("--figdir", default="paper/gemma3_figures")
    args = parser.parse_args()

    ko = knockout_rows(args.root, args.tag)
    fig1(ko, args.span, args.figdir, args.tag, args.n_layers)
    fig_decomposition(args.root, args.tag, args.figdir, args.span, args.n_layers)
    fig_single(
        single_layer_rows(args.root, args.tag, args.site), ko, args.figdir, args.tag, args.n_layers, args.span
    )
    ml = args.multilayer_dir or next(
        iter(glob.glob(os.path.join(args.root, f"{args.tag}_multilayer_*"))), None
    )
    if ml and os.path.isdir(os.path.join(ml, "conditions")):
        if args.concentrated is not None:
            fig6(ml, args.figdir, args.tag, args.concentrated)
        if os.path.exists(os.path.join(ml, "analysis", "multilayer_summary.json")):
            fig8(ml, args.figdir, args.tag)


if __name__ == "__main__":
    main()
