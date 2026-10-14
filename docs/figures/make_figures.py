"""Build the result figures for docs/mini-wam-paper.md.

Reads the per-seed JSON files under docs/results/<experiment>/seed*/ (copied
from the runs of scripts/run_toy.sh) and writes SVG figures next to this
script. Every number plotted comes from those files.

    python docs/figures/make_figures.py
"""

import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

# Okabe-Ito colorblind-safe palette
WAM = "#D55E00"      # vermillion
VLA = "#0072B2"      # blue
BASE = "#7F7F7F"     # grey: "copy" / "no-op" reference baselines
ACCENT = "#009E73"   # green

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
})
WIDTH = 6.5  # inches, full text width


# ------------------------------------------------------------------ loading


def load_experiment(name):
    """Per-seed dicts with keys eval / wam_test / vla_test (missing files -> None)."""
    seeds = []
    for d in sorted(glob.glob(os.path.join(RESULTS, name, "seed*"))):
        rec = {}
        for key, fname in [("eval", "wam_eval.json"), ("wam_test", "wam_test.json"), ("vla_test", "vla_test.json")]:
            path = os.path.join(d, fname)
            rec[key] = json.load(open(path)) if os.path.exists(path) else None
        seeds.append(rec)
    return seeds


def curve(seeds, getter):
    """Stack a per-seed list-valued metric -> (mean, std, n) over seeds."""
    rows = [getter(s) for s in seeds]
    rows = [np.asarray(r, dtype=float) for r in rows if r is not None]
    if not rows:
        return None
    L = min(len(r) for r in rows)
    a = np.stack([r[:L] for r in rows])
    return a.mean(0), a.std(0), len(rows)


def scalar(seeds, getter):
    vals = []
    for s in seeds:
        try:
            v = getter(s)
        except (TypeError, KeyError, IndexError):
            v = None
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def panel_label(ax, letter):
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def band(ax, x, c, color, label, ls="-", marker="o", start=0):
    m, s, _ = c
    x = np.asarray(x)[: len(m)]
    x, m, s = x[start:], m[start:], s[start:]
    ax.plot(x, m, color=color, ls=ls, marker=marker, ms=2.5, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path)
    if os.environ.get("PREVIEW_DIR"):  # optional raster preview for quick checks
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".svg", ".png")), dpi=200)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


# ------------------------------------------------------------------ figures


def fig_imagination_quality(base):
    """(a) factual imagination vs copy baseline, (b) counterfactual vs no-op, (c) image."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))

    ax = axes[0]
    k = np.arange(0, 16)
    # k = 0 is the current state, where "nothing changes" is exact (0 on a log axis)
    band(ax, k, curve(base, lambda s: s["eval"]["state_nmse_copy"]), BASE, '"nothing changes"', ls="--", marker="s", start=1)
    band(ax, k, curve(base, lambda s: s["eval"]["state_nmse"]), WAM, "mini-WAM imagination")
    ax.set_yscale("log")
    ax.set_xlabel("imagined steps $k$")
    ax.set_ylabel("state error (normalized MSE)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32))
    panel_label(ax, "a")

    ax = axes[1]
    k = np.arange(1, 11)
    band(ax, k, curve(base, lambda s: s["eval"]["counterfactual"]["noop_nmse"]), BASE, "ignore the actions (no-op)", ls="--", marker="s")
    band(ax, k, curve(base, lambda s: s["eval"]["counterfactual"]["wam_nmse"]), WAM, "mini-WAM imagination")
    ax.set_yscale("log")
    ax.set_xlabel("imagined steps $k$ (random actions)")
    ax.set_ylabel("counterfactual error (nMSE)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32))
    panel_label(ax, "b")

    ax = axes[2]
    k = np.arange(0, 16)
    band(ax, k, curve(base, lambda s: s["eval"]["image_mse_copy"]), BASE, "copy current frame", ls="--", marker="s")
    band(ax, k, curve(base, lambda s: s["eval"]["image_mse"]), WAM, "mini-WAM decoded imagination")
    ax.set_xlabel("imagined steps $k$")
    ax.set_ylabel("image error (pixel MSE)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32))
    panel_label(ax, "c")

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig-imagination-quality.svg")


def fig_exploration(exps):
    """Factual vs counterfactual error and closed-loop success as exploration varies."""
    xs, names = [], []
    for p, name in [(0.0, "explore-0.0"), (0.3, "baseline"), (0.6, "explore-0.6")]:
        if exps.get(name):
            xs.append(p)
            names.append(name)
    if len(xs) < 2:
        return

    def series(getter):
        vals = [scalar(exps[n], getter) for n in names]
        return np.array([v[0] for v in vals]), np.array([v[1] for v in vals])

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.8, 2.6))
    ax = axes[0]
    for getter, color, label, marker in [
        (lambda s: s["eval"]["state_nmse"][10], ACCENT, "factual (real actions), $k$=10", "s"),
        (lambda s: s["eval"]["counterfactual"]["wam_nmse"][9], WAM, "counterfactual (random actions), $k$=10", "o"),
        (lambda s: s["eval"]["counterfactual"]["noop_nmse"][9], BASE, "ignore the actions (no-op)", "x"),
    ]:
        m, sd = series(getter)
        ax.errorbar(xs, m, yerr=sd, color=color, marker=marker, ms=3.5, capsize=2, label=label,
                    ls="--" if color == BASE else "-")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xlabel("random-action probability in the data")
    ax.set_ylabel("imagined state error (nMSE)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3))
    panel_label(ax, "a")

    ax = axes[1]
    for off, (key, color, label) in zip([-0.008, 0.008], [("wam_test", WAM, "mini-WAM"), ("vla_test", VLA, "VLA baseline")]):
        m, sd = series(lambda s, key=key: s[key]["success_rate"])
        ax.errorbar(np.array(xs) + off, 100 * m, yerr=100 * sd, color=color, marker="o", ms=3.5, capsize=2, label=label)
    ax.set_xticks(xs)
    ax.set_ylim(0, 105)
    ax.set_xlabel("random-action probability in the data")
    ax.set_ylabel("closed-loop success (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    panel_label(ax, "b")

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig-exploration.svg")


def fig_ablations(exps):
    """Dot plot: each objective/horizon variant vs the baseline on three metrics."""
    variants = [
        ("baseline", "full objective, $H$=5"),
        ("no-latent", "no latent-consistency loss"),
        ("no-image", "no image loss"),
        ("horizon-1", "imagination horizon $H$=1"),
        ("horizon-10", "imagination horizon $H$=10"),
    ]
    variants = [(n, l) for n, l in variants if exps.get(n)]
    if len(variants) < 2:
        return

    metrics = [
        ("imagined state error, $k$=15", lambda s: s["eval"]["state_nmse"][15], True),
        ("counterfactual error, $k$=10", lambda s: s["eval"]["counterfactual"]["wam_nmse"][9], True),
        ("closed-loop success (%)", lambda s: 100 * s["wam_test"]["success_rate"], False),
        ("image error, $k$=1", lambda s: s["eval"]["image_mse"][1], False),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(WIDTH, 0.35 * len(variants) + 0.7), sharey=True)
    y = np.arange(len(variants))[::-1]
    for j, (ax, (label, getter, log)) in enumerate(zip(axes, metrics)):
        hi = 0.0
        for yi, (name, _) in zip(y, variants):
            v = scalar(exps[name], getter)
            if v is None:
                continue
            hi = max(hi, v[0] + v[1])
            color = BASE if name == "baseline" else WAM
            ax.errorbar(v[0], yi, xerr=v[1], color=color, marker="o", ms=4, capsize=2, lw=1)
        base = scalar(exps["baseline"], getter)
        if base:
            ax.axvline(base[0], color=BASE, lw=0.6, ls=":")
        ax.set_xlim(0, 105 if "success" in label else 1.2 * hi)
        ax.set_xlabel(label)
        ax.set_yticks(y)
        if j == 0:
            ax.set_yticklabels([l for _, l in variants])
        panel_label(ax, "abcd"[j])
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig-ablations.svg")


def fig_data_and_state(exps):
    """(a) closed-loop success vs number of demonstrations, (b) with vs without state input."""
    pts = [(n, name) for n, name in [(25, "episodes-25"), (50, "episodes-50"), (100, "episodes-100"), (400, "baseline")] if exps.get(name)]
    has_img = bool(exps.get("image-only"))
    if len(pts) < 2 and not has_img:
        return

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.75, 2.0), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    xs = [n for n, _ in pts]
    for key, color, label in [("wam_test", WAM, "mini-WAM"), ("vla_test", VLA, "VLA baseline")]:
        vals = [scalar(exps[name], lambda s, key=key: 100 * s[key]["success_rate"]) for _, name in pts]
        ax.errorbar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], color=color, marker="o", ms=3.5,
                    capsize=2, label=label)
    if exps.get("episodes-25-matched"):
        # same 25 episodes, trained ~15x longer to match the baseline's gradient steps
        for off, (key, color) in zip([0.94, 1.06], [("wam_test", WAM), ("vla_test", VLA)]):
            v = scalar(exps["episodes-25-matched"], lambda s, key=key: 100 * s[key]["success_rate"])
            ax.errorbar(25 * off, v[0], yerr=v[1], color=color, marker="o", mfc="white", ms=4, capsize=2, ls="none")
        ax.plot([], [], color="black", marker="o", mfc="white", ls="none", label="25 episodes, matched steps")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(0, 105)
    ax.set_xlabel("demonstration episodes (per seed)")
    ax.set_ylabel("closed-loop success (%)")
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    ax = axes[1]
    if has_img:
        groups = [("baseline", "state + image"), ("image-only", "image only")]
        xpos = np.arange(len(groups))
        for off, (key, color, label) in zip([-0.17, 0.17], [("wam_test", WAM, "mini-WAM"), ("vla_test", VLA, "VLA baseline")]):
            vals = [scalar(exps[g], lambda s, key=key: 100 * s[key]["success_rate"]) for g, _ in groups]
            ax.bar(xpos + off, [v[0] for v in vals], yerr=[v[1] for v in vals], width=0.32, color=color,
                   label=label, capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(xpos)
        ax.set_xticklabels([l for _, l in groups])
        ax.set_ylim(0, 105)
        ax.set_ylabel("closed-loop success (%)")
        panel_label(ax, "b")
    else:
        ax.axis("off")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig-data-and-state.svg")


def main():
    names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(RESULTS, "*"))) if os.path.isdir(p)]
    exps = {n: load_experiment(n) for n in names}
    print("experiments:", ", ".join(f"{n} ({len(s)} seeds)" for n, s in exps.items()))
    fig_imagination_quality(exps["baseline"])
    fig_exploration(exps)
    fig_ablations(exps)
    fig_data_and_state(exps)


if __name__ == "__main__":
    main()
