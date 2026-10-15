"""Build every data figure of the pico-WAM paper.

Inputs (all committed):
    docs/results/<experiment>/seed*/        per-seed evaluation and closed-loop JSON
                                            files written by scripts/run_toy.sh
    docs/results/figure-data/               model outputs for the qualitative figures,
                                            written by docs/figures/export_figure_data.py

Outputs: <name>.pdf (LaTeX paper, docs/paper/) and <name>.svg (Markdown paper)
next to this script, built at the paper's text width (5.5 in). Every value
plotted is read from those files; nothing is typed in by hand.

    python docs/figures/make_figures.py
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import figstyle as fs  # noqa: E402

fs.use_style()
import matplotlib.pyplot as plt  # noqa: E402  (after use_style selects the backend)
from matplotlib.lines import Line2D  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
FIGDATA = os.path.join(RESULTS, "figure-data")
W = fs.TEXT  # 5.5 in, the paper's \textwidth

# one colour per role, used identically in every figure
WAM = fs.PALETTE["red"]        # pico-WAM
VLA = fs.PALETTE["blue"]       # VLA baseline
FACT = fs.PALETTE["green"]     # factual imagination (real actions)
DREAM = fs.PALETTE["purple"]   # dreamed (own policy)
TRUE = "#000000"               # ground truth
REF = fs.BASELINE              # reference baselines ("nothing changes", "ignore the actions")

STATE_NAMES = {"agent_x": "agent $x$", "agent_y": "agent $y$", "vel_x": "velocity $v_x$",
               "vel_y": "velocity $v_y$", "target_x": "goal $g_x$", "target_y": "goal $g_y$"}


# ---------------------------------------------------------------- helpers


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


def per_seed(seeds, getter):
    """Array of one scalar per seed."""
    return np.asarray([float(getter(s)) for s in seeds])


def curves(seeds, getter):
    """(n_seeds, n_points) array of a list-valued metric."""
    rows = [np.asarray(getter(s), dtype=float) for s in seeds]
    n = min(len(r) for r in rows)
    return np.stack([r[:n] for r in rows])


def finish(fig, name):
    if os.environ.get("PREVIEW_DIR"):  # optional raster preview for visual checks
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name + ".png"), dpi=220)
    fs.save(fig, os.path.join(HERE, name))


def band(ax, x, runs, color, label, ls="-", marker="o"):
    """Seed mean as a line, ± one standard deviation across seeds as a band."""
    m, s = runs.mean(0), runs.std(0)
    ax.plot(x, m, color=color, ls=ls, marker=marker, ms=2.2, lw=1.1, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, lw=0)


def image_axis(ax):
    """Image panels carry no axes (they are pictures, not plots)."""
    ax.set_axis_off()


# ------------------------------------------------- imagination quality


def fig_imagination_quality(base):
    fig, axes = fs.figure_grid(1, 3, width=W, ratio=0.42)

    ax = axes[0]
    k = np.arange(1, 16)  # k = 0 is the current state, where "nothing changes" is exact
    band(ax, k, curves(base, lambda s: s["eval"]["state_nmse_copy"])[:, 1:16], REF, '"nothing changes"', ls="--", marker="s")
    band(ax, k, curves(base, lambda s: s["eval"]["state_nmse"])[:, 1:16], WAM, "pico-WAM")
    ax.set_yscale("log")
    ax.set_xlabel("imagined steps $k$")
    ax.set_ylabel("state error (normalized MSE)")

    ax = axes[1]
    k = np.arange(1, 11)
    band(ax, k, curves(base, lambda s: s["eval"]["counterfactual"]["noop_nmse"]), REF, "ignore the actions", ls="--", marker="s")
    band(ax, k, curves(base, lambda s: s["eval"]["counterfactual"]["wam_nmse"]), WAM, "pico-WAM")
    ax.set_yscale("log")
    ax.set_xlabel("imagined steps $k$ (random actions)")
    ax.set_ylabel("counterfactual error (nMSE)")

    ax = axes[2]
    k = np.arange(1, 16)
    band(ax, k, curves(base, lambda s: s["eval"]["image_mse_copy"])[:, 1:16], REF, "copy current frame", ls="--", marker="s")
    band(ax, k, curves(base, lambda s: s["eval"]["image_mse"])[:, 1:16], WAM, "pico-WAM (decoded)")
    ax.set_ylim(0, 0.009)
    ax.set_xlabel("imagined steps $k$")
    ax.set_ylabel("image error (pixel MSE)")

    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    fs.panel_labels(axes)
    finish(fig, "fig-imagination-quality")


# --------------------------------------------------- episode strip


def fig_episode_strip():
    d = np.load(os.path.join(FIGDATA, "episode_strip.npz"))
    rows = [("real", d["real"]), ("decoded $z_t$", d["recon"]),
            ("imagined\n(real actions)", d["imagined"]), ("dreamed\n(own policy)", d["dreamed"])]
    t = d["timesteps"]
    fig, axes = plt.subplots(len(rows), len(t), figsize=(W, 0.60 * W * len(rows) / len(t) + 0.3),
                             layout="constrained", gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    for r, (name, frames) in enumerate(rows):
        for c in range(len(t)):
            ax = axes[r, c]
            ax.imshow(frames[c], interpolation="nearest")
            image_axis(ax)
            if r == 0:
                ax.text(0.5, 1.06, f"$t={t[c]}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
            if c == 0:
                ax.text(-0.08, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontsize=7)
    finish(fig, "fig-episode-strip")


# --------------------------------------------------- exploration


def fig_exploration(exps):
    settings = [(0.0, "explore-0.0"), (0.3, "baseline"), (0.6, "explore-0.6")]
    xs = np.array([p for p, _ in settings])
    fig, axes = fs.figure_grid(1, 2, width=W, ratio=0.44)

    ax = axes[0]
    series = [
        (lambda s: s["eval"]["counterfactual"]["noop_nmse"][9], REF, "ignore the actions", "x", "--"),
        (lambda s: s["eval"]["counterfactual"]["wam_nmse"][9], WAM, "counterfactual (random actions)", "o", "-"),
        (lambda s: s["eval"]["state_nmse"][10], FACT, "factual (real actions)", "s", "-"),
    ]
    for getter, color, label, marker, ls in series:
        vals = np.stack([per_seed(exps[n], getter) for _, n in settings])  # (settings, seeds)
        ax.errorbar(xs, vals.mean(1), yerr=vals.std(1), color=color, marker=marker, ms=3.5, capsize=2, lw=1.1, ls=ls, label=label)
        for i, x in enumerate(xs):  # individual seeds
            ax.scatter(np.full(vals.shape[1], x), vals[i], s=5, color=color, alpha=0.45, lw=0, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xlabel("random-action probability in the data")
    ax.set_ylabel("imagined state error, $k=10$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)

    ax = axes[1]
    for off, key, color, label in [(-0.012, "wam_test", WAM, "pico-WAM"), (0.012, "vla_test", VLA, "VLA baseline")]:
        vals = np.stack([per_seed(exps[n], lambda s, key=key: 100 * s[key]["success_rate"]) for _, n in settings])
        ax.errorbar(xs + off, vals.mean(1), yerr=vals.std(1), color=color, marker="o", ms=3.5, capsize=2, lw=1.1, label=label)
    ax.set_xticks(xs)
    ax.set_ylim(0, 105)
    ax.set_xlabel("random-action probability in the data")
    ax.set_ylabel("closed-loop success (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)

    fs.panel_labels(axes)
    finish(fig, "fig-exploration")


# --------------------------------------------------- ablations


def fig_ablations(exps):
    variants = [
        ("baseline", "full objective ($H=5$)"),
        ("no-latent", "no latent consistency"),
        ("no-image", "no image loss"),
        ("horizon-1", "horizon $H=1$"),
        ("horizon-10", "horizon $H=10$"),
    ]
    metrics = [
        ("state error, $k=15$", lambda s: s["eval"]["state_nmse"][15]),
        ("counterfactual error, $k=10$", lambda s: s["eval"]["counterfactual"]["wam_nmse"][9]),
        ("closed-loop success (%)", lambda s: 100 * s["wam_test"]["success_rate"]),
    ]
    fig, axes = fs.figure_grid(1, len(metrics), width=W, ratio=0.36, sharey=True)
    y = np.arange(len(variants))[::-1]
    for ax, (label, getter) in zip(axes, metrics):
        hi = 0.0
        for yi, (name, _) in zip(y, variants):
            vals = per_seed(exps[name], getter)
            color = REF if name == "baseline" else WAM
            ax.errorbar(vals.mean(), yi, xerr=vals.std(), color=color, marker="o", ms=3.5, capsize=2, lw=1.0)
            ax.scatter(vals, np.full(len(vals), yi), s=6, color=color, alpha=0.45, lw=0, zorder=3)
            hi = max(hi, vals.max(), vals.mean() + vals.std())
        ax.axvline(per_seed(exps["baseline"], getter).mean(), color=REF, lw=0.6, ls=":")
        ax.set_xlim(0, 105 if "success" in label else 1.15 * hi)
        ax.set_xlabel(label)
        ax.set_yticks(y)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    axes[0].set_yticklabels([l for _, l in variants])
    axes[0].set_ylabel("training variant")
    fs.panel_labels(axes)
    finish(fig, "fig-ablations")


# --------------------------------------------------- beliefs


def fig_beliefs():
    d = np.load(os.path.join(FIGDATA, "episode_beliefs.npz"))
    labels = [str(l) for l in d["labels"]]
    show = ["agent_x", "agent_y", "vel_x", "vel_y"]
    fig, axes = fs.figure_grid(1, 4, width=W, ratio=0.34, sharex=True)
    t = np.arange(len(d["true"]))
    for ax, name in zip(axes, show):
        j = labels.index(name)
        ax.plot(t, d["true"][:, j], color=TRUE, lw=1.4, label="true state")
        ax.plot(t, d["now"][:, j], color=VLA, lw=1.0, label="read from encoded frame")
        ax.plot(t, d["imagined"][:, j], color=WAM, lw=1.0, ls="--", label="imagined from $t=0$ (real actions)")
        ax.plot(t, d["dreamed"][:, j], color=DREAM, lw=1.0, ls=":", label="dreamed from $t=0$ (own policy)")
        ax.set_ylabel(STATE_NAMES[name])
        ax.set_xlabel("timestep")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=2)
    fs.panel_labels(axes)
    finish(fig, "fig-beliefs")


# --------------------------------------------------- what if


def fig_what_if():
    d = np.load(os.path.join(FIGDATA, "what_if.npz"))
    names = [str(n) for n in d["names"]]
    fig = plt.figure(figsize=(W, 0.40 * W), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.25])

    # (a) trajectories in the arena, over the start frame
    ax = fig.add_subplot(gs[0])
    ax.imshow(d["start_image"], extent=[0, 1, 0, 1], alpha=0.35, interpolation="nearest")
    cmap = plt.get_cmap("twilight", 10)
    for i in range(len(names)):
        color = TRUE if i == 0 else cmap(i)
        ax.plot(d["true"][i, :, 0], d["true"][i, :, 1], color=color, lw=0.8, alpha=0.75)
        ax.plot(d["imagined"][i, :, 0], d["imagined"][i, :, 1], color=color, lw=1.3, ls="--")
        ax.scatter(d["imagined"][i, -1, 0], d["imagined"][i, -1, 1], color=color, s=7, zorder=4, lw=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("agent $x$")
    ax.set_ylabel("agent $y$")
    ax.legend([Line2D([], [], color="0.3", ls="--", lw=1.3), Line2D([], [], color="0.3", lw=0.8)],
              ["imagined", "true simulator"], loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6)

    # (b) decoded frames after 15 imagined steps, compass layout
    sub = gs[1].subgridspec(3, 3, wspace=0.04, hspace=0.04)
    order = {(0, 0): 4, (0, 1): 3, (0, 2): 2, (1, 0): 5, (1, 1): 0, (1, 2): 1, (2, 0): 6, (2, 1): 7, (2, 2): 8}
    frame_axes = []
    for (r, c), i in order.items():
        a = fig.add_subplot(sub[r, c])
        a.imshow(d["final_frames"][i], interpolation="nearest")
        image_axis(a)
        a.text(0.05, 0.95, names[i], transform=a.transAxes, ha="left", va="top", fontsize=5,
               bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
        frame_axes.append(a)

    # (c) endpoint error per push direction, all held-out episodes of all seeds
    ax_c = fig.add_subplot(gs[2])
    err, ign = d["endpoint_error"], d["ignore_push_error"]  # (seeds, directions)
    x = np.arange(len(names))
    ax_c.bar(x - 0.2, ign.mean(0), 0.4, color=fs.MUTED, edgecolor=REF, lw=0.5, label="ignore the push")
    ax_c.bar(x + 0.2, err.mean(0), 0.4, yerr=err.std(0), color=WAM, capsize=1.5, error_kw={"lw": 0.6}, label="pico-WAM")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(names, rotation=60, fontsize=6)
    ax_c.set_ylim(0, 0.68)
    ax_c.set_ylabel("endpoint error, 15 steps (arena units)")
    ax_c.set_xlabel("constant push direction")
    ax_c.legend(loc="upper center", ncol=2, fontsize=6)

    fs.panel_labels([ax, frame_axes[0], ax_c], pad=4.0)
    finish(fig, "fig-what-if")


# --------------------------------------------------- latent map


def fig_latent_map():
    d = np.load(os.path.join(FIGDATA, "latent_map.npz"))
    short = {"agent_x": "$x$", "agent_y": "$y$", "vel_x": "$v_x$", "vel_y": "$v_y$", "target_x": "$g_x$", "target_y": "$g_y$"}
    labels = [short[str(l)] for l in d["labels"]]
    ev = d["explained"]
    fig, axes = fs.figure_grid(1, 2, width=W, ratio=0.48, gridspec_kw={"width_ratios": [1.0, 1.05]})

    ax = axes[0]
    sc = ax.scatter(d["xy"][:, 0], d["xy"][:, 1], c=d["progress"], cmap=fs.SEQUENTIAL, s=2, alpha=0.6, lw=0, rasterized=True)
    ax.plot(d["real_traj"][:, 0], d["real_traj"][:, 1], color=TRUE, lw=1.2, label="encoded (real frames)")
    ax.plot(d["imagined_traj"][:, 0], d["imagined_traj"][:, 1], color=WAM, lw=1.2, ls="--", label="imagined from $t=0$")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)
    fig.colorbar(sc, ax=ax, label="episode progress", fraction=0.05, pad=0.02)

    ax = axes[1]
    im = ax.imshow(d["corr"], cmap=fs.DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("state variable")
    ax.set_yticks(range(len(ev)))
    ax.set_yticklabels([f"PC{i + 1} ({e:.0%})" for i, e in enumerate(ev)])
    ax.grid(False)
    for i in range(d["corr"].shape[0]):
        for j in range(d["corr"].shape[1]):
            v = d["corr"][i, j]
            if abs(v) >= 0.5:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5, color="white" if abs(v) > 0.75 else "black")
    fig.colorbar(im, ax=ax, label="Pearson correlation", fraction=0.05, pad=0.02)

    fs.panel_labels(axes)
    finish(fig, "fig-latent-map")


# --------------------------------------------------- keypoints


def fig_keypoints():
    runs = [("baseline", "(a) state +\nimage input"), ("image-only", "(b) image\ninput only")]
    stats = json.load(open(os.path.join(FIGDATA, "keypoint_stats.json")))
    fig = plt.figure(figsize=(W, 0.23 * W), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.0])
    strips = gs[0].subgridspec(2, 8, wspace=0.03, hspace=0.03)
    cmap = plt.get_cmap("tab20")
    for r, (name, title) in enumerate(runs):
        d = np.load(os.path.join(FIGDATA, f"keypoints_{name}.npz"))
        size = d["frames"].shape[1]
        for c in range(d["frames"].shape[0]):
            a = fig.add_subplot(strips[r, c])
            a.imshow(d["frames"][c], interpolation="nearest")
            kp = d["keypoints"][c]
            a.scatter((kp[:, 0] + 1) / 2 * (size - 1), (kp[:, 1] + 1) / 2 * (size - 1),
                      c=[cmap(i % 20) for i in range(kp.shape[0])], s=7, marker="x", lw=0.7)
            image_axis(a)
            if r == 0:
                a.text(0.5, 1.04, f"$t={d['timesteps'][c]}$", transform=a.transAxes, ha="center", va="bottom", fontsize=6)
            if c == 0:
                a.text(-0.1, 0.5, title, transform=a.transAxes, ha="right", va="center", fontsize=6.5)

    ax = fig.add_subplot(gs[1])
    for i, (name, _) in enumerate(runs):
        v = np.asarray(stats[name]["spread"])
        ax.bar(i, np.exp(np.log(v).mean()), 0.6, color=REF if name == "baseline" else WAM, alpha=0.85)  # geometric mean
        ax.scatter(np.full(len(v), i), v, s=8, color="black", zorder=3, lw=0)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["state +\nimage", "image\nonly"], fontsize=6.5)
    ax.set_ylabel("keypoint movement\n(std of position)")

    fs.panel_labels([ax], labels=["(c)"])
    finish(fig, "fig-keypoints")


# --------------------------------------------------- data and state input


def fig_data_and_state(exps):
    pts = [(25, "episodes-25"), (50, "episodes-50"), (100, "episodes-100"), (400, "baseline")]
    fig, axes = fs.figure_grid(1, 2, width=W, ratio=0.40, gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax = axes[0]
    xs = np.array([n for n, _ in pts])
    for off, key, color, label in [(0.96, "wam_test", WAM, "pico-WAM"), (1.04, "vla_test", VLA, "VLA baseline")]:
        vals = [per_seed(exps[name], lambda s, key=key: 100 * s[key]["success_rate"]) for _, name in pts]
        ax.errorbar(xs * off, [v.mean() for v in vals], yerr=[v.std() for v in vals], color=color, marker="o", ms=3.5,
                    capsize=2, lw=1.1, label=label)
        m = per_seed(exps["episodes-25-matched"], lambda s, key=key: 100 * s[key]["success_rate"])
        ax.errorbar([25 * off], [m.mean()], yerr=[m.std()], color=color, marker="o", mfc="white", ms=4, capsize=2, ls="none")
    ax.plot([], [], color="0.3", marker="o", mfc="white", ls="none", label="25 episodes, matched steps")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_ylim(70, 102)
    ax.set_xlabel("demonstration episodes per run")
    ax.set_ylabel("closed-loop success (%)")
    ax.legend(loc="lower right", fontsize=6)

    ax = axes[1]
    groups = [("baseline", "state + image"), ("image-only", "image only")]
    xpos = np.arange(len(groups))
    for off, key, color, label in [(-0.18, "wam_test", WAM, "pico-WAM"), (0.18, "vla_test", VLA, "VLA baseline")]:
        vals = [per_seed(exps[g], lambda s, key=key: 100 * s[key]["success_rate"]) for g, _ in groups]
        ax.bar(xpos + off, [v.mean() for v in vals], 0.34, color=color, label=label)
        for xp, v in zip(xpos, vals):
            ax.scatter(np.full(len(v), xp + off), v, s=7, color="black", zorder=3, lw=0)
    ax.set_xticks(xpos)
    ax.set_xticklabels([l for _, l in groups])
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("closed-loop success (%)")
    ax.legend(loc="upper center", ncol=2, fontsize=6)

    fs.panel_labels(axes)
    finish(fig, "fig-data-and-state")


# --------------------------------------------------- policy outcomes


def fig_policy(exps):
    """(a) data scaling, (b) state vs pixel input, (c) matched-supervision pilot."""
    pilot = json.load(open(os.path.join(RESULTS, "supervision-pilot", "summary.json")))
    fig, axes = fs.figure_grid(1, 3, width=W, ratio=0.38, gridspec_kw={"width_ratios": [1.25, 0.9, 1.0]})

    ax = axes[0]
    pts = [(25, "episodes-25"), (50, "episodes-50"), (100, "episodes-100"), (400, "baseline")]
    xs = np.array([n for n, _ in pts])
    for off, key, color, label in [(0.95, "wam_test", WAM, "pico-WAM"), (1.05, "vla_test", VLA, "VLA baseline")]:
        vals = [per_seed(exps[name], lambda s, key=key: 100 * s[key]["success_rate"]) for _, name in pts]
        ax.errorbar(xs * off, [v.mean() for v in vals], yerr=[v.std() for v in vals], color=color, marker="o",
                    ms=3, capsize=2, lw=1.1, label=label)
        m = per_seed(exps["episodes-25-matched"], lambda s, key=key: 100 * s[key]["success_rate"])
        ax.errorbar([25 * off], [m.mean()], yerr=[m.std()], color=color, marker="o", mfc="white", ms=3.5, capsize=2, ls="none")
    ax.plot([], [], color="0.3", marker="o", mfc="white", ls="none", label="25 episodes, matched updates")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_ylim(70, 102)
    ax.set_xlabel("demonstration episodes per run")
    ax.set_ylabel("closed-loop success (%)")
    ax.legend(loc="lower right", fontsize=5.5)

    ax = axes[1]
    groups = [("baseline", "state +\nimage"), ("image-only", "image\nonly")]
    xpos = np.arange(len(groups))
    for off, key, color, label in [(-0.18, "wam_test", WAM, "pico-WAM"), (0.18, "vla_test", VLA, "VLA")]:
        vals = [per_seed(exps[g], lambda s, key=key: 100 * s[key]["success_rate"]) for g, _ in groups]
        ax.bar(xpos + off, [v.mean() for v in vals], 0.34, color=color, label=label)
        for xp, v in zip(xpos, vals):
            ax.scatter(np.full(len(v), xp + off), v, s=6, color="black", zorder=3, lw=0)
    ax.set_xticks(xpos)
    ax.set_xticklabels([l for _, l in groups])
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("closed-loop success (%)")
    ax.legend(loc="upper center", ncol=2, fontsize=5.5)

    ax = axes[2]
    conds = [("bc", "actions\nonly"), ("current", "+ current\nstate"), ("future", "+ future\nstate")]
    rows = pilot["rows"]
    x = np.arange(len(conds))
    for r in rows:  # one line per training run, paired across conditions
        ax.plot(x, [100 * r[c] for c, _ in conds], color=REF, lw=0.6, alpha=0.7, marker="o", ms=2)
    means = [100 * pilot["summary"][c]["mean"] for c, _ in conds]
    sds = [100 * pilot["summary"][c]["sample_sd"] for c, _ in conds]
    ax.errorbar(x, means, yerr=sds, color=WAM, marker="s", ms=3.5, capsize=2, lw=1.4, label="mean ± SD")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in conds], fontsize=6)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0, 45)
    ax.set_ylabel("pixel-input success (%)")
    ax.legend(loc="lower right", fontsize=5.5)

    fs.panel_labels(axes)
    finish(fig, "fig-policy")


# --------------------------------------------------- efficiency


def fig_efficiency():
    d = json.load(open(os.path.join(RESULTS, "efficiency", "efficiency.json")))
    P = d["parameters"]
    fig, axes = fs.figure_grid(1, 3, width=W, ratio=0.36, gridspec_kw={"width_ratios": [1.25, 0.85, 1.0]})

    # (a) parameters per module, stacked
    ax = axes[0]
    groups = [("encoders + fusion", ["img_encoder", "txt_encoder", "state_encoder", "fusion"], "#BBBBBB"),
              ("action head", ["policy_head"], VLA),
              ("dynamics + readout", ["dynamics", "state_readout"], WAM),
              ("image decoder", ["image_decoder"], fs.PALETTE["orange"])]
    models = [("VLA", P["vla"]), ("WAM,\nstates only", {k: v for k, v in P["wam"].items() if k != "image_decoder"}), ("WAM", P["wam"])]
    y = np.arange(len(models))[::-1]
    for yi, (name, parts) in zip(y, models):
        left = 0.0
        for gname, keys, color in groups:
            w = sum(parts.get(k, 0) for k in keys) / 1e3
            ax.barh(yi, w, left=left, color=color, height=0.6, label=gname if yi == y[-1] else None)
            left += w
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in models], fontsize=6.5)
    ax.set_xlabel("parameters (thousands)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=2, fontsize=5.5)

    # (b) one optimizer step
    ax = axes[1]
    T = d["train_step_ms"]
    names = [("VLA", "vla", VLA), ("WAM,\nstates", "wam_state_only", WAM), ("WAM", "wam", fs.PALETTE["orange"])]
    ax.bar(range(3), [T[k] for _, k, _ in names], color=[c for _, _, c in names], width=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _, _ in names], fontsize=6.5)
    ax.set_ylabel("training step (ms, batch 64)")

    # (c) imagination cost vs horizon
    ax = axes[2]
    L = d["latency_ms"]
    ks = [1, 5, 10, 20]
    ax.plot(ks, [L[f"imagine_{k}_states"] for k in ks], color=WAM, marker="o", ms=3, label="latent + state readout")
    ax.plot(ks, [L[f"imagine_{k}_states_and_images"] for k in ks], color=fs.PALETTE["orange"], marker="s", ms=3, label="+ decode every image")
    ax.axhline(L["act_wam"], color=REF, ls="--", lw=0.8, label="one policy action")
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("imagined steps $k$")
    ax.set_ylabel("latency (ms, batch 1)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, fontsize=5.5)

    fs.panel_labels(axes)
    finish(fig, "fig-efficiency")


# --------------------------------------------------- toy world


def fig_toy_world():
    """(a) rendered frames of one expert episode, (b) one training episode with exploration bursts."""
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    from envs.toy_point_env import ToyPointEnv

    env = ToyPointEnv(seed=71)
    img, state, info = env.reset(seed=71, target="blue")
    frames, traj = [img], [state[:2]]
    for _ in range(40):
        img, state, _, done, _ = env.step(env.expert_action(noise=0.0))
        frames.append(img)
        traj.append(state[:2])
        if done:
            break
    T = len(frames) - 1
    picks = np.unique(np.linspace(0, T, 4).round().astype(int))

    # a training episode as collected by scripts/collect_data.py: bursts of 1-3 random actions, p = 0.3
    rng = np.random.default_rng(3)
    env2 = ToyPointEnv(seed=5)
    env2.reset(seed=5, target="green")
    acts, expert, burst = [], [], 0
    for _ in range(80):
        if burst == 0 and rng.random() < 0.3:
            burst = int(rng.integers(1, 4))
        if burst > 0:
            a = rng.uniform(-1, 1, 2)
            burst -= 1
            expert.append(False)
        else:
            a = env2.expert_action(noise=0.1)
            expert.append(True)
        acts.append(np.clip(a, -1, 1))
        _, _, _, done, _ = env2.step(a)
        if done:
            break
    acts = np.asarray(acts)

    fig = plt.figure(figsize=(W, 0.26 * W), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.4])
    strip = gs[0].subgridspec(1, len(picks), wspace=0.04)
    first = None
    for c, t in enumerate(picks):
        a = fig.add_subplot(strip[0, c])
        a.imshow(frames[t], interpolation="nearest")
        image_axis(a)
        a.text(0.5, 1.03, f"$t={t}$", transform=a.transAxes, ha="center", va="bottom", fontsize=6.5)
        first = first or a
    first.text(0.03, 0.04, '"go to the blue goal"', transform=first.transAxes, fontsize=5.5, color="0.2",
               bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})

    ax = fig.add_subplot(gs[1])
    t = np.arange(len(acts))
    for i, e in enumerate(expert):
        if not e:
            ax.axvspan(i - 0.5, i + 0.5, color=WAM, alpha=0.18, lw=0)
    ax.plot(t, acts[:, 0], color=TRUE, lw=1.0, label="$a_x$")
    ax.plot(t, acts[:, 1], color=VLA, lw=1.0, ls="--", label="$a_y$")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("timestep")
    ax.set_ylabel("action")
    ax.legend([Line2D([], [], color=TRUE, lw=1.0), Line2D([], [], color=VLA, lw=1.0, ls="--"),
               plt.Rectangle((0, 0), 1, 1, color=WAM, alpha=0.18)], ["$a_x$", "$a_y$", "random burst"],
              loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=5.5)

    fs.panel_labels([first, ax], pad=7.0)
    finish(fig, "fig-toy-world")


def main():
    names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(RESULTS, "*")))
             if os.path.isdir(p) and os.path.basename(p) not in ("figure-data", "efficiency", "research-audit", "supervision-pilot")]
    exps = {n: load_experiment(n) for n in names}
    fig_imagination_quality(exps["baseline"])
    fig_episode_strip()
    fig_exploration(exps)
    fig_ablations(exps)
    fig_beliefs()
    fig_what_if()
    fig_latent_map()
    fig_keypoints()
    fig_data_and_state(exps)
    fig_policy(exps)
    fig_efficiency()
    fig_toy_world()


if __name__ == "__main__":
    main()
