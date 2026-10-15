"""Export the data behind the paper's qualitative figures.

Loads trained checkpoints (one directory per seed, each holding wam.pt and
toy.npz as written by scripts/run_toy.sh) and saves exactly what the figures
show to docs/results/figure-data/. The figures are then drawn from those
files only (docs/figures/make_figures.py), so every pixel and every curve can
be traced back to a model output.

    python docs/figures/export_figure_data.py \\
        --baseline /path/to/baseline   # contains seed0/, seed1/, seed2/
        --image-only /path/to/image-only

Everything is computed on held-out episodes (the checkpoint's val_episodes).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from envs.toy_point_env import ToyPointEnv  # noqa: E402
from scripts.evaluate import load_model_and_data, get_episode, encode_episode  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "figure-data")
SEEDS = [0, 1, 2]
STRIP_FRAMES = 8
WHATIF_HORIZON = 15


def to_uint8(x):
    """(..., 3, H, W) float in [0, 1] -> (..., H, W, 3) uint8"""
    x = x.clamp(0, 1).cpu().numpy()
    return (np.moveaxis(x, -3, -1) * 255).round().astype(np.uint8)


def load(run_dir, seed):
    d = os.path.join(run_dir, f"seed{seed}")
    model, ckpt, data = load_model_and_data(os.path.join(d, "wam.pt"), os.path.join(d, "toy.npz"))
    return model, ckpt, data


@torch.no_grad()
def export_episode_views(model, ckpt, data):
    """Imagination strip and belief traces for the first held-out episode."""
    episode = ckpt["val_episodes"][0]
    ep = get_episode(data, episode)
    T = ep["actions"].shape[0]

    z_real = encode_episode(model, ep)
    imag = model.imagine(ep["images"][:1], ep["text_ids"], ep["states"][:1], T, ep["actions"][None])
    dream = model.imagine(ep["images"][:1], ep["text_ids"], ep["states"][:1], T)
    recon = model.decode(z_real)

    cols = np.unique(np.linspace(0, T, STRIP_FRAMES).round().astype(int))
    np.savez_compressed(
        os.path.join(OUT, "episode_strip.npz"),
        timesteps=cols,
        real=to_uint8(ep["images"][cols]),
        recon=to_uint8(recon["image"][cols]),
        imagined=to_uint8(imag["image"][0, cols]),
        dreamed=to_uint8(dream["image"][0, cols]),
        instruction=np.asarray(ep["text"]),
        episode=episode,
    )
    np.savez_compressed(
        os.path.join(OUT, "episode_beliefs.npz"),
        true=ep["states"].cpu().numpy(),
        now=recon["state"].cpu().numpy(),
        imagined=imag["state"][0].cpu().numpy(),
        dreamed=dream["state"][0].cpu().numpy(),
        labels=np.asarray(ckpt["state_labels"]),
        episode=episode,
    )


def push_plans(horizon):
    """No-op plus constant unit pushes in 8 compass directions: (9, horizon, 2)."""
    angles = np.deg2rad(np.arange(0, 360, 45))
    dirs = np.concatenate([np.zeros((1, 2)), np.stack([np.cos(angles), np.sin(angles)], 1)])
    return np.repeat(dirs[:, None, :], horizon, axis=1).astype(np.float32), ["no-op"] + [f"{a}°" for a in range(0, 360, 45)]


@torch.no_grad()
def imagine_plans(model, ep, t, plans):
    n = len(plans)
    return model.imagine(
        ep["images"][t : t + 1].expand(n, -1, -1, -1),
        ep["text_ids"].expand(n, -1),
        ep["states"][t : t + 1].expand(n, -1),
        plans.shape[1],
        torch.from_numpy(plans),
    )


@torch.no_grad()
def export_what_if(models):
    """Constant-push counterfactuals: one example (seed 0) + endpoint errors for all seeds."""
    sim = ToyPointEnv()
    plans, names = push_plans(WHATIF_HORIZON)

    model, ckpt, data = models[0]
    ep = get_episode(data, ckpt["val_episodes"][0])
    s0 = ep["states"][0].cpu().numpy()
    out = imagine_plans(model, ep, 0, plans)
    truth = np.stack([np.concatenate([s0[None], sim.simulate(s0, p)]) for p in plans])
    example = dict(
        start_image=to_uint8(ep["images"][0]),
        start_state=s0,
        imagined=out["state"].cpu().numpy(),
        true=truth,
        final_frames=to_uint8(out["image"][:, -1]),
        names=np.asarray(names),
    )

    # endpoint error (Euclidean distance of agent position after the push) for
    # every held-out episode start of every seed
    errors = np.zeros((len(models), len(plans)))
    ignore = np.zeros((len(models), len(plans)))  # reference: predict the no-op endpoint for every push
    counts = []
    for i, (model, ckpt, data) in enumerate(models):
        per_ep, per_ep_ignore = [], []
        for e in ckpt["val_episodes"]:
            ep = get_episode(data, e)
            s0 = ep["states"][0].cpu().numpy()
            pred = imagine_plans(model, ep, 0, plans)["state"][:, -1, :2].cpu().numpy()
            true = np.stack([sim.simulate(s0, p)[-1, :2] for p in plans])
            per_ep.append(np.linalg.norm(pred - true, axis=1))
            per_ep_ignore.append(np.linalg.norm(true[0][None] - true, axis=1))
        errors[i] = np.mean(per_ep, axis=0)
        ignore[i] = np.mean(per_ep_ignore, axis=0)
        counts.append(len(per_ep))

    np.savez_compressed(os.path.join(OUT, "what_if.npz"), endpoint_error=errors, ignore_push_error=ignore, n_episodes=np.asarray(counts), **example)


@torch.no_grad()
def export_latent_map(model, ckpt, data):
    """PCA of held-out latents (seed 0) and correlation of components with state variables."""
    zs, ss, prog = [], [], []
    for e in ckpt["val_episodes"]:
        ep = get_episode(data, e)
        z = encode_episode(model, ep).cpu().numpy()
        zs.append(z)
        ss.append(ep["states"].cpu().numpy())
        prog.append(np.linspace(0, 1, len(z)))
    Z, S, P = np.concatenate(zs), np.concatenate(ss), np.concatenate(prog)

    mean = Z.mean(0)
    _, sing, Vt = np.linalg.svd(Z - mean, full_matrices=False)
    explained = sing ** 2 / (sing ** 2).sum()
    pcs = (Z - mean) @ Vt[:8].T
    corr = np.array([[np.corrcoef(pcs[:, i], S[:, j])[0, 1] for j in range(S.shape[1])] for i in range(8)])

    ep = get_episode(data, ckpt["val_episodes"][0])
    T = ep["actions"].shape[0]
    z_real = encode_episode(model, ep).cpu().numpy()
    z_imag = model.imagine(ep["images"][:1], ep["text_ids"], ep["states"][:1], T, ep["actions"][None])["z"][0].cpu().numpy()

    np.savez_compressed(
        os.path.join(OUT, "latent_map.npz"),
        xy=pcs[:, :2],
        progress=P,
        explained=explained[:8],
        corr=corr,
        labels=np.asarray(ckpt["state_labels"]),
        real_traj=((z_real - mean) @ Vt[:2].T),
        imagined_traj=((z_imag - mean) @ Vt[:2].T),
    )


@torch.no_grad()
def keypoint_stats(model, ckpt, data):
    """
    How well do the spatial-softmax keypoints track the agent?

    For each keypoint, the squared correlation between its x (y) coordinate and
    the agent's x (y) position over all held-out frames; the score is the best
    keypoint's mean of the two. Also returns the keypoints' mean spatial spread.
    """
    kps, pos = [], []
    for e in ckpt["val_episodes"]:
        ep = get_episode(data, e)
        kps.append(model.img_encoder.keypoints(ep["images"]).cpu().numpy())   # (T+1, K, 2), x right, y down
        pos.append(ep["states"][:, :2].cpu().numpy())                          # (T+1, 2), y up
    K, A = np.concatenate(kps), np.concatenate(pos)
    r2x = np.array([np.corrcoef(K[:, k, 0], A[:, 0])[0, 1] ** 2 for k in range(K.shape[1])])
    r2y = np.array([np.corrcoef(K[:, k, 1], A[:, 1])[0, 1] ** 2 for k in range(K.shape[1])])
    score = np.nan_to_num((r2x + r2y) / 2)
    spread = K.std(axis=0).mean()
    return float(score.max()), float(spread)


@torch.no_grad()
def export_keypoints(runs):
    """Example frames (seed 0) and tracking scores (all seeds) for each run."""
    stats = {}
    for name, models in runs.items():
        model, ckpt, data = models[0]
        ep = get_episode(data, ckpt["val_episodes"][0])
        T = ep["images"].shape[0]
        cols = np.unique(np.linspace(0, T - 1, STRIP_FRAMES).round().astype(int))
        kp = model.img_encoder.keypoints(ep["images"][cols]).cpu().numpy()
        np.savez_compressed(
            os.path.join(OUT, f"keypoints_{name}.npz"),
            frames=to_uint8(ep["images"][cols]),
            keypoints=kp,
            timesteps=cols,
        )
        scores = [keypoint_stats(*m) for m in models]
        stats[name] = {
            "tracking_r2": [s[0] for s in scores],
            "spread": [s[1] for s in scores],
        }
    with open(os.path.join(OUT, "keypoint_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--image-only", required=True)
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    baseline = [load(args.baseline, s) for s in SEEDS]
    image_only = [load(args.image_only, s) for s in SEEDS]

    export_episode_views(*baseline[0])
    export_what_if(baseline)
    export_latent_map(*baseline[0])
    stats = export_keypoints({"baseline": baseline, "image-only": image_only})

    for f in sorted(os.listdir(OUT)):
        print("wrote", os.path.join("docs/results/figure-data", f))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
