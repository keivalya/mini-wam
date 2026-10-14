"""Offline evaluation of a trained mini-WAM world model on held-out episodes.

Questions this answers (all on episodes the model never trained on):

1. Does the latent know the current world?        state readout R^2 at k = 0
2. Can it imagine the future?                      state / image error vs imagination step k
3. Is imagination better than "nothing changes"?   compared against a copy-current baseline
4. Which physical quantities live in the latent?   R^2 per state variable

The imagination here is open-loop: encode o_t once, then roll the latent
dynamics forward with the actions that were really taken, never looking at
o_t+1 ... o_t+k.

Usage:
    python -m scripts.evaluate --checkpoint checkpoints/wam.pt --dataset-path data/toy.npz
"""

import argparse
import json
import os

import numpy as np
import torch

from models.wam_policy import MiniWAMPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--dataset-path", type=str, default=None, help="Defaults to the training dataset.")
    parser.add_argument("--max-horizon", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Defaults to <checkpoint>_eval.json")
    return parser.parse_args()


# ----------------------------------------------------------------- loading


def load_model_and_data(checkpoint, dataset_path=None, device="cpu"):
    model, ckpt = MiniWAMPolicy.from_checkpoint(checkpoint, device=device)
    data = dict(np.load(dataset_path or ckpt["dataset_path"], allow_pickle=True))
    return model, ckpt, data


def to_image_tensor(images, device):
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float in [0, 1]"""
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div(255.0).to(device)


def get_episode(data, episode, device="cpu"):
    """
    All T transitions of one episode as T+1 observations and T actions:
        images  : (T+1, 3, H, W)
        states  : (T+1, state_dim)
        actions : (T, action_dim)
        text_ids: (1, T_text)
    """
    idx = np.nonzero(data["episode_ids"] == episode)[0]
    images = np.concatenate([data["images"][idx], data["next_images"][idx[-1:]]], axis=0)
    states = np.concatenate([data["states"][idx], data["next_states"][idx[-1:]]], axis=0)

    return {
        "images": to_image_tensor(images, device),
        "states": torch.from_numpy(states).float().to(device),
        "actions": torch.from_numpy(data["actions"][idx]).float().to(device),
        "text_ids": torch.from_numpy(data["text_ids"][idx[:1]]).long().to(device),
        "text": str(data["texts"][idx[0]]) if "texts" in data else "",
    }


# --------------------------------------------------------------- latents


@torch.no_grad()
def encode_episode(model, ep):
    """Real latents z_0 ... z_T, each encoded from its own observation. (T+1, D)"""
    n = ep["images"].shape[0]
    return model.encode_obs(ep["images"], ep["text_ids"].expand(n, -1), ep["states"])


@torch.no_grad()
def imagine_from_every_start(model, ep, horizon):
    """
    For every start t, imagine k = 1..horizon steps with the real actions.

    returns z_imag: (T, horizon, D) where z_imag[t, k-1] is the latent the model
    imagines for time t+k having only seen o_t. Entries past the end of the
    episode are NaN.
    """
    z_real = encode_episode(model, ep)
    T, A = ep["actions"].shape
    D = z_real.shape[-1]

    acts = torch.zeros(T, horizon, A, device=z_real.device)
    valid = torch.zeros(T, horizon, dtype=torch.bool, device=z_real.device)
    for t in range(T):
        k = min(horizon, T - t)
        acts[t, :k] = ep["actions"][t : t + k]
        valid[t, :k] = True

    z_imag = model.dynamics.rollout(z_real[:T], acts)
    z_imag = torch.where(valid[..., None], z_imag, torch.full_like(z_imag, float("nan")))
    return z_real, z_imag.view(T, horizon, D)


# --------------------------------------------------------------- metrics


def r2_per_dim(pred, true):
    """Coefficient of determination per column; NaN for (near-)constant columns."""
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r2[ss_tot < 1e-8 * len(true)] = np.nan
    return r2


@torch.no_grad()
def evaluate(model, data, episodes, max_horizon=15):
    """
    Collect, for every held-out (start t, step k) pair:
        true state s_t+k, imagined readout, copy baseline s_t
        image MSE of the imagined decode and of the copy baseline image_t
    """
    K = max_horizon
    per_k = {k: {"true": [], "pred": [], "copy": [], "img": [], "img_copy": []} for k in range(K + 1)}

    for e in episodes:
        ep = get_episode(data, e, device=model.state_mean.device)
        z_real, z_imag = imagine_from_every_start(model, ep, K)
        T = ep["actions"].shape[0]

        dec_real = model.decode(z_real)
        dec_imag = model.decode(torch.nan_to_num(z_imag))

        for t in range(T + 1):
            # k = 0: read out the real latent (reconstruction)
            per_k[0]["true"].append(ep["states"][t])
            per_k[0]["copy"].append(ep["states"][t])
            if "state" in dec_real:
                per_k[0]["pred"].append(dec_real["state"][t])
            if "image" in dec_real:
                per_k[0]["img"].append(((dec_real["image"][t] - ep["images"][t]) ** 2).mean())
                per_k[0]["img_copy"].append(torch.zeros(()))

        for t in range(T):
            for k in range(1, min(K, T - t) + 1):
                per_k[k]["true"].append(ep["states"][t + k])
                per_k[k]["copy"].append(ep["states"][t])
                if "state" in dec_imag:
                    per_k[k]["pred"].append(dec_imag["state"][t, k - 1])
                if "image" in dec_imag:
                    per_k[k]["img"].append(((dec_imag["image"][t, k - 1] - ep["images"][t + k]) ** 2).mean())
                    per_k[k]["img_copy"].append(((ep["images"][t] - ep["images"][t + k]) ** 2).mean())

    std = model.state_std.cpu().numpy()
    results = {"horizon": [], "n": [], "state_nmse": [], "state_nmse_copy": [],
               "image_mse": [], "image_mse_copy": [], "state_r2": []}

    for k in range(K + 1):
        d = per_k[k]
        if not d["true"]:
            break
        true = torch.stack(d["true"]).cpu().numpy()
        copy = torch.stack(d["copy"]).cpu().numpy()

        results["horizon"].append(k)
        results["n"].append(len(true))
        results["state_nmse_copy"].append(float((((copy - true) / std) ** 2).mean()))

        if d["pred"]:
            pred = torch.stack(d["pred"]).cpu().numpy()
            results["state_nmse"].append(float((((pred - true) / std) ** 2).mean()))
            results["state_r2"].append([None if np.isnan(r) else float(r) for r in r2_per_dim(pred, true)])
        if d["img"]:
            results["image_mse"].append(float(torch.stack(d["img"]).mean()))
            results["image_mse_copy"].append(float(torch.stack(d["img_copy"]).mean()))

    return results


def main():
    args = parse_args()
    model, ckpt, data = load_model_and_data(args.checkpoint, args.dataset_path, args.device)

    if not model.is_wam:
        raise SystemExit("This checkpoint is a VLA (no world model); nothing to evaluate here.")

    episodes = ckpt["val_episodes"] or sorted(set(data["episode_ids"].tolist()))
    results = evaluate(model, data, episodes, args.max_horizon)
    results["state_labels"] = ckpt["state_labels"]
    results["episodes"] = episodes
    results["checkpoint"] = args.checkpoint

    labels = ckpt["state_labels"]
    print(f"Held-out episodes: {len(episodes)}")
    print(f"{'k':>3} {'n':>6} {'state nMSE':>11} {'copy':>8} {'image MSE':>10} {'copy':>8}")
    for i, k in enumerate(results["horizon"]):
        s = results["state_nmse"][i] if results["state_nmse"] else float("nan")
        im = results["image_mse"][i] if results["image_mse"] else float("nan")
        imc = results["image_mse_copy"][i] if results["image_mse_copy"] else float("nan")
        print(f"{k:>3} {results['n'][i]:>6} {s:>11.4f} {results['state_nmse_copy'][i]:>8.4f} {im:>10.5f} {imc:>8.5f}")

    if results["state_r2"]:
        print("\nState readout R^2 (k=0: from the real latent, k>0: from the imagined latent)")
        show = [k for k in [0, 1, 5, 10, args.max_horizon] if k < len(results["state_r2"])]
        print(f"{'variable':>14} " + " ".join(f"{'k=' + str(k):>7}" for k in sorted(set(show))))
        for j, name in enumerate(labels):
            row = [results["state_r2"][k][j] for k in sorted(set(show))]
            print(f"{name:>14} " + " ".join(f"{'—':>7}" if r is None else f"{r:>7.3f}" for r in row))

    output = args.output or os.path.splitext(args.checkpoint)[0] + "_eval.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
