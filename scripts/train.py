"""Train mini-WAM / mini-VLA on image, state, action, and text-instruction datasets.

Expected dataset format: .npz with at least:

    images      : (N, H, W, 3)
    states      : (N, state_dim)
    actions     : (N, action_dim)
    text_ids    : (N, T_text)
    vocab       : dict

For WAM training, the dataset should also include:

    next_states : (N, state_dim)
    next_images : (N, H, W, 3)
    episode_ids : (N,)    enables multi-step imagination windows (--horizon > 1)
    is_expert   : (N,)    the policy imitates only expert actions; the world
                          model learns from every transition

scripts/collect_data.py writes all of these.

Each training sample is a window of `horizon` transitions inside one episode:

    o_t, a_t, a_t+1, ..., a_t+H-1, o_t+1, ..., o_t+H

The WAM encodes o_t, imagines H steps with the real actions, and is scored
against the real future observations (see models/losses.py).
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.wam_policy import MiniWAMPolicy
from models.losses import LossWeights, compute_wam_loss


class TrainingDataset(Dataset):
    def __init__(self, path: str, horizon: int = 1, resize_to: int = 64):
        data = np.load(path, allow_pickle=True)

        self.images = data["images"]          # (N, H, W, 3)
        self.states = data["states"]          # (N, state_dim)
        self.actions = data["actions"]        # (N, action_dim)
        self.text_ids = data["text_ids"]      # (N, T_text)

        self.next_states = data["next_states"] if "next_states" in data else None
        self.next_images = data["next_images"] if "next_images" in data else None
        # 1 = expert action (imitate it), 0 = exploration action (world model only)
        self.is_expert = data["is_expert"] if "is_expert" in data else np.ones(len(self.actions), np.float32)

        n = self.images.shape[0]
        if "episode_ids" in data:
            self.episode_ids = data["episode_ids"]
        else:
            if horizon > 1:
                raise ValueError("--horizon > 1 needs 'episode_ids' in the dataset.")
            self.episode_ids = np.zeros(n, dtype=np.int64)

        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
        self.state_labels = (
            [str(s) for s in data["state_labels"]]
            if "state_labels" in data
            else [f"s{i}" for i in range(self.states.shape[1])]
        )
        self.env = str(data["env"]) if "env" in data else "unknown"

        self.horizon = horizon
        self.resize_to = resize_to

        # a window starting at i is valid if i ... i+H-1 are all in the same episode
        last = np.arange(n) + horizon - 1
        valid = last < n
        valid[valid] &= self.episode_ids[last[valid]] == self.episode_ids[np.arange(n)[valid]]
        self.starts = np.nonzero(valid)[0]

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

    def __len__(self):
        return len(self.starts)

    def subset(self, episodes):
        """Same arrays (no copy), only windows from the given episodes."""
        sub = copy.copy(self)
        sub.starts = self.starts[np.isin(self.episode_ids[self.starts], episodes)]
        return sub

    def _process_image(self, img):
        if self.cv2 is not None and (
            img.shape[0] != self.resize_to or img.shape[1] != self.resize_to
        ):
            img = self.cv2.resize(img, (self.resize_to, self.resize_to))

        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return img

    def __getitem__(self, idx):
        i = self.starts[idx]
        window = slice(i, i + self.horizon)

        batch = {
            "image": self._process_image(self.images[i]),
            "state": torch.from_numpy(self.states[i]).float(),
            "action": torch.from_numpy(self.actions[i]).float(),
            "is_expert": torch.tensor(float(self.is_expert[i])),
            "text_ids": torch.from_numpy(self.text_ids[i]).long(),
            "actions": torch.from_numpy(self.actions[window]).float(),
        }

        # o_t+k is next_*[t+k-1], so the H future observations are next_*[t : t+H]
        if self.next_states is not None:
            batch["future_states"] = torch.from_numpy(self.next_states[window]).float()

        if self.next_images is not None:
            batch["future_images"] = torch.stack(
                [self._process_image(img) for img in self.next_images[window]]
            )

        return batch


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset / environment identity flags.
    # These do not directly construct envs in train.py, but they make
    # checkpoints and CLI usage consistent with collect_data.py / test.py.
    parser.add_argument(
        "--env",
        type=str,
        default="toy",
        choices=["toy", "metaworld", "mujoco"],
    )

    parser.add_argument(
        "--task",
        type=str,
        default="push-v3",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="wam",
        choices=["vla", "wam"],
    )

    parser.add_argument(
        "--action_head",
        type=str,
        default="mlp",
        choices=["mlp", "diffusion", "flow"],
    )

    parser.add_argument(
        "--image_encoder",
        type=str,
        default="keypoints",
        choices=["cnn", "keypoints"],
        help="cnn = mini-VLA TinyCNN (global pooling); keypoints = spatial-softmax CNN",
    )

    parser.add_argument("--predict_state", action="store_true")
    parser.add_argument("--predict_image", action="store_true")
    parser.add_argument(
        "--mlp_readout",
        action="store_true",
        help="Use an MLP instead of a linear probe to read the state out of the latent.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
    )

    # Training/data flags.
    parser.add_argument("--dataset-path", type=str, default="data/dataset.npz")
    parser.add_argument("--resize-to", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=5, help="Imagination steps per training window.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of episodes held out.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    # Model flags.
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--diffusion-T", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=8)

    # Loss weights.
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=1.0)

    parser.add_argument("--save-path", type=str, default="checkpoints/model.pt")

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if device_arg == "mps":
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def validate_args_and_dataset(args, dataset):
    if args.model_type == "vla":
        if args.predict_state:
            print("[warning] --model_type vla ignores --predict_state.")
        if args.predict_image:
            print("[warning] --model_type vla ignores --predict_image.")

        args.predict_state = False
        args.predict_image = False
        return

    if dataset.next_states is None or dataset.next_images is None:
        raise ValueError(
            "WAM training requires 'next_states' and 'next_images' in the dataset "
            "(the latent dynamics is trained against encoded future observations). "
            "Re-collect with scripts/collect_data.py."
        )

    if not (args.predict_state or args.predict_image):
        print(
            "[warning] WAM with neither --predict_state nor --predict_image: the latent "
            "dynamics is only anchored by the action loss and may collapse."
        )


def split_episodes(episode_ids, val_frac, seed):
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val = int(round(val_frac * len(episodes)))
    if val_frac > 0 and len(episodes) > 1:
        n_val = max(n_val, 1)
    return np.sort(episodes[n_val:]), np.sort(episodes[:n_val])


def run_epoch(model, loader, weights, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    totals, num_samples = {}, 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["image"].size(0)

        with torch.set_grad_enabled(training):
            outputs = model(batch)
            loss, logs = compute_wam_loss(outputs, batch, model, weights)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        num_samples += batch_size
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value.item() * batch_size

    return {key: value / max(num_samples, 1) for key, value in totals.items()}


def format_logs(logs):
    short = {
        "total_loss": "loss",
        "action_loss": "action",
        "state_recon_loss": "s_rec",
        "state_pred_loss": "s_pred",
        "image_recon_loss": "img_rec",
        "image_pred_loss": "img_pred",
        "latent_loss": "latent",
    }
    return " ".join(f"{short[k]}={logs[k]:.4f}" for k in short if k in logs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    horizon = args.horizon if args.model_type == "wam" else 1

    full = TrainingDataset(args.dataset_path, horizon=horizon, resize_to=args.resize_to)
    validate_args_and_dataset(args, full)

    train_eps, val_eps = split_episodes(full.episode_ids, args.val_frac, args.seed)
    train_set = full.subset(train_eps)
    val_set = full.subset(val_eps)
    print(
        f"Episodes: {len(train_eps)} train / {len(val_eps)} val | "
        f"windows: {len(train_set)} train / {len(val_set)} val | horizon={horizon}"
    )

    vocab_size = max(full.vocab.values()) + 1
    state_dim = full.states.shape[1]
    action_dim = full.actions.shape[1]

    model = MiniWAMPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        action_head=args.action_head,
        image_encoder=args.image_encoder,
        image_size=args.resize_to,
        model_type=args.model_type,
        predict_state=args.predict_state,
        predict_image=args.predict_image,
        linear_readout=not args.mlp_readout,
        diffusion_T=args.diffusion_T,
        flow_steps=args.flow_steps,
    ).to(device)

    train_states = full.states[np.isin(full.episode_ids, train_eps)]
    model.set_state_stats(train_states.mean(axis=0), train_states.std(axis=0))

    train_idx = np.nonzero(np.isin(full.episode_ids, train_eps))[0]
    sample = train_idx[:: max(1, len(train_idx) // 2000)]
    model.set_image_mean(torch.stack([full._process_image(full.images[i]) for i in sample]).mean(dim=0))

    weights = LossWeights(
        action=args.action_weight,
        state=args.state_weight,
        image=args.image_weight,
        latent=args.latent_weight,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        train_logs = run_epoch(model, train_loader, weights, device, optimizer)
        val_logs = run_epoch(model, val_loader, weights, device) if len(val_set) else {}

        history.append({"epoch": epoch + 1, "train": train_logs, "val": val_logs})
        print(f"Epoch {epoch + 1}/{args.epochs} train: {format_logs(train_logs)}")
        if val_logs:
            print(f"{'':>{len(str(args.epochs)) * 2 + 7}} val:   {format_logs(val_logs)}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "vocab": full.vocab,
            "state_labels": full.state_labels,
            "train_episodes": train_eps.tolist(),
            "val_episodes": val_eps.tolist(),
            "horizon": horizon,
            "env": full.env if full.env != "unknown" else args.env,
            "task": args.task,
            "resize_to": args.resize_to,
            "dataset_path": args.dataset_path,
            "args": vars(args),
            "history": history,
        },
        args.save_path,
    )

    with open(os.path.splitext(args.save_path)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Saved checkpoint:", args.save_path)


if __name__ == "__main__":
    main()
