"""Train mini-WAM / mini-VLA on image, state, action, and text-instruction datasets.

Expected dataset format: .npz with at least:

    images      : (N, H, W, 3)
    states      : (N, state_dim)
    actions     : (N, action_dim)
    text_ids    : (N, T_text)
    vocab       : dict

For WAM next-state prediction, dataset should also include:

    next_states : (N, state_dim)

Optional future extension:

    next_images : (N, H, W, 3)
"""

import argparse
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.wam_policy import MiniWAMPolicy


class TrainingDataset(Dataset):
    def __init__(self, path: str, resize_to: int = 64):
        data = np.load(path, allow_pickle=True)

        self.images = data["images"]          # (N, H, W, 3)
        self.states = data["states"]          # (N, state_dim)
        self.actions = data["actions"]        # (N, action_dim)
        self.text_ids = data["text_ids"]      # (N, T_text)

        self.next_states = data["next_states"] if "next_states" in data else None
        self.next_images = data["next_images"] if "next_images" in data else None

        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
        self.resize_to = resize_to

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

    def __len__(self):
        return self.images.shape[0]

    def _process_image(self, img):
        if self.cv2 is not None and (
            img.shape[0] != self.resize_to or img.shape[1] != self.resize_to
        ):
            img = self.cv2.resize(img, (self.resize_to, self.resize_to))

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    def __getitem__(self, idx):
        batch = {
            "image": self._process_image(self.images[idx]),
            "state": torch.from_numpy(self.states[idx]).float(),
            "action": torch.from_numpy(self.actions[idx]).float(),
            "text_ids": torch.from_numpy(self.text_ids[idx]).long(),
        }

        if self.next_states is not None:
            batch["next_state"] = torch.from_numpy(self.next_states[idx]).float()

        if self.next_images is not None:
            batch["next_image"] = self._process_image(self.next_images[idx])

        return batch


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset / environment identity flags.
    # These do not directly construct envs in train.py yet, but they make
    # checkpoints and CLI usage consistent with collect_data.py / inference.py.
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

    parser.add_argument("--predict_state", action="store_true")
    parser.add_argument("--predict_image", action="store_true")

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )

    # Training/data flags.
    parser.add_argument("--dataset-path", type=str, default="data/dataset.npz")
    parser.add_argument("--resize-to", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Model flags.
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--diffusion-T", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=8)

    # Loss weights.
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--image-weight", type=float, default=1.0)

    parser.add_argument("--save-path", type=str, default="checkpoints/model.pt")

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_total_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    args,
):
    """Compute VLA/WAM training loss.

    VLA:
        loss = action loss

    WAM:
        loss = action loss + optional next-state loss + optional next-image loss
    """

    logs = {}

    if args.action_head == "mlp":
        if "action" not in outputs:
            raise KeyError("Model output must contain 'action' for action_head='mlp'.")

        action_loss = F.mse_loss(outputs["action"], batch["action"])

    elif args.action_head == "diffusion":
        # Your MiniWAMPolicy should expose diffusion training loss as either:
        # outputs["action_loss"] or outputs["loss"].
        if "action_loss" in outputs:
            action_loss = outputs["action_loss"]
        elif "loss" in outputs:
            action_loss = outputs["loss"]
        else:
            raise KeyError(
                "Diffusion model output must contain 'action_loss' or 'loss'."
            )

    elif args.action_head == "flow":
        # Future extension: your FlowMatchingHead should return a training loss.
        if "action_loss" in outputs:
            action_loss = outputs["action_loss"]
        elif "flow_loss" in outputs:
            action_loss = outputs["flow_loss"]
        elif "loss" in outputs:
            action_loss = outputs["loss"]
        else:
            raise KeyError(
                "Flow model output must contain 'action_loss', 'flow_loss', or 'loss'."
            )

    else:
        raise ValueError(f"Unknown action_head: {args.action_head}")

    logs["action_loss"] = action_loss.detach()
    total_loss = args.action_weight * action_loss

    if args.model_type == "wam" and args.predict_state:
        if "next_state" not in batch:
            raise KeyError(
                "You passed --predict_state, but dataset has no 'next_states' array. "
                "Update collect_data.py to save next_states."
            )

        if "next_state" not in outputs:
            raise KeyError(
                "You passed --predict_state, but model output has no 'next_state'."
            )

        state_loss = F.mse_loss(outputs["next_state"], batch["next_state"])
        logs["state_loss"] = state_loss.detach()
        total_loss = total_loss + args.state_weight * state_loss

    if args.model_type == "wam" and args.predict_image:
        if "next_image" not in batch:
            raise KeyError(
                "You passed --predict_image, but dataset has no 'next_images' array."
            )

        if "next_image" not in outputs:
            raise KeyError(
                "You passed --predict_image, but model output has no 'next_image'."
            )

        image_loss = F.mse_loss(outputs["next_image"], batch["next_image"])
        logs["image_loss"] = image_loss.detach()
        total_loss = total_loss + args.image_weight * image_loss

    logs["total_loss"] = total_loss.detach()

    return total_loss, logs


def forward_model(model, batch, args):
    """Call model in a batch-driven way.

    This keeps the train loop conceptually clean:

        outputs = forward_model(model, batch, args)
        loss, logs = compute_total_loss(outputs, batch, args)

    If you later update MiniWAMPolicy.forward(batch), you can replace this helper
    with simply:

        outputs = model(batch)
    """

    # Current MiniWAMPolicy signature:
    #   model(image, text_tokens, state, actions=None)
    return model(
        image=batch["image"],
        text_tokens=batch["text_ids"],
        state=batch["state"],
        actions=batch["action"],
    )


def validate_args_and_dataset(args, dataset):
    if args.model_type == "vla":
        if args.predict_state:
            print("[warning] --model_type vla ignores --predict_state.")
        if args.predict_image:
            print("[warning] --model_type vla ignores --predict_image.")

        args.predict_state = False
        args.predict_image = False

    if args.model_type == "wam" and args.predict_state and dataset.next_states is None:
        raise ValueError(
            "WAM next-state training requires 'next_states' in the dataset. "
            "Update scripts/collect_data.py to save next_states."
        )

    if args.model_type == "wam" and args.predict_image and dataset.next_images is None:
        raise ValueError(
            "WAM next-image training requires 'next_images' in the dataset. "
            "For v0.1, run without --predict_image."
        )


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    dataset = TrainingDataset(args.dataset_path, resize_to=args.resize_to)
    validate_args_and_dataset(args, dataset)

    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    model = MiniWAMPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        diffusion_T=args.diffusion_T,
        action_head=args.action_head,
        predict_state=(args.model_type == "wam" and args.predict_state),
        flow_steps=args.flow_steps,
    ).to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()

        running_logs = {
            "total_loss": 0.0,
            "action_loss": 0.0,
            "state_loss": 0.0,
            "image_loss": 0.0,
        }

        num_samples = 0

        for batch in loader:
            batch = move_batch_to_device(batch, device)
            batch_size = batch["image"].size(0)

            outputs = forward_model(model, batch, args)

            loss, logs = compute_total_loss(
                outputs=outputs,
                batch=batch,
                args=args,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_samples += batch_size

            for key in running_logs:
                if key in logs:
                    running_logs[key] += logs[key].item() * batch_size

        avg_total = running_logs["total_loss"] / num_samples
        avg_action = running_logs["action_loss"] / num_samples
        avg_state = running_logs["state_loss"] / num_samples
        avg_image = running_logs["image_loss"] / num_samples

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={avg_total:.4f} "
            f"action={avg_action:.4f} "
            f"state={avg_state:.4f} "
            f"image={avg_image:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": dataset.vocab,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": args.d_model,
            "diffusion_T": args.diffusion_T,
            "flow_steps": args.flow_steps,
            "model_type": args.model_type,
            "action_head": args.action_head,
            "predict_state": args.predict_state,
            "predict_image": args.predict_image,
            "env": args.env,
            "task": args.task,
            "resize_to": args.resize_to,
        },
        args.save_path,
    )

    print("Saved checkpoint:", args.save_path)


if __name__ == "__main__":
    main()