"""Test a mini-WAM / mini-VLA policy in closed loop (toy env or Meta-World MT1).

With --save-video, every frame of a WAM rollout shows what the model sees
and what it is thinking:

    | real observation | decode(z_t): what it believes now | decode(z_t+k): what it expects k steps ahead |

so you can watch the world model's predictions succeed (or fail) live.

Usage:
    python -m scripts.test --checkpoint checkpoints/wam.pt --episodes 20 --save-video
"""

import os
import argparse

import numpy as np
import torch

from models.wam_policy import MiniWAMPolicy
from utils.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test a mini-WAM policy in closed loop")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--env", type=str, default=None, choices=["toy", "metaworld"],
                        help="Defaults to the env the checkpoint was trained on.")
    parser.add_argument("--env-name", type=str, default="push-v3", help="Meta-World task")
    parser.add_argument("--camera-name", type=str, default="topview")
    parser.add_argument("--seed", type=int, default=10_000, help="Use seeds not seen in training.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Meta-World: language instruction (toy env picks one per episode).",
    )
    parser.add_argument("--lookahead", type=int, default=5, help="Imagination steps shown in the video.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos")

    return parser.parse_args()


def make_env(args, env_name, image_size):
    if env_name == "toy":
        from envs.toy_point_env import ToyPointEnv
        return ToyPointEnv(seed=args.seed, image_size=image_size)

    from envs.metaworld_env import MetaWorldMT1Wrapper
    return MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )


def to_tensor(img, size, device):
    if img.shape[0] != size:
        import cv2
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)


def to_np_img(x, size=192):
    import cv2
    img = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def caption(img, text):
    import cv2
    return cv2.putText(img.copy(), text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


@torch.no_grad()
def mind_frame(model, image_t, text_ids, state_t, lookahead):
    """Real frame | what the WAM believes now | what it expects `lookahead` steps ahead."""
    panels = [caption(to_np_img(image_t[0]), "real")]
    if model.predict_image:
        dream = model.imagine(image_t, text_ids, state_t, horizon=lookahead)
        panels.append(caption(to_np_img(dream["image"][0, 0]), "believes now"))
        panels.append(caption(to_np_img(dream["image"][0, -1]), f"expects t+{lookahead}"))
    return np.concatenate(panels, axis=1)


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"[test] Loading checkpoint from {args.checkpoint}")
    model, ckpt = MiniWAMPolicy.from_checkpoint(args.checkpoint, device=device)
    tokenizer = SimpleTokenizer(vocab=ckpt["vocab"])
    size = model.config["image_size"]

    env_name = args.env or ckpt.get("env", "toy")
    env = make_env(args, env_name, size)
    print(f"[test] env={env_name} state_dim={env.state_dim} action_dim={env.action_dim} "
          f"model_type={model.config['model_type']}")

    if args.save_video:
        import imageio.v2 as imageio
        os.makedirs(args.video_dir, exist_ok=True)

    successes, lengths = 0, []

    for ep in range(args.episodes):
        img, state, info = env.reset(seed=args.seed + ep)
        instruction = info.get("instruction") or args.instruction or "push the object to the goal"
        text_ids = torch.tensor([tokenizer.encode(instruction)], dtype=torch.long, device=device)

        frames, success, step = [], 0, 0
        done = False
        while not done and step < args.max_steps:
            img_t = to_tensor(img, size, device)
            state_t = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)

            if args.save_video:
                frames.append(mind_frame(model, img_t, text_ids, state_t, args.lookahead)
                              if model.is_wam else to_np_img(img_t[0]))

            action = model.act(img_t, text_ids, state_t).squeeze(0).cpu().numpy()
            img, state, reward, done, info = env.step(action)
            success = max(success, int(info.get("success", 0)))
            step += 1

            # Meta-World doesn't stop on success by itself
            done = done or bool(success)

        successes += success
        lengths.append(step)
        print(f"[test] Episode {ep + 1}/{args.episodes}: \"{instruction}\" success={success} steps={step}")

        if args.save_video:
            path = os.path.join(args.video_dir, f"{env_name}_ep{ep + 1:03d}.mp4")
            with imageio.get_writer(path, fps=10) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"[test] Saved video to {path}")

    env.close()
    print(f"[test] Success rate: {successes}/{args.episodes} = {successes / args.episodes:.0%} "
          f"| mean steps: {np.mean(lengths):.1f}")


if __name__ == "__main__":
    main()
