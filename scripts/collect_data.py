"""Collect WAM-compatible demonstration data from Meta-World MT1 environments.

mini-VLA data:
    observation_t -> action_t

mini-WAM data:
    observation_t -> action_t + observation_t+1

Saved .npz keys:
    images       : (N, H, W, 3)
    states       : (N, state_dim)
    actions      : (N, action_dim)
    next_images  : (N, H, W, 3)
    next_states  : (N, state_dim)
    text_ids     : (N, T_text)
    texts        : (N,)
    vocab        : dict
"""

import os
import argparse
import time

import numpy as np
import gymnasium as gym
from metaworld.policies import ENV_POLICY_MAP

from utils.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", type=str, default="push-v3")

    parser.add_argument(
        "--camera-name",
        type=str,
        default="topview",
        help=(
            "Meta-World camera: corner, corner2, corner3, corner4, "
            "topview, behindGripper, gripperPOV"
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=150)

    parser.add_argument(
        "--output-path",
        type=str,
        default="data/dataset.npz",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between steps for visualization.",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default="push the object to the goal",
        help="Fixed instruction for all episodes.",
    )

    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="End each episode early once Meta-World reports success.",
    )

    return parser.parse_args()


def extract_state(obs):
    """Meta-World observations are flat numpy arrays."""
    return np.asarray(obs, dtype=np.float32).ravel()


def render_image(env):
    """Render RGB image from the current environment state."""
    img = env.render()

    if img is None:
        raise RuntimeError(
            "env.render() returned None. Make sure render_mode='rgb_array' "
            "is passed when creating the Meta-World environment."
        )

    return np.asarray(img, dtype=np.uint8)


def tokenize_texts(texts):
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(texts)

    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in text_ids_list)

    text_ids = np.zeros((len(texts), max_len), dtype=np.int64)

    for i, seq in enumerate(text_ids_list):
        text_ids[i, : len(seq)] = np.asarray(seq, dtype=np.int64)

    return text_ids, tokenizer.vocab


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    env = gym.make(
        "Meta-World/MT1",
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )

    policy = ENV_POLICY_MAP[args.env_name]()

    images = []
    states = []
    actions = []
    next_images = []
    next_states = []
    texts = []

    instruction = args.instruction

    total_successes = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)

        done = False
        steps = 0
        episode_success = 0

        while not done and steps < args.max_steps:
            # Current observation o_t
            image_t = render_image(env)
            state_t = extract_state(obs)

            # Expert action a_t
            action_t = policy.get_action(obs)
            action_t = np.asarray(action_t, dtype=np.float32)

            # Step to next observation o_{t+1}
            next_obs, reward, terminated, truncated, info = env.step(action_t)

            image_tp1 = render_image(env)
            state_tp1 = extract_state(next_obs)

            success = int(info.get("success", 0))
            episode_success = max(episode_success, success)

            # Store WAM transition:
            # observation_t -> action_t + observation_t+1
            images.append(image_t.copy())
            states.append(state_t.copy())
            actions.append(action_t.copy())
            next_images.append(image_tp1.copy())
            next_states.append(state_tp1.copy())
            texts.append(instruction)

            obs = next_obs
            steps += 1

            done = bool(terminated or truncated)

            if args.stop_on_success and success == 1:
                done = True

            if args.sleep > 0:
                time.sleep(args.sleep)

        total_successes += episode_success

        print(
            f"Episode {ep + 1}/{args.episodes} "
            f"finished after {steps} steps, "
            f"success={episode_success}"
        )

    env.close()

    images = np.stack(images, axis=0)
    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    next_images = np.stack(next_images, axis=0)
    next_states = np.stack(next_states, axis=0)
    texts = np.asarray(texts)

    text_ids, vocab = tokenize_texts(texts)

    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        next_images=next_images,
        next_states=next_states,
        text_ids=text_ids,
        texts=texts,
        vocab=vocab,
    )

    print()
    print("Saved WAM-compatible Meta-World dataset to:", args.output_path)
    print("  images:      ", images.shape)
    print("  states:      ", states.shape)
    print("  actions:     ", actions.shape)
    print("  next_images: ", next_images.shape)
    print("  next_states: ", next_states.shape)
    print("  text_ids:    ", text_ids.shape)
    print("  vocab size:  ", len(vocab))
    print("  successes:   ", f"{total_successes}/{args.episodes}")


if __name__ == "__main__":
    main()