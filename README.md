# mini-WAM

(use with caution. this repo is still a work in progress)

A tiny World Action Model from scratch for robot learning.

mini-WAM extends the educational philosophy of [mini-VLA](https://github.com/keivalya/mini-vla).

Where mini-VLA learns:

```text
image_t + language + state_t → action_t
```

mini-WAM learns:

```text
image_t + language + state_t → action_t + state_t+1 + image_t+1
```

## Why mini-WAM?

A VLA learns to act.
A WAM learns to act while predicting what its action will do to the world.

And because that prediction happens in a latent space you can decode, you
can *look inside*: watch the model imagine the future, ask it "what if I
pushed left instead?", and check which latent directions mean "where is
the agent" or "how fast is it moving".

## Read the paper

**[mini-WAM: Building a World Action Model from Scratch](docs/mini-wam-paper.md)** walks through the whole model for people coming from VLAs, with a small three-seed study on the toy world. Main findings:

* The latent world model imagines 15 steps ahead with ~80× less state error than "nothing changes", and every imagined latent is linearly readable.
* Trained on expert demos only, it predicts what happened *better* but understands other actions **15× worse**. A few random-action bursts in the data fix it.
* With the state as input, world modeling doesn't help the policy. From **pixels alone** it more than doubles closed-loop success (80% vs. 36%), because imagination recovers the velocity a single frame can't show.

## The model in one picture

```text
image_t ──► ImageEncoder ──┐
text    ──► TextEncoder  ──┼─► Fusion ──► z_t   (latent world state)
state_t ──► StateEncoder ──┘               │
                                           ├─► ActionHead    ──► a_t           act
                                           ├─► StateReadout  ──► s_t           (linear probe)
                                           ├─► ImageDecoder  ──► image_t
                                           │
                   a_t ──► LatentDynamics(z_t, a_t) ──► z_t+1   imagine
                                           ├─► StateReadout  ──► s_t+1
                                           ├─► ImageDecoder  ──► image_t+1
                                           └─► ActionHead    ──► a_t+1 ...     dream
```

Everything lives on one latent `z`. The policy acts from `z`, the dynamics
moves `z` forward given an action, and the readouts turn any `z` (real or
imagined) back into a state and an image. So the WAM can roll its own
policy forward entirely inside its head, then show you what it expects to see.

| file | what it does |
|---|---|
| `models/encoders.py` | image (TinyCNN or spatial-softmax keypoints), text (GRU), state (MLP) encoders |
| `models/fusion.py` | fuses the three tokens into the latent world state `z_t` |
| `models/action_heads.py` | `mlp` / `diffusion` / `flow` policy heads behind one interface |
| `models/dynamics_head.py` | `LatentDynamicsMLP` (z, a → z′) and `StateReadout` (z → s) |
| `models/image_decoder.py` | z → 64×64 image |
| `models/losses.py` | action + state + image + latent-consistency losses |
| `models/wam_policy.py` | wires everything together: `forward`, `act`, `imagine` |

### Training objective

For a window of `H` steps inside one episode (`--horizon`, default 5):

```text
z_t       = encode(o_t)
ẑ_t+k     = dynamics(ẑ_t+k-1, a_t+k-1)               k = 1..H, using the actions really taken

L = L_action(a_t)                                     imitate the expert (expert steps only)
  + w_s · ( |readout(z_t) − s_t|² + |readout(ẑ_t+k) − s_t+k|² )
  + w_i · ( |decode(z_t) − img_t|² + |decode(ẑ_t+k) − img_t+k|² )
  + w_z · |ẑ_t+k − stopgrad(encode(o_t+k))|²          keep imagination "in distribution"
```

> [!IMPORTANT]
> **A world model trained only on expert demos learns to ignore actions.**
> In expert data, `a_t` is a function of `s_t`, so `s_t+1` can be predicted
> from `s_t` alone. The model then imagines the expert's future no matter
> which action you ask about. You can see this happen in the "what if" plot.
> `collect_data.py` therefore mixes in short bursts of random actions
> (`--random-action-prob`, default 0.3). The world model learns from every
> transition; the policy imitates only the expert ones (`is_expert`).

## Quickstart

```sh
pip install -r requirements.txt

# 1. collect demos in the toy world (numpy only, ~1 second)
python -m scripts.collect_data --env toy --episodes 400 --max-steps 80 \
  --output-path data/toy.npz

# 2. train the WAM (CPU is fine)
python -m scripts.train --dataset-path data/toy.npz \
  --model_type wam --action_head mlp \
  --predict_state --predict_image \
  --epochs 12 --device cpu --save-path checkpoints/wam.pt

# 3. how good is its imagination on held-out episodes?
python -m scripts.evaluate --checkpoint checkpoints/wam.pt

# 4. look inside its head
python -m scripts.visualize_rollout --checkpoint checkpoints/wam.pt --video

# 5. run it in closed loop, with a "mind's eye" video
python -m scripts.test --checkpoint checkpoints/wam.pt --episodes 20 --save-video
```

For the mini-VLA baseline, train the same thing with `--model_type vla`.

> [!TIP]
> **Meta-World.** Use `--env metaworld --env-name push-v3 --camera-name corner`
> in `collect_data.py` (images are resized to 64×64 at collection time) and
> `--env metaworld` in `test.py`. Everything else is the same.

## The toy world

`envs/toy_point_env.py` is a 2D point mass with momentum and two colored
goals. The instruction (`"go to the blue goal"` / `"go to the green goal"`)
says which goal counts, so the policy has to read the language. Velocity is
not visible in a single image, so the world model has to infer it to
predict the next frame. The true state is just 6 numbers, which makes it easy
to check whether the latent really encodes them.

## Looking inside the world model

`scripts/visualize_rollout.py` writes these figures for held-out episodes:

| figure | question it answers |
|---|---|
| `1_imagination.png` | real frames vs. reconstructed, imagined (open-loop, real actions) and dreamed (own policy) frames |
| `2_beliefs.png` | for each state variable: the truth, what the latent says now, and what it imagined from t=0 |
| `3_latent_map.png` | PCA map of the latent space with real vs. imagined trajectories, plus which latent directions correlate with which physical quantities |
| `4_what_if.png` | counterfactuals: same start, 9 different pushes. Where does the WAM think the agent goes? (toy: compared against the true simulator) |
| `5_horizon.png` | imagination error vs. number of imagined steps, against a "nothing changes" baseline; readout R² per variable |
| `6_keypoints.png` | what the spatial-softmax image encoder is tracking |
| `dream.mp4` | real vs. imagined vs. dreamed video |

`scripts/evaluate.py` prints the same numbers and saves them as JSON.

### Why is the latent interpretable?

* `StateReadout` is **linear** by default. If a linear map recovers agent
  position and velocity from `z`, then those quantities are explicitly
  encoded in the latent, and the readout weights say which directions mean what.
  (`--mlp_readout` swaps in an MLP if you want a stronger, less interpretable decoder.)
* The **readout is shared** between real and imagined latents, so imagined
  latents can be read with the same "dictionary" as real ones.
* The **image encoder** (`--image_encoder keypoints`) uses a spatial softmax:
  each channel becomes an (x, y) location you can draw on the image.

> [!NOTE]
> `state_t` is also an input, so reading `s_t` back out of the real `z_t` is
> easy. The interesting readouts are of *imagined* latents (k ≥ 1), which have
> never seen the future state.

## Status

| piece | status |
|---|---|
| image + language + state → action (mlp / diffusion / flow) | ✅ |
| → state_t+1 (action-conditioned latent dynamics + linear readout) | ✅ |
| → image_t+1 (latent → image decoder) | ✅ |
| multi-step imagination & dreaming | ✅ |
| toy env, Meta-World MT1 | ✅ |
| `envs/mujoco_env.py`, `envs/wrappers.py`, `configs/`, `scripts/inference.py` | 🚧 not yet |
