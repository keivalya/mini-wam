<div align="center">

# pico-WAM

</div>

pico-WAM is a compact World Action Model (WAM): a 0.75M-parameter model for studying how a robot policy and a world model interact. One shared latent state feeds a policy, an action-conditioned transition model, and state and image readouts, so the same model both acts and predicts how the world will change.

> [!IMPORTANT]
> pico-WAM is a research and teaching implementation, not a production system. A VLA learns `image_t + language + state_t → action_t`; pico-WAM learns `image_t + language + state_t → action_t + state_t+1 + image_t+1`. (Inspired by, and extending, [mini-VLA](https://github.com/keivalya/mini-vla).)

<div align="center">
  <img src="./assets/toy_dream.gif" alt="real episode, predicted future, and policy-driven rollout side by side">
  <br>
  <em>Left: the real episode. Middle: pico-WAM's open-loop prediction under the real actions. Right: its rollout when its own policy chooses the actions.</em>
</div>

The codebase is intentionally small (the whole model package is under 1,000 lines) so that,
- beginners coming from VLAs can follow the complete WAM training and prediction pipeline
- researchers can quickly prototype ideas around latent world models
- students can *look inside* a world model: decode its predictions, ask "what if I pushed left?", and check them against the true simulator

> [!TIP]
> Start with the paper. It describes the design and every experiment in this repository.
> - **pico-WAM: Design and Evaluation of a Compact World Action Model** ([PDF](docs/paper/pico-wam-paper.pdf), [LaTeX source](docs/paper/pico-wam-paper.tex))
> - [Research plan](docs/paper/RESEARCH_PLAN.md): open questions, confounds, and next experiments
> - [Hands-on tutorial](docs/pico-wam-paper.md): an earlier, longer walkthrough (the PDF is authoritative where they differ)
> - Blog: From VLAs to WAMs (Math, Code, and Intuition), **TBD**

This project is not meant to be state-of-the-art. The architecture uses established components; the contribution is a clear, hackable implementation and a controlled study of how its parts behave.

## What the paper finds

All experiments use a 2D point-mass navigation task with known dynamics, so every prediction, including predictions under actions the expert never took, can be checked against the true simulator. Main comparisons use three independent runs and the supervision pilot uses five; ± is the standard deviation across runs.

- **State prediction works.** Fifteen steps ahead, normalized state error is 0.0071 ± 0.0004, about 80× lower than persistence (0.5824).
- **Image prediction does not, yet.** Decoded frames are worse than copying the current frame at every horizon. In the paper's example episode, the decoder keeps the agent and the selected goal but loses the distractor.
- **Data coverage matters for "what if" questions.** Expert-only training gives lower factual error but about 15× higher error under random alternative actions (0.112 vs. 0.0072 at 10 steps) than training with random-action bursts. An affine model with the known clipping rule still recovers the simulator from expert data alone, so the gap comes from neural model fitting, not from missing information in the demonstrations.
- **Auxiliary supervision helps the pixel-input policy.** With image inputs, success rises from 36.0 ± 19.8% for the VLA baseline to 80.3 ± 12.0% for the WAM, but the WAM also receives state targets that the VLA does not. A matched five-run pilot gives 17.0% for behavioral cloning alone, 28.4% with current-state supervision, and 33.6% with added future-state prediction, so current-state supervision contributes more of the gain (11.4 points) than future-state prediction (5.2 points).
- **With state inputs, both policies are strong.** On 400 episodes, WAM and VLA both reach 100% success. On 25 episodes the WAM trails (87.0% vs. 99.7%) until longer training closes most of the gap (96.3% vs. 98.7%).
- **Image decoding dominates cost.** The decoder holds about half of the 747,739 parameters. Removing it cuts a CPU optimizer step from 2.55 s to 0.27 s (about 90%). Policy inference takes about 0.42 ms for both WAM and VLA, since neither calls the dynamics at deployment.

The paper also places pico-WAM's design choices alongside eight open-source systems (GR-1, UWM, WorldVLA, Genie Envisioner, Cosmos Policy, V-JEPA 2-AC, DreamGen and LingBot-VA) and lists its limitations: the dynamics are affine away from the walls, the policy does not plan with its predictions, and a history-based baseline is still missing.

## How pico-WAM is built

pico-WAM is a VLA-style policy with a world model trained alongside it. The policy and the transition model share one latent state `z_t`; at deployment, only the policy is used.

- [models/encoders.py](models/encoders.py) encodes the 64×64 image (a three-layer CNN with 16 spatial-softmax keypoints, or a plain pooled CNN), the instruction (a GRU) and the six-dimensional state (an MLP)
- [models/fusion.py](models/fusion.py) fuses them with an MLP and layer norm into a 128-dimensional latent `z_t`
- [models/action_heads.py](models/action_heads.py) turns `z_t` into an action with an MLP head; [diffusion](models/diffusion_head.py) and [flow-matching](models/flow_matching_head.py) heads are also included (the paper's control experiments use the MLP head)
- [models/dynamics_head.py](models/dynamics_head.py) holds the transition model, a residual MLP `z_t+1 = LN(z_t + MLP[z_t, a_t])`, and a *linear* `StateReadout` that maps any latent, encoded or predicted, to the physical state
- [models/image_decoder.py](models/image_decoder.py) decodes any latent into an image, predicted as a residual on the training-set mean image
- [models/losses.py](models/losses.py) combines the action, state, image and latent-consistency losses
- [models/wam_policy.py](models/wam_policy.py) puts everything together, including `imagine()` for multi-step rollouts in latent space

Training uses windows of `H = 5` transitions and minimizes `L_act + 0.5 L_state + L_image + L_latent`. The action loss uses only expert-labeled steps; the prediction losses use every transition, including random-action bursts. Dropping the auxiliary losses gives the behavioral-cloning baseline, labeled VLA in the paper.

Scripts:
- [scripts/collect_data.py](scripts/collect_data.py) collects demonstrations with an expert controller, plus short random-action bursts so the world model sees actions the expert would never take
- [scripts/train.py](scripts/train.py) trains the WAM, or the VLA baseline with `--model_type vla`
- [scripts/evaluate.py](scripts/evaluate.py) measures prediction on held-out episodes, both factual (recorded actions) and counterfactual (random actions checked against the simulator)
- [scripts/visualize_rollout.py](scripts/visualize_rollout.py) shows what the model predicts: rollout strips, state beliefs, latent maps, what-if plots and keypoints
- [scripts/test.py](scripts/test.py) runs the policy in closed loop and saves videos of the real frame next to the decoded present and future
- [scripts/run_toy.sh](scripts/run_toy.sh) runs the full pipeline over three independent runs
- [scripts/supervision_ablation.py](scripts/supervision_ablation.py) runs the matched-supervision pilot
- [scripts/research_audit.py](scripts/research_audit.py) runs the affine dynamics diagnostic

# Video

**TBD**: a video walkthrough of pico-WAM is coming soon.

## Getting started

Create (or activate) a conda environment

```
conda create --name pico-wam python=3.11
conda activate pico-wam
```

Clone pico-WAM project

```
git clone https://github.com/keivalya/pico-wam.git
cd pico-wam
```

Install dependencies

```
pip install -r requirements.txt
```

## Collect demonstration data

This gathers trajectories with an expert controller in the 2D navigation task (numpy only, seconds on a laptop) and saves them in a `.npz` dataset. When no random burst is active, the collector starts one with probability 0.3 (`--random-action-prob`); each burst lasts one to three steps. These random actions train the world model but are excluded from action regression.

```
python -m scripts.collect_data \
  --env toy \
  --episodes 400 \
  --max-steps 80 \
  --output-path data/toy.npz
```

> [!TIP]
> For Meta-World, use `--env metaworld --env-name push-v3 --camera-name corner` (images are resized to 64×64 when collecting).

## Train your WAM model

Train the policy and the world model together on your collected dataset.

```
python -m scripts.train \
  --dataset-path data/toy.npz \
  --model_type wam \
  --action_head mlp \
  --predict_state \
  --predict_image \
  --epochs 15 \
  --save-path checkpoints/wam.pt \
  --device cpu
```

Train the VLA baseline (same encoders, no world model) on the same data with `--model_type vla` (and drop `--predict_state --predict_image`). Add `--no_state_input` to either model for the pixel-input setting: the policy then sees only the image and instruction, and the WAM still uses state as a training target.

## Test your model in sim

Run the trained WAM on unseen episodes. With `--save-video`, every frame shows the real observation next to what the model believes now and what it expects a few steps ahead.

```
python -m scripts.test \
  --checkpoint checkpoints/wam.pt \
  --episodes 20 \
  --device cpu \
  --save-video \
  --video-dir videos
```

## Look inside the world model

Measure how well it predicts held-out futures (and "what if" futures under random actions), then visualize what it believes.

```
python -m scripts.evaluate --checkpoint checkpoints/wam.pt
python -m scripts.visualize_rollout --checkpoint checkpoints/wam.pt --video
```

## Reproducing the paper

Stored results for every experiment are in [docs/results/](docs/results/), and the figures can be regenerated from them without retraining. Original training datasets and checkpoints are not included.

| Paper result | Script | Stored outputs |
|---|---|---|
| Prediction, data coverage, ablations and policy success | `scripts/run_toy.sh`, then `python -m scripts.summarize_runs --root results` | `docs/results/<experiment>/` |
| Matched-supervision pilot | `scripts/hpc/supervision_pilot.sbatch` (Slurm), which runs `scripts/supervision_ablation.py` | [docs/results/supervision-pilot/](docs/results/supervision-pilot/README.md) |
| Affine dynamics diagnostic | `python3 scripts/research_audit.py --seeds 5 --output docs/results/research-audit` | [docs/results/research-audit/](docs/results/research-audit/README.md) |
| Parameter and runtime benchmark | `python docs/figures/benchmark_efficiency.py --run-dir <dir with wam.pt, vla.pt, toy.npz>` | `docs/results/efficiency/` |
| Figures | `python docs/figures/make_figures.py` | `docs/figures/` |

Each experiment in the main study changes one setting of `scripts/run_toy.sh` through an environment variable, for example:

```sh
RANDOM_ACTION_PROB=0.0 bash scripts/run_toy.sh   # expert-only data
STATE_INPUT=0 bash scripts/run_toy.sh            # pixel-input policies
HORIZON=10 bash scripts/run_toy.sh               # longer training windows
LATENT_WEIGHT=0 bash scripts/run_toy.sh          # no latent consistency
```

> [!WARNING]
> `scripts/run_toy.sh` **deletes the root `results/` directory when it starts**, so run it only in an isolated checkout.

Check the pilot's stored outputs (paired evaluation seeds, equal update counts, complete conditions) with:

```sh
python3 scripts/summarize_supervision.py --root docs/results/supervision-pilot
```

Build the paper PDF with Tectonic, as described in [docs/paper/BUILD_AND_VALIDATION.md](docs/paper/BUILD_AND_VALIDATION.md):

```sh
tectonic docs/paper/pico-wam-paper.tex
```

## Next steps

Planning to,
- plan with the world model: compare candidate action sequences in imagination and pick the best (MPC), and test whether better prediction leads to better decisions
- add a history-based baseline, since a single frame cannot reveal velocity
- rerun the supervision comparison with more runs and a fixed, confirmatory protocol
- run the study on Meta-World manipulation tasks, **TBD**
- a standalone inference script with arbitrary text input, **TBD**
- blogs, tutorials and a video walkthrough, **TBD**
