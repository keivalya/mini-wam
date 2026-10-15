# Fresh dynamics audit

This is a new, deliberately small privileged-state experiment. It is not a rerun of the archived VLA/WAM checkpoints and cannot establish a neural architecture comparison. No neural training was performed. The experiment tests whether the toy system needs a complex learned transition model and whether observational error measures action dependence.

Run from the repository root with Python 3 and NumPy:

```sh
python3 scripts/research_audit.py --seeds 5 --output docs/results/research-audit
```

The committed `metrics.json` contains all 30 fitted models, per-seed errors, training-only standardization parameters, coefficients, design ranks, sample counts, seed definitions, source hashes, and runtime versions. `console.json` is the compact console output from the same run. The run took about three seconds on this machine; timing is not a benchmark.

## Protocol

Five independent collection seed blocks (10000, 11000, 12000, 13000, 14000). Each condition collects 60 episodes, stops at success or 60 steps, and fits the same two ordinary least-squares models. Conditions are deterministic expert, expert plus Gaussian action noise with standard deviation 0.1, and noisy expert with probability 0.3 of an independent uniform random action at each step. This last condition uses independent action mixing, not the main collector's burst sampler. Episode counts match; transition counts do not, because termination times vary. This is a sanity check, not a controlled sample-efficiency comparison.

Models predict the six-dimensional state increment from either state alone or state plus action and an intercept. Each feature is standardized using training data only; outputs are not standardized. All models receive true target position and velocity. Predicted positions are clipped to the unit square at every step; this is an explicit known-dynamics prior. No other simulator coefficients are supplied to the fitted models.

Each seed has 20 separate held-out noisy-expert episodes. Factual prediction uses their actual actions. Counterfactual evaluation selects up to 512 of those same held-out states without replacement and applies ten independent uniform actions. All training conditions and both models share exactly the same evaluation states and actions within a seed. Counterfactual rollouts feed predicted states back recursively. `h10` is terminal error at step 10, not mean error over ten steps. Samples from one episode are correlated; uncertainty is the sample standard deviation across five collection seeds, not across transitions and not a confidence interval.

The state layout is `[x,y,vx/0.05,vy/0.05,target_x,target_y]`. Position MSE is the mean squared error across two unit-square coordinates. Raw velocity MSE converts normalized velocity errors back to displacement per step before squaring. The combined six-state MSE mixes position and normalized-velocity scales and includes two essentially static target coordinates; it must not be mistaken for an interpretable physical error or compared to separately standardized archived neural metrics.

## Observed position MSE (mean ± sample SD across five seeds)

| Training data | Model | Factual 1-step | Random-action 1-step | Random-action terminal step 10 |
|---|---|---:|---:|---:|
| Deterministic expert | State only | 2.045e-6 ± 6.154e-8 | 4.677e-5 ± 1.929e-6 | 9.871e-3 ± 8.430e-4 |
| Deterministic expert | State + action | 3.483e-16 ± 2.791e-17 | 1.789e-16 ± 1.698e-17 | 4.269e-16 ± 1.589e-16 |
| Noisy expert | State only | 2.047e-6 ± 6.185e-8 | 4.690e-5 ± 1.943e-6 | 9.813e-3 ± 8.342e-4 |
| Noisy expert | State + action | 3.487e-16 ± 2.783e-17 | 1.837e-16 ± 1.844e-17 | 5.357e-16 ± 3.023e-16 |
| Mixed, 30% random | State only | 2.927e-6 ± 3.072e-7 | 4.139e-5 ± 1.897e-6 | 7.780e-3 ± 7.582e-4 |
| Mixed, 30% random | State + action | 3.479e-16 ± 2.785e-17 | 1.758e-16 ± 1.832e-17 | 2.403e-16 ± 3.649e-17 |

The affine action model reaches numerical precision even on deterministic expert data. Its standardized design has rank 9/9 in every seed, with condition numbers 13.5–16.2 in that condition. The expert is clipped: about 8% of action components saturate in the first deterministic seed. Therefore actions are not a single affine function of state over the collected data. This gives the restricted affine model enough information to identify dynamics. It contradicts an unconditional statement that deterministic expert data necessarily prevents learning action dependence. The result does not establish identification for unrestricted nonlinear models: many functions could agree on the expert manifold and differ off it.

The action-free model's small factual position error understates its random-action rollout error. However, this is a diagnostic example, not evidence that any particular neural WAM ignores action. The nearly exact affine result also shows that this environment is too simple to substantiate the necessity of a sophisticated dynamics architecture when true state is provided.

## Environment checks

Two states with the same positions and goals but velocities `[0.02,-0.01]` and `[-0.02,0.01]` render bit-identical images. Under the same action `[0.3,-0.7]`, their next positions differ by Euclidean distance 0.0357771. A single image cannot uniquely determine velocity or the next physical state; history, a belief state, or uncertainty is required. This is a constructive counterexample, not an empirical claim that every observed image has equally likely velocities.

`simulate` and `step` agree exactly in returned float32 states over 50 random initial states and 20 actions each, including actions outside the allowed interval to check clipping. The starting environment state is aligned to the float32 state passed to `simulate`, avoiding a precision mismatch. These checks cover sampled cases, not every possible state.

## What remains untested

Image reconstruction, instruction understanding, latent representation quality, policy success, neural counterfactual calibration, long-horizon uncertainty, unseen dynamics, and robotics transfer. Fresh image-history baselines and matched neural ablations are needed. Do not present this audit as evidence that the proposed WAM beats a VLA or as a publication-level contribution by itself. Treat it as a lower bound on the validation required before making those claims.
