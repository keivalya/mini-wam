"""Fresh privileged-state sanity checks; not a re-evaluation of archived neural models."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.toy_point_env import ToyPointEnv


def collect(seed, episodes, steps, noise, random_prob):
    env = ToyPointEnv(seed=seed)
    rng = np.random.default_rng(seed + 1_000_000)
    rows = []
    for ep in range(episodes):
        _, state, _ = env.reset(seed=seed + ep)
        for _ in range(steps):
            action = rng.uniform(-1, 1, 2) if rng.random() < random_prob else env.expert_action(noise=noise)
            _, nxt, _, done, _ = env.step(action)
            rows.append((state.copy(), action.copy(), nxt.copy()))
            state = nxt
            if done:
                break
    return tuple(np.asarray([r[i] for r in rows], dtype=np.float64) for i in range(3))


def fit(data, use_action):
    state, action, nxt = data
    x = np.concatenate([state, action], axis=1) if use_action else state
    # All preprocessing is estimated on training data only. Delta targets retain
    # the environment's normalized-velocity convention (v_raw / 0.05).
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1)
    z = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    coef, _, rank, singular = np.linalg.lstsq(z, nxt - state, rcond=None)
    def predict(s, a):
        xx = np.concatenate([s, a], axis=1) if use_action else s
        zz = np.column_stack([(xx - mean) / scale, np.ones(len(xx))])
        out = s + zz @ coef
        out[:, :2] = np.clip(out[:, :2], 0, 1)
        return out
    return predict, {'rank': int(rank), 'columns': z.shape[1], 'condition_number': float(singular[0]/singular[-1]), 'feature_mean': mean.tolist(), 'feature_std': scale.tolist(), 'delta_coefficients_standardized_features': coef.tolist()}


def errors(pred, truth):
    e = pred - truth
    return {'position_mse': float(np.mean(e[..., :2]**2)),
            'velocity_raw_mse': float(np.mean((e[..., 2:4]*0.05)**2)),
            'velocity_normalized_mse': float(np.mean(e[..., 2:4]**2)),
            'target_mse': float(np.mean(e[..., 4:6]**2)),
            'state_mse_environment_units': float(np.mean(e**2))}


def checks():
    env = ToyPointEnv(seed=7)
    env.pos = np.array([0.5, 0.5]); env.vel = np.array([0.02, -0.01])
    plus, state = env.render().copy(), env._state()
    env.vel *= -1
    minus = env.render().copy()
    action = np.array([0.3, -0.7])
    pred_plus = env.simulate(state, action[None])[0]
    pred_minus = env.simulate(env._state(), action[None])[0]
    rng = np.random.default_rng(9)
    max_err = 0
    for _ in range(50):
        env.reset()
        env.pos = rng.uniform(0, 1, 2); env.vel = rng.uniform(-0.05, 0.05, 2)
        states = env._state()
        # Align actual simulator start with its float32 state representation.
        env.pos = states[:2].astype(float); env.vel = states[2:4].astype(float)*env.max_speed
        env.goals[env.target] = states[4:6].astype(float)
        actions = rng.uniform(-2, 2, (20, 2))
        simulated = env.simulate(states, actions)
        stepped = np.array([env.step(a)[1] for a in actions])
        max_err = max(max_err, float(np.max(np.abs(simulated-stepped))))
    return {'opposite_velocity_images_identical': bool(np.array_equal(plus, minus)),
            'opposite_velocity_next_position_distance': float(np.linalg.norm(pred_plus[:2]-pred_minus[:2])),
            'simulate_step_max_abs_error_50_starts_20_steps': max_err}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'docs/results/research-audit')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    start = time.time()
    conditions = {'expert_deterministic': (0., 0.), 'expert_noisy': (0.1, 0.), 'mixed_iid30': (0.1, 0.3)}
    runs = []
    for seed in range(args.seeds):
        base = 10000 + 1000*seed
        # Fresh episodes; same factual states/actions and counterfactual initial
        # states/action sequences for every training condition and model in seed.
        factual = collect(base+500, 20, 60, 0.1, 0.)
        rng = np.random.default_rng(base+700)
        idx = rng.choice(len(factual[0]), min(512, len(factual[0])), replace=False)
        initial = factual[0][idx].copy()
        interventions = rng.uniform(-1, 1, (len(idx), 10, 2))
        env = ToyPointEnv(seed=base+800)
        counter = np.asarray([env.simulate(s, a) for s, a in zip(initial, interventions)])
        for name, (noise, rp) in conditions.items():
            data = collect(base, 60, 60, noise, rp)
            for use_action in [False, True]:
                predict, fit_info = fit(data, use_action)
                predicted = initial.copy()
                rollout = []
                for h in range(10):
                    predicted = predict(predicted, interventions[:, h])
                    rollout.append(predicted.copy())
                rollout = np.stack(rollout, axis=1)
                runs.append({'seed': seed, 'training_seed_base': base, 'condition': name,
                             'model': 'affine_state_action' if use_action else 'affine_state_only',
                             'train_transitions': len(data[0]), 'test_transitions': len(factual[0]),
                             'training_action_component_saturation_fraction': float(np.mean(np.abs(data[1]) >= 1.0)),
                             'training_next_position_boundary_fraction': float(np.mean((data[2][:, :2] == 0) | (data[2][:, :2] == 1))),
                             'counterfactual_starts': len(initial), 'fit': fit_info,
                             'factual': errors(predict(factual[0], factual[1]), factual[2]),
                             'counterfactual_h1': errors(rollout[:, 0], counter[:, 0]),
                             'counterfactual_h10': errors(rollout[:, -1], counter[:, -1])})
    summary = []
    for condition in conditions:
        for model in ['affine_state_only', 'affine_state_action']:
            subset = [r for r in runs if r['condition']==condition and r['model']==model]
            out = {'condition': condition, 'model': model}
            for evaluation in ['factual', 'counterfactual_h1', 'counterfactual_h10']:
                out[evaluation] = {metric: {'mean': float(np.mean(vals := [r[evaluation][metric] for r in subset])), 'sample_sd': float(np.std(vals, ddof=1)) if len(vals)>1 else None} for metric in subset[0][evaluation]}
            summary.append(out)
    result = {'description': __doc__, 'protocol': {'seeds': args.seeds, 'training_episodes': 60, 'test_episodes': 20, 'max_episode_steps': 60, 'stop_on_success': True, 'conditions': conditions, 'mixed_random_actions': 'iid per step; deliberately not collector burst sampling', 'counterfactual': '10 iid uniform actions from heldout noisy-expert states, shared across conditions', 'fit': 'unregularized least-squares affine delta; train-only feature standardization; clip predicted positions', 'units': 'position and target: unit-square coordinates; velocity normalized by 0.05 in state; velocity_raw_mse multiplies normalized error by 0.05 before squaring', 'uncertainty': 'mean and sample standard deviation across fresh collection seeds; not confidence intervals; no neural model results'},
              'provenance': {'python': platform.python_version(), 'numpy': np.__version__, 'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'environment_sha256': hashlib.sha256((ROOT/'envs/toy_point_env.py').read_bytes()).hexdigest(), 'elapsed_seconds': time.time()-start}, 'checks': checks(), 'summary': summary, 'runs': runs}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'checks': result['checks'], 'summary': summary}, indent=2))

if __name__ == '__main__':
    main()
