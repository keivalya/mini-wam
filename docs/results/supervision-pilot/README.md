# Explorer matched-supervision pilot

Completed on October 14, 2026, on Northeastern Explorer, Slurm array **11088783**. All five tasks completed with exit code `0:0`; all recorded GPUs were Tesla V100-SXM2-32GB. Task elapsed times reported by `sacct`: seed 0 1:46, seed 1 0:56, seed 2 1:07, seed 3 0:55, seed 4 0:58. These are scheduler records, not controlled hardware benchmarks.

## Protocol

Run `scripts/hpc/supervision_pilot.sbatch` from the isolated remote checkout `/scratch/pandya.kei/mini-wam-research-20261014`. The exact script is in the repository. Existing Python environment: `/home/pandya.kei/.cache/mini-wam/venv-944118425/bin/python` (version and package details in each manifest).

Each seed collects 200 episodes with the repository burst collector (probability 0.3, maximum 80 steps), takes a 90/10 episode training/validation split, and trains three pixel-input conditions for 15 epochs with Adam, learning rate 0.001, batch size 64. The latent dimension is 128. All conditions use H=5 starts, the same initialization, expert labels, minibatch order and update count. State statistics are fit only on training episodes. Hyperparameters are not selected on the final test episodes. No claim of convergence is made.

- `bc`: expert-action regression only.
- `current`: action regression plus 0.5 × current-state MSE.
- `future`: the current objective plus 0.5 × five-step future-state MSE.

No condition uses the image-reconstruction or latent-consistency objective. All initialize the same WAM modules, so unused dynamics in BC/current are untrained. The policy receives one image and instruction; true state is only a training target and normalization reference. Every policy acts reactively, without using imagined states at deployment.

Evaluate 100 paired episodes per training seed, reset seeds `50000 + 1000 * seed + episode`, maximum 80 steps. Store individual outcomes and episode lengths. Seed-level pairs are the comparison units; results are exploratory, without a significance claim.

## Results

| Seed | BC | + Current state | + Future state |
|---|---:|---:|---:|
| 0 | 21% | 26% | 33% |
| 1 | 18% | 24% | 33% |
| 2 | 0% | 26% | 30% |
| 3 | 27% | 38% | 36% |
| 4 | 19% | 28% | 36% |
| Mean ± sample SD | 17.0 ± 10.1% | 28.4 ± 5.5% | 33.6 ± 2.5% |

Current-state supervision improves mean success by 11.4 percentage points. Future supervision adds 5.2 points on average, with paired changes +7, +9, +4, -2, +8. This suggests testing both mechanisms more carefully; it does not establish that learned dynamics explains the earlier archived result.

## Provenance and reproduction

`seed-*/ablation/manifest.json` records dataset SHA-256, source hashes, split IDs, training starts, software and GPU. `results.json` records all three conditions' loss histories, updates and per-episode results. Original datasets and 15 checkpoints remain on Explorer under `runs/job-11088783/seed-*/`; they are not included in this small local results directory.

Recompute the summary and validate paired evaluation seeds, equal update counts, source consistency, complete condition sets and outcome counts:

```sh
python3 scripts/summarize_supervision.py --root docs/results/supervision-pilot
```

The script refuses incomplete seed sets. The `training_seconds` field in the initial runner includes evaluation, so interpret it as condition wall time, not isolated training time. GPU kernels were not forced into deterministic algorithms; reproducible seeds do not guarantee cross-device bitwise agreement.

This study differs from the archived full WAM in dataset size, seeds, objectives and the matching of training starts. Its absolute success must not be compared directly with the archived 80.3% result. Low absolute success, no history baseline, no state-free training condition and no confirmatory test on a frozen schedule remain material limitations.
