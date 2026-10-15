# Building and checking the manuscript

The authoritative manuscript is `pico-wam-paper.tex`; `docs/pico-wam-paper.md` is the explicitly marked historical tutorial. The verified working PDF is 12 pages including references, within the requested 10–12 pages. It is not formatted for an actual conference submission.

From the repository root:

```sh
python3 docs/figures/research_figures.py
tectonic --keep-logs docs/paper/pico-wam-paper.tex
pdfinfo docs/paper/pico-wam-paper.pdf
pdftoppm -scale-to 1600 -png docs/paper/pico-wam-paper.pdf /tmp/pico-wam-paper
```

The built-in Codex editor was opened and its compiler attempted. That compiler cannot resolve the external figure assets in this multi-file project. The existing system Tectonic compiler successfully resolves them and produces the PDF; no TeX installation was added. Keep the source open for editing and use the compiled PDF for the complete preview.

Validation performed during revision:

- Recomputed selected archived values directly from per-experiment summaries; spreads in those files use population SD, not sample SD or confidence intervals.
- Ran the fresh affine/simulator audit over five collection seeds; retained complete metrics and source hashes.
- Compiled the three new experiment/figure scripts and ran a one-epoch CPU smoke experiment through all three supervision conditions on a tiny eight-episode dataset. Its zero-success outcomes are smoke-test data only, not paper results.
- Checked the batched policy evaluator against sequential evaluation on four identical initial-state seeds.
- Checked that the current-state auxiliary objective sends gradients into the visual encoder and none into the dynamics module.
- Compiled the revised LaTeX without undefined-reference or overfull-box warnings; checked page count and rendered pages visually.

The pilot uses the same full model initialization for each condition, leaving unused modules untouched in BC/current-state conditions. Only their policy outputs are meaningful; those unused dynamics checkpoints must not be reported as trained world models. `training_seconds` in the initial pilot script includes its subsequent evaluation time, so it is an end-to-end condition runtime, not a training-only timing benchmark. CUDA kernels are not forced into deterministic algorithms: seeds control initialization and minibatch order, not guaranteed cross-device bitwise reproducibility.

The five-seed Explorer pilot completed successfully (array 11088783, all exit codes 0:0). All 15 condition outputs, paired test IDs, equal update counts and local source hashes were verified. Final PDF pages were rendered and visually inspected, including the pilot table; compilation had no undefined-reference or overfull-box warnings.
