# The demo page

The demo of *Attention at absolute zero*. Every CLRS-30 task in the browser: pick a task, take an official instance or draw a fresh one at any size, and watch the
trained block loop to its answer, in the page (a Web Worker; no server). Click a pair to see its witnesses.

## Build

The page reads its instances and models from `static/`. The instances (the official val and test splits of every task,
written by `python scripts/export_instances.py data site/static/instances` from the splits `generate.py` makes) are in
git. The models are not; fetch them from the Hugging Face repositories first:

```bash
python scripts/page_assets.py site/static        # the trained models (seed 0), at pinned revisions, with their reference predictions
cd site && pnpm install && pnpm dev               # or: pnpm build (a static site in build/)
```

(`python scripts/site_assets.py <models dir> site/static 0` does the same from a local copy or a trainer's output.)
GitHub Pages (https://neovand.github.io/cold-attention) builds and deploys on every push to main
(`.github/workflows/pages.yml`: the same two steps, with the base path `/cold-attention`; locally `pnpm build:pages`).

## Layout

```
src/lib/model/   the block in TypeScript: engine.ts (CPU, the parity reference), gpu.ts (WebGPU), runner.ts (one
                 interface over both), the safetensors reader, model inputs, the read-out rule, parity
src/lib/clrs/    the library's samplers and reference algorithms in TypeScript (exact on every official instance)
src/lib/demo/    the page's state, the worker that runs the block, the views of each task family
src/routes/      the page (/), the parity check (/parity), and /gpucheck (the two engines side by side, loop by loop)
```

## Engines

The page runs the block on WebGPU when the browser has it (at 64 items on an M4, about 20 ms a loop for the general
block, 30 ms for the cold block, and 60 to 90 ms for the cold block with pair numbers), and on the CPU otherwise (0.6 to 1 s a loop at 64 items). On WebGPU the
dense work runs in compute shaders (linear maps, norm, MLP, the witness counts and the best-witness join); the register
comparisons of the cold blocks run on the CPU with the reference code.

## Parity

The CPU engine is a line-by-line port of `witness_clrs/model.py` (forward only); its arithmetic differs from XLA's only in
summation order. On every official test and val instance of the page's 41 models (36,864 inputs, 1,080,960 answer
entries) it gives the Python release's answer on all but 3 entries (one DFS pointer, two LCS cells), and the same loop
count at rest on all but 4 inputs (answers equal). These come from floating-point summation order, not from the model:
the Python release itself differs at a similar rate between its CPU and GPU builds (10 of 524,288 DFS predictions over
loops 1 to 16, none at the read). The models added later match on all of their official test and val instances
(depth-first search on the cold block with positions encoded and with positions compared, pair numbers on
Floyd-Warshall and Prim, quickselect and task scheduling with numbers compared, topological sort on the cold block: 4,480 inputs, 156,224 entries, no difference), except the cold block on strongly connected
components: 8 of its 2,560 entries differ, on 2 of its 32 test graphs. Its tests there are as close as 4e-8 to their
threshold, so the summation order decides them: XLA's own CPU build differs from the GPU reference on one of the two
graphs (3 entries), and the WebGPU engine, summing in float32 in yet another order, differs on 11 of the 512 entries of
the first 8 test graphs. The WebGPU engine gives the same answers as the CPU engine on the 64-item test inputs checked
(8 per model, the same 3 entries).
`/parity` runs the check in the browser on either engine; `pnpm test` checks the first instances of every model and every
official instance of the reference algorithms; `scripts/parity-cli.ts` runs any range of instances under node.
