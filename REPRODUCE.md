# What was checked

Every released model was read again with this code and compared with the predictions it made when it was trained.
"Exact" means every prediction equal, at the registered read and after every loop, with jax 0.8.0 on the hardware the
model was trained on. The reports are produced by `scripts/check_release.py` (per-task models), `check_shared` in
`scripts/modal_app.py` (shared blocks) and the checks named below.

## Reads of the released models

| Repository | Models | Hardware | Result |
|---|---:|---|---|
| cold-attention-clrs30 (one block per task) | 152 | NVIDIA L4 | exact, all 152 |
| cold-attention-clrs30-shared: `shared/` | 90 task-seed models | NVIDIA B200 | exact, all 90 (on an L4, 4 of 2,739,072 read entries differ: near ties) |
| cold-attention-clrs30-shared: `shared-units-dfs/` (the paper's shared block) | 90 | B200 | exact, all 90; depth-first search, read after n loops, is read one compiled loop at a time, as when it was trained |
| cold-attention-clrs30-shared: `shared-units-dfs/weight_averaged/` | 90 | B200 | exact, all 90 |
| cold-attention-clrs30-shared: `shared-warm-plus-cold/`, `-final/` | 180 | B200 | exact, all 180 |
| cold-attention-clrs30-shared: `shared-five-tasks/` | 5 | L4 | exact, all 5 |
| cold-attention-clrs30-eleven | 44 task models | B200 | exact, all 44, on every split including the deeper graphs; the two-block models read one compiled loop at a time |
| cold-attention-clrs30-variants | 73 | L4 (67), H100 (6) | exact, all 73 |
| cold-attention-scc-committee | 3 | L4 | exact, all 3 |
| cold-attention-csp | 9 | H100 | exact, all 9 |
| cold-attention-max-retrieval | 40 | CPU (4 cores) | all 7,884,800 test answers right or wrong exactly as recorded |
| cold-attention-organ | 25 kinship starts, 3 comparison organs, 2 controls | H100 (block), L40S and A10G (frozen models) | every recorded row equal (240 kinship rows, 27 comparison rows, 28 rows of the two controls); read-ins from text bit-identical |
| cold-attention-clutrr | 18 organs | CPU (block), L40S (frozen model) | every stage, split and seed gives the recorded answer from stored readings; from text, 324 of 324 zero-shot and 321 of 324 trained answers |

The page's TypeScript engine (`site/`) gives the same predictions as this package on the official instances
(`pnpm exec vitest --run src/lib/model/parity.test.ts`; `/parity` in the browser).

## Trainers checked against the original training code

A trainer is "verified" when, from the same seed and data, it produces weights equal bit for bit to those of the code
that trained the released models.

| Trainer | Check | Result |
|---|---|---|
| `witness_clrs.train`, general block (`census`) | 6 updates on CPU, on BFS, Kruskal's algorithm and activity selection (seed 0) | weights bit-identical (largest difference 0); the losses logged every 2 updates equal |
| `witness_clrs.train`, cold blocks (`cold`, `compare_first`, `pair_numbers`) | 6 updates on CPU, on depth-first search (positions encoded; positions compared), topological sort (3 seeds), strongly connected components, insertion sort, quickselect, task scheduling, Bellman-Ford, Kruskal's algorithm, Floyd-Warshall and Prim's algorithm | weights bit-identical (largest difference 0); losses equal |
| `witness_clrs.train`, the block variants: tests that start open (pair numbers on Kruskal's algorithm; the string matcher's program), the dynamic programs with fixed and with learned tests (maximum subarray, LCS), triangle pairs with the learned form (segment intersection), item memory (BFS, insertion sort, Kruskal's algorithm with edge values, LCS) | 6 updates on CPU, each against the code that trained that run | weights bit-identical (largest difference 0); losses equal |
| `witness_clrs.train_shared`, the shared general block (`configs/shared/units_dfs.json` cut to six tasks: BFS, Kruskal's algorithm, DFS with its own training loops and read after n loops, insertion sort, task scheduling with deadlines divided by n, the naive string matcher at its own size) | 6 updates on CPU (seed 0), weight averaging from half way, every 2 updates | final and weight-averaged weights bit-identical (69 arrays, largest difference 0); each task's loss equal at every logged update; every prediction after every loop on val and test equal (24 arrays) |
| `witness_clrs.train`, the constraint problems (`configs/csp`: per-input loops, a second pool, positions withheld, weight averaging) | 8 updates on CPU on 2-SAT and two-colouring (seed 0), weight averaging from half way, every 2 updates | final and weight-averaged weights bit-identical (largest difference 0); the logged losses equal |
| `witness_clrs.train_shared` with `loop_rule` (`witness_clrs/train_shared_rule.py`): the eleven-task blocks and the 30-task general-and-cold block (`configs/shared/eleven_*.json`, `general_and_cold.json`) | 6 updates on CPU on five tasks (DFS, topological sort with its deep pool drawn, Prim with weights in the graph's own units, LCS, insertion sort), weight averaging from half way every 2 updates, three cases: the general block alone; with the cold second block (8 tied legs); with a second general block, loops rematerialised | each case bit-identical to the code that trained the released blocks: final and weight-averaged weights (59, 118 and 120 arrays, largest difference 0), every task's loss equal |
| `witness_clrs.pools` (the deep and multi-entry training pools of the eleven-task and general-and-cold blocks, and the depth study's deeper test graphs) | every file regenerated on CPU (dm-clrs 2.0.3, numpy 2.5.4): 32 deep pools (topological sort, strong components, Bellman-Ford and Prim at sizes 4 to 16 and 24 to 32), 5 multi-entry pools (strong components, sizes 4 to 16), 43 test-graph and depth-label files | every array digest equal to the files the blocks were trained and scored on |
| `witness_clrs.committee` (the training graphs for strong components chosen by a committee of three trained blocks) and `witness_clrs.train` on them | the walk rerun on an NVIDIA L4 (the GPU the paper's walk ran on) at sizes 4 and 7; the extra starts and the assembled pools at all five sizes; 6 training updates on CPU on the committee's pools (seed 0) | the walk keeps exactly the paper's graphs (8,953 and 5,911 of 80,000 starts; the same starts, changes and answers); extra starts and pools have the same array digests; weights bit-identical (19 arrays, largest difference 0), losses equal |
| `organ/train.py comparison` (the comparison organ: kept-bit block with tied legs and a yes-only read; arms: the frozen model's layer-0 comparators, names, true relations; and the general-block control) | 6 updates on CPU per case, weight averaging every 2 updates: names (as released, and untied), layer-0 comparators calibrated on the stored states, true relations, the general-block control | every case bit-identical to the code that trained the released organs: final and weight-averaged weights (largest difference 0), losses, read-in comparators and thresholds equal |
| `organ/train.py kinship` (the kinship organ: mention items; pooled, per-sentence or wording planes; within or partner links; the restart rule; the general-block control) | 6 updates on CPU per case (750 for one, two averaged snapshots; 6 + 3 for a forced restart): names, 7B pooled within, 7B per-sentence partner, 0.5B wording, 0.5B general-block control | every case bit-identical: weights (largest difference 0), losses and calibration equal. The feature writer (`organ/train.py features`, the frozen models on a GPU) is ported, not checked bit-identically: it uses the readers whose reads were matched to the stored ones on every evaluation split |
| `clutrr/train.py block` (the bit block: on the true graphs, and on facts from the frozen model's reading, zero-shot or through the trained read-in) and `facts`, `reading` (the facts at a threshold; the read-in's reading) | split 2-3-4, seed 0, on CPU: 6 updates and the released length (2,000 updates) for the graph, zero-shot and fallback blocks; the reading and the facts files | bit-identical to the code that trained the released organs: weights and weight-averaged weights (largest difference 0), losses, rule tables, rows and answers; reading and facts files exactly equal. Against the released organs themselves (trained on another CPU): the same tables (62, 77 and 80 rules), rows and every answer, weights within 2e-6 |
| `clutrr/train.py readin` (the linear read-in from Qwen2.5-7B's stored states, distilled from the zero-shot reading, then trained from answers through the cost read) | one distillation epoch and 6 updates on CPU, split 2-3-4, seed 0 | bit-identical: both read-ins, all probability files and the log equal. The released read-ins (12,000 updates on an NVIDIA L40S) are ported, not checked bit-identically: that run is on a GPU |
| `max_retrieval.train` | seed 0 of each of the four networks, the full 100,000 updates, CPU | bit-identical for the cold count, the soft count and the softmax network; for the softmax network with the halved penalty the first retrain drifted (largest difference 0.43), and three further retrains, each on its own container, were bit-identical (the likely cause is the processor type) |

The constraint-problem generators (`csp/`) rebuild every split of the paper's data: all 63 splits (2-SAT, two-colouring,
parity) have the same array digest as the files the models were trained and scored on.

Not in this repository, or not checked bit for bit:
- the organ's feature writer (`organ/train.py features`, the frozen models on a GPU) and the CLUTRR read-in's full
  12,000-update run (an NVIDIA L40S): both ported, not checked bit-identically, for the reason given in their rows;
- the paper's plotting scripts (the figures are built from the released results).

## Lean

`lean/OrganCert.lean` and `lean/OrganTables.lean` compile with Lean 4.35.0; `#print axioms` gives 22 lines, the same as
the original build, none with `sorryAx`.
