# CLRS-30 instances, samplers and reference answers

TypeScript (no dependencies) for the demo page: decode the official CLRS-30 test and val splits, draw fresh
instances of any of the 30 tasks the way DeepMind's `clrs` library (dm-clrs 2.0.3) draws them, and compute the
library's answer exactly. `reference.test.ts` checks `solve` against every official instance (28,096 across both
splits) and the samplers against the val split's statistics.

## Files

**`instances.ts`** names the 30 tasks (`TASKS`, `GRAPH_TASKS`, `SEQ_TASKS`, `isGraph`) and fixes each task's
inputs and outputs (`schema(task)`: library name, split-file name, kind, dtype). `decodeSplit(json)` turns a file
`static/instances/<task>/<split>.json` into typed arrays (`Split.arrays` by file name, `Split.M` the padded size).
`instanceAt(split, i)` / `instances(split)` return unpadded `Instance`s `{task, n, inputs, outputs}` keyed by the
library's names. `toPadded(list, M)` packs instances back into the padded layout of a split file (inputs padded
with 0, per-item and per-pair outputs with -1; identical bytes to the file for the official splits), and
`encodeArray`/`decodeArray` convert one array to and from the file's base64 form.

**`rng.ts`** is a seedable PRNG (`makeRng(seed)`: sfc32 seeded through splitmix32) with `random()` (53-bit doubles),
`uniform`, `int`, `bernoulli`, `permutation` and `choice`. Same seed, same stream everywhere; it is not numpy's
MT19937, so samples have the library's distribution, not its numbers.

**`samplers.ts`** draws fresh instances: `sampleInputs(task, length, rng, opts?)` (inputs only) and
`sample(task, length, rng, opts?)` (inputs plus `solve`'s outputs). `length` is the library's sampler length;
`itemsFor(task, length)` gives the instance's n, `splitLength(task, size)` the length the official splits use for a
nominal size (5/4 of it for the string matchers: 80 for test's 64, 20 for val's 16), `minLength(task)` the smallest
length the library accepts. Defaults are the library's (`p = 0.5`, or 0.2 for articulation_points, bridges and
mst_kruskal); `opts.p` takes a number or a list to draw from per instance and `opts.lengthNeedle` a needle length
(negative `-k`: uniform in 1..k), as `run.py`'s training does. Positions are the library's deterministic ones;
`randomizePos(inst, rng)` applies the library's training-time `process_random_pos` (sorted uniforms within each
string). Graph weights are drawn in float64 and stored as float32, as the split files store them.

**`reference.ts`** has `solve(task, instance)`, ports of `clrs/_src/algorithms/*.py` without probes or hints: the
same loop orders and comparisons (`<` vs `<=`, first index on ties), quickselect's rank `n // 2`, the iterative
relaxations of matrix_chain_order, Bellman-Ford and Floyd-Warshall, the library's own KMP and its 0.01-step DFS
clocks, all in float64. Graph tasks read the float32 weights as stored; on the official splits this reproduces the
library's float64 answers on every instance. Where the library calls numpy's unstable default `argsort`
(Kruskal's edge order, activity_selector, task_scheduling, SCC's second pass, Graham's angles, Prim's and
Dijkstra's extract-min) exact ties go to the lower index; numpy's choice on an exact tie depends on the array
length and CPU sort kernel, and random continuous inputs have none. `diffOutputs(got, expected)` lists differing
entries.

**`reference.test.ts`** runs `solve` on every instance of both official splits of every task and requires exact
equality; checks that the official inputs are laid out as the samplers lay them out and that `toPadded` restores
each file byte for byte; samples 12 instances per task at sizes 4, 8, 16 and 32 and checks shapes, ranges, a few
semantics of the answer (sorted order, minimum, real match, valid topological order) and the padded round trip;
and compares per-instance means of every input and output of 400 samples with the val split (within 5 standard
errors).

## Instance shapes

Arrays are unpadded: `node` = one value per item (length n), `edge` = n·n row-major (`[i * n + j]` is i → j),
`cat` = n·4 row-major one-hot; scalars are plain numbers. Graph tasks store `pos` and `A` as float32 (as the
files do); every other number input is float64, masks and one-hots uint8, outputs int16 (scalars as numbers).

All graph tasks: n = length; `pos` f32 node (i / n), `A` f32 edge (weight, 0 = no edge, 1 for unweighted tasks;
self-loops occur), `adj` u8 edge (`A + I != 0`, so the diagonal is always 1). Undirected weights are
`sqrt(u·u' + 0.001)`, u, u' ~ U[0, 1); DAG weights U[0, 1).

| task | graph drawn | extra input | outputs |
| --- | --- | --- | --- |
| bfs | undirected ER, p 0.5, unweighted | `s` source | `pi` node: parent (s and unreached nodes point to themselves) |
| dfs | directed ER, p 0.5, unweighted | | `pi` node: DFS-forest parent (roots point to themselves) |
| topological_sort | DAG (ER above the diagonal, relabelled), p 0.5 | | `topo` node: next node in the order (last points to itself); `topo_head` number: first node |
| articulation_points | undirected ER, p 0.2 | | `is_cut` node 0/1 |
| bridges | undirected ER, p 0.2 | | `is_bridge` edge: 1 bridge, 0 other edge or diagonal, -1 no edge |
| mst_kruskal | undirected weighted ER, p 0.2 | | `in_mst` edge 0/1 (symmetric) |
| mst_prim | undirected weighted ER, p 0.5 | `s` | `pi` node: tree parent |
| bellman_ford, dijkstra | undirected weighted ER, p 0.5 | `s` | `pi` node: shortest-path parent |
| dag_shortest_paths | weighted DAG, p 0.5 | `s` | `pi` node |
| floyd_warshall | undirected weighted ER, p 0.5 | | `Pi` edge: predecessor of j on the shortest i → j path (i if none) |
| strongly_connected_components | directed, 4 ER communities (p 0.5) plus 1% forward links, relabelled | | `scc_id` node: one node index per component |

Other tasks (`pos` f64 node = i / n unless stated):

| task | n | inputs | outputs |
| --- | --- | --- | --- |
| insertion_sort, bubble_sort, heapsort, quicksort | length | `key` U[0, 1) | `pred` node: predecessor in sorted order (smallest points to itself) |
| minimum | length | `key` U[0, 1) | `min` number: first index of the minimum |
| quickselect | length | `key` U[0, 1) | `median` number: index of the element of rank n // 2 |
| binary_search | length | `key` sorted U[0, 1), `target` number U[0, 1) | `return` number: first index with key ≥ target (n − 1 if none) |
| find_maximum_subarray_kadane | length | `key` U[−1, 1) | `start`, `end` numbers |
| matrix_chain_order | length | `p` U[0, 1) (dimensions) | `s` edge: best split k of chain (i, j), 1 ≤ i < j; 0 elsewhere |
| optimal_bst | length + 1 | `p` (length key probabilities, then 0), `q` (length + 1 gaps); all sum to 1 | `root` edge: root of keys i..j−1 |
| lcs_length | length | `string` u8 (0 = x, 1 = y; \|y\| = length // 2), `pos` from 0 in each string, `key` cat (4 letters) | `b` edge: arrow at row i, column \|x\| + j (0 match, 1 up, 2 left); −1 elsewhere |
| activity_selector | length | `s`, `f`: min and max of two U[0, 1) | `selected` node 0/1 |
| task_scheduling | length | `d` integer deadlines 1..n, `w` U[0, 1) | `selected` node 0/1 |
| naive_string_matcher, kmp_matcher | length (5/4 of nominal) | `string` u8 (0 haystack, 1 needle of length // 5, 1 below 5; the needle is planted in the haystack), `pos` from 0 in each string, `key` cat | `match` number: haystack index of the first match (haystack length if none) |
| segments_intersect | 4 | `x`, `y` U[0, 1); half the draws intersect | `intersect` number 0/1 |
| graham_scan, jarvis_march | length | `x`, `y` uniform on the disk of radius 2, no three points collinear (\|cross\| < 1e−12) | `in_hull` node 0/1 |
