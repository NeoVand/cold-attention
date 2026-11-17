"""Write the Hugging Face model card (README.md) of a folder of shared blocks (one block trained on all 30 tasks).

  python scripts/card_shared.py <shared dir> <repo id>
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from witness_clrs.tasks import FLOYDNET, NAMES, TASKS  # noqa: E402

VARIANTS = {
    'shared': ('The shared block', 'the census block, read after 8 loops on every task'),
    'shared-units-dfs': ('The shared block with deadlines in units of n and depth-first search read at n loops',
                         'the census block; task scheduling\'s deadlines divided by the input\'s n, as positions are; '
                         'depth-first search read after n loops'),
    'shared-warm-plus-cold': ('The shared block with a cold block beside it',
                              'the census block with a cold block beside it (the paper\'s shared block with a warm and a cold '
                              'half), weight-averaged parameters, each input read after ceil(log2 n) + 3 loops (its registered read)'),
    'shared-five-tasks': ('One census block shared by five tasks',
                          'the census block trained on five tasks only (BFS, Bellman-Ford, quickselect, LCS length, Graham scan), '
                          'final parameters, read after 8 loops; the paper takes its LCS score into the best-of-our-blocks average'),
    'shared-warm-plus-cold-final': ('The same, final parameters read after 8 loops',
                                    'the same three trainings, final parameters, read after 8 loops like the census block'),
}
TWO_BLOCKS = ('A cold block sits beside the census block. It hands on only bits and keeps one built number per pair (a best '
              'bottleneck over the edge weights, max arithmetic), and is rebuilt from the input and its bits every loop; every '
              'number leaves its input (weights to the built number\'s register, positions and item numbers to three register '
              'rows that are only compared). In every loop the cold block runs first and its new bits reach the census state '
              'through a linear map that starts at zero; it never reads the census state. Each answer adds the census read-out '
              'and the cold read-out, so every task has its own second embedding and read-outs.')

VARIANT_TRAIN = {
    'shared-warm-plus-cold': ' ' + TWO_BLOCKS,
    'shared-warm-plus-cold-final': ' The same trainings as `shared-warm-plus-cold/`; the final parameters instead of the average.',
    'shared-units-dfs': (' Depth-first search trains at 8, 14, 20, 26 or 32 loops (one value per update, from the update\'s draw '
                         'of 4 to 8), in a group of its own, and is read after n loops. Task scheduling\'s deadlines are divided by '
                         'the input\'s n in every split and in training.'),
}

UNITS_DFS_RESULT = """### What the second recipe changes (`shared-units-dfs/`)

Two fixes on the first block's recipe, trained the same way otherwise: task scheduling's deadlines divided by the input's n,
and depth-first search trained on 8 to 32 loops and read after n loops. Official test, three seeds:

| read | seed 0 | seed 1 | seed 2 | mean | `shared/` |
|---|---:|---:|---:|---:|---:|
| **registered read** (29 tasks after 8 loops, depth-first search after n loops) | 97.34 | 97.26 | 97.01 | **97.20** | 96.87 |
| all 30 tasks after 8 loops | 97.24 | 96.67 | 96.86 | 96.92 | 96.87 |
| weight-averaged parameters, registered form (secondary; `shared-units-dfs/weight_averaged/`) | 97.70 | 97.53 | 97.49 | 97.57 | |

On a fresh split drawn the same way, the registered read scores 97.06, against 96.65 for `shared/` after 8 loops.

- **Task scheduling:** 91.85 to 99.07 (+7.22), every seed between 98.95 and 99.23.
- **Depth-first search after n loops:** 79.62 (76.42, 91.02, 71.44), against 73.03 for `shared/` after 8 loops (+6.59). The
  seeds spread widely. Read after 8 loops, the same models score 71.21 (-1.82).

**Every loss against `shared/`** (three-seed means, after 8 loops): DAG shortest paths -2.10, which comes from one seed (97.85,
93.07, 99.61); LCS length -0.99; activity selector -0.54; binary search -0.44; optimal BST -0.36; maximum subarray -0.24;
topological sort -0.07; strongly connected components -0.05; quickselect -0.05; Graham scan -0.04; Jarvis' march -0.04; segments
intersect -0.02; bubble sort -0.02; minimum -0.02; MST-Kruskal -0.01. Gains on the other tasks: Prim +0.44, naive string
matcher +0.39, KMP +0.21, Dijkstra +0.08, Floyd-Warshall +0.06, Bellman-Ford +0.05.

Six forecasts were registered before training; 2 of 6 held (every seed's registered read at or above 96.87; task scheduling
at or above 97) and four failed (registered read at or above 97.3, and at or above 97.6; depth-first search after n loops at or
above 85; no other task falling by more than 1.0)."""

CARD = """---
license: apache-2.0
library_name: jax
tags:
- algorithmic-reasoning
- clrs-30
- attention
- graph-neural-network
- length-generalization
- multi-task
---

# Attention at absolute zero: one shared block for all 30 CLRS-30 algorithms

One block of **cold attention** ({shared_params:,} parameters) shared by all 30 algorithms of DeepMind's CLRS-30
benchmark. Each task keeps only its own input encoder and read-outs ({own_lo:,} to {own_hi:,} parameters); the block
that runs the algorithm is the same for all 30. It was trained on all 30 at once, from inputs of at most 16 items, and
is read at 64.

Cold attention, in one paragraph: every ordered pair of items (u, v) carries a vector of 128 numbers. In each loop the
block asks two learned yes-or-no questions of every middle item k, one about the pair (u, k) and one about (k, v); an
item that passes both is a *witness* for (u, v). At zero temperature a sum over witnesses keeps two exact quantities,
the best witness and how many there are. Half of the channels count their witnesses exactly and pass on only which of
two counts is larger, or that they are equal; the other half keep the best witness. This repository holds the block
in its general setting (the census block), where the input's numbers enter through the learned embedding{two_note}.

**Code:** the companion repository (link added at release) has the block, the loader used below, and the per-task
models.

## Results

Test accuracy (%) on the official CLRS-30 test split (64 items; 80 for the string matchers), read after 8 loops{dfs_read}{two_read},
scored with the library's scorer.

{summary}

{table}

FloydNet (arXiv 2601.19094, Table 9) trains one model per task (20 to 215 million parameters) and reports the best of
runs with and without hints.

## Training

{training}

## Files

```
{files}
```

`config.json` lists every task with its outputs, input features, training sizes and test and val scores.
`reference_predictions.npz` holds every task's predictions on the official test and val splits after each of loops
1 to 16{dfs_loops} and at the read (`<task>/<split>/per_loop/<output>`, `<task>/<split>/read/<output>`), made when the block was
trained, on an NVIDIA B200 with jax 0.8.0. The code reproduces them exactly there (all {n_models} task-seed models, every loop).
On an NVIDIA L4, 4 of the 2,739,072 read entries of `shared/` differ (Optimal BST, seed 1) and 139 over loops 1 to 16:
answers at near ties, decided by the order in which the GPU sums.{five_check}

## Use

```python
from witness_clrs import load_shared_hub, registered_read, score
from witness_clrs.data import load_split
params, cfg = load_shared_hub("{repo}", "shared/seed0", "dijkstra")   # the shared block with Dijkstra's own parameters
d = load_split("data/dijkstra/test.npz")                                # official split, written by witness_clrs.generate
outs = [(o["name"], o["kind"]) for o in cfg["outputs"]]
pred, loops = registered_read(params, cfg["model"], outs, d)          # read after 8 loops
print(score(pred, d, outs))
```

## Limits

{limits}

## License

Apache 2.0.
"""


def main(root, repo):
    root = Path(root)
    variants = [v for v in VARIANTS if (root / v).exists()]
    summary, tables, trainings, files = [], [], [], []
    shared_params, own, n_models = None, [], 0
    lows = []
    for v in variants:
        cfgs = [json.loads((root / v / f'seed{k}' / 'config.json').read_text()) for k in range(3) if (root / v / f'seed{k}').exists()]
        if v == 'shared' or shared_params is None:
            shared_params = cfgs[0]['parameters_shared']
        if cfgs[0]['block'] == 'census':
            own += [cfgs[0]['tasks'][t]['parameters_own'] for t in cfgs[0]['tasks']]
        n_models += len(cfgs) * len(cfgs[0]['tasks']) if set(cfgs[0]['tasks']) == set(TASKS) else 0   # the B200-trained blocks
        title, what = VARIANTS[v]
        if set(cfgs[0]['tasks']) != set(TASKS):   # a block shared by a few tasks: its own small table
            ts = [t for t in TASKS if t in cfgs[0]['tasks']]
            rows = ['| Task | FloydNet | ' + ' | '.join(f'seed {k}' for k in range(len(cfgs))) + ' |', '|---|---:|' + '---:|' * len(cfgs)]
            rows += [f"| {NAMES[t]} | {FLOYDNET[t]:.2f} | " + ' | '.join(f"{c['tasks'][t]['scores']['test']:.2f}" for c in cfgs) + ' |' for t in ts]
            tables.append(f'### {title} (`{v}/`): {what}\n\n' + '\n'.join(rows))
            files.append(f'{v}/seed<k>/\n  model.safetensors           the shared block (blk/*) and every task\'s own parameters (task/<task>/*), float32\n  config.json\n  reference_predictions.npz')
            tr = cfgs[0]['train']
            trainings.append(f'`{v}/`: {tr["updates"]:,} updates; in every update one batch of {tr["batch"]} inputs from each of its {len(ts)} tasks, the '
                             f'losses summed; loops drawn from {tr["loops"][0]} to {tr["loops"][-1]} per update; AdamW (learning rate {tr["lr"]}, '
                             f'{tr["warmup"]:,} warm-up steps, cosine decay to {tr["final_lr_frac"]:.0%}, weight decay {tr["weight_decay"]}); '
                             f'{tr["train_pool_per_size"]:,} training inputs per size from the library; seed {tr["seed"]}; the final parameters.')
            continue
        means = [st.mean(c['tasks'][t]['scores']['test'] for t in TASKS) for c in cfgs]
        ahead = sum(st.mean(c['tasks'][t]['scores']['test'] for c in cfgs) >= FLOYDNET[t] - 1e-9 for t in TASKS)
        summary.append(f'| {title} (`{v}/`) | ' + ' | '.join(f'{m:.2f}' for m in means) + f' | **{st.mean(means):.2f}** | {ahead} of 30 |')
        rows = ['| Task | FloydNet | ' + ' | '.join(f'seed {k}' for k in range(len(cfgs))) + ' | mean |', '|---|---:|' + '---:|' * (len(cfgs) + 1)]
        for t in TASKS:
            s = [c['tasks'][t]['scores']['test'] for c in cfgs]
            rows.append(f'| {NAMES[t]} | {FLOYDNET[t]:.2f} | ' + ' | '.join(f'{x:.2f}' for x in s) + f' | {st.mean(s):.2f} |')
            if st.mean(s) < 95 and cfgs[0]['block'] == 'census':
                lows.append((NAMES[t], st.mean(s)))
        tables.append(f'### {title} (`{v}/`): {what}\n\n' + '\n'.join(rows))
        tr = cfgs[0]['train']
        trainings.append(f'`{v}/`: {tr["updates"]:,} updates; in every update one batch of {tr["batch"]} inputs from every '
                         f'task, the 30 losses summed, so the shared block and every task\'s own parameters train together. ' +
                         (f'Loops: {tr["loops"]}; ' if isinstance(tr['loops'], str) else f'Loops drawn from {tr["loops"][0]} to {tr["loops"][-1]} per update; ') +
                         f'AdamW (learning rate {tr["lr"]}, '
                         f'{tr["warmup"]:,} warm-up steps, cosine decay to {tr["final_lr_frac"]:.0%}, weight decay '
                         f'{tr["weight_decay"]}); gradients clipped to norm 1. Training inputs: {tr["train_pool_per_size"]:,} per '
                         'size, drawn by the library itself (dm-clrs 2.0.3) at sizes 4, 7, 11, 13 and 16 (20 for the string '
                         'matchers), as its own training script draws them' + (f', except {tr["pools"]}' if tr.get('pools') else '') +
                         '. No hints (intermediate steps) are used. Seeds 0, 1 and 2. Released parameters: ' + ('the final ones' if tr.get('parameters', 'final') == 'final' else tr['parameters']) + '.' +
                         (VARIANT_TRAIN.get(v, '')))
        files.append(f'{v}/seed<k>/\n  model.safetensors           the shared block (blk/*) and every task\'s own parameters '
                     f'(task/<task>/*)' + (', the cold block (second/blk/*), every task\'s second embedding and read-outs '
                     '(second/task/<task>/*) and the map between the blocks (couple/*)' if cfgs[0]['block'] == 'two_blocks' else '') +
                     ', float32\n  config.json\n  reference_predictions.npz')
    wa = root / 'shared-units-dfs' / 'weight_averaged'
    if wa.exists():   # the secondary read's parameters, beside the registered ones
        wcf = [json.loads(q.read_text()) for q in sorted(wa.glob('seed*/config.json'))]
        n_models += sum(len(c['tasks']) for c in wcf)
        files.append('shared-units-dfs/weight_averaged/seed<k>/   the same trainings\' weight-averaged parameters (the secondary read), '
                     'with their config.json and reference_predictions.npz')
    pair = [v for v in ('shared', 'shared-units-dfs') if v in variants]
    if len(pair) == 2:   # the two census blocks side by side, every task, both numbers
        A, B = ([json.loads((root / v / f'seed{k}' / 'config.json').read_text()) for k in range(3)] for v in pair)
        variants_pair = pair
        m = lambda cs, t: st.mean(c['tasks'][t]['scores']['test'] for c in cs)
        rows = [f'| Task | FloydNet | `{variants_pair[0]}/` | `{variants_pair[1]}/` | change |', '|---|---:|---:|---:|---:|']
        for t in TASKS:
            rows.append(f'| {NAMES[t]} | {FLOYDNET[t]:.2f} | {m(A, t):.2f} | {m(B, t):.2f} | {m(B, t) - m(A, t):+.2f} |')
        lower = sorted(((NAMES[t], m(B, t) - m(A, t)) for t in TASKS if m(B, t) < m(A, t) - 1e-9), key=lambda q: q[1])
        higher = sum(m(B, t) > m(A, t) + 1e-9 for t in TASKS)
        d8 = [st.mean([c['tasks'][t]['scores']['test'] if t != 'dfs' else c['tasks'][t].get('scores_after_8_loops', c['tasks'][t]['scores'])['test'] for t in TASKS]) for c in B]
        hb = [st.mean([c['tasks'][t]['scores']['test'] for t in TASKS]) for c in B]   # the quoted numbers must be these models'
        assert [f'{x:.2f}' for x in hb] == ['97.34', '97.26', '97.01'] and f'{st.mean(hb):.2f}' == '97.20', hb
        tables.insert(0, UNITS_DFS_RESULT)
        tables.insert(0, f'### The two blocks, task by task (mean of three seeds)\n\n' + '\n'.join(rows) +
                      f'\n\nOf the 30 tasks, {higher} {"is" if higher == 1 else "are"} higher in `{variants_pair[1]}/` and {len(lower)} {"is" if len(lower) == 1 else "are"} lower' + (': ' if lower else '') +
                      ', '.join(f'{n} ({d:+.2f})' for n, d in lower) + '. ' +
                      f'Read with depth-first search also after 8 loops, `{variants_pair[1]}/` averages ' +
                      ' / '.join(f'{x:.2f}' for x in d8) + f' over the 30 tasks (mean {st.mean(d8):.2f}).')
    summary = '| | ' + ' | '.join(f'seed {k}' for k in range(3)) + ' | mean | tasks at or above FloydNet |\n|---|---:|---:|---:|---:|---:|\n' + '\n'.join(summary)
    summary = '| | FloydNet (one model per task) |\n|---|---:|\n| average over the 30 tasks | 96.64 |\n\n' + summary
    lows = sorted(set(lows), key=lambda q: q[1])
    limits = ('Read after 8 loops, the block reaches only patterns about 2^8 steps long, and soft numbers must still be told '
              'apart at 64 items. The tasks below 95% (mean of seeds): ' + ', '.join(f'{n} ({m:.1f})' for n, m in lows) +
              '. The per-task models of the companion release reach 100% on depth-first search with loops that grow with '
              'the input, and on the order and path tasks with the cold setting (numbers only as comparisons).')
    text = CARD.format(shared_params=shared_params, own_lo=min(own), own_hi=max(own), summary=summary,
                       n_models=n_models, five_check=(' `shared-five-tasks/` was trained and read on an NVIDIA L4; there the code reproduces '
                                                      'its reference exactly (all 5 tasks, every loop).' if 'shared-five-tasks' in variants else ''), table='\n\n'.join(tables), training='\n\n'.join(trainings), files='\n'.join(files), repo=repo, limits=limits,
                       dfs_read=' (depth-first search in `shared-units-dfs/`: after n loops, here 64)' if 'shared-units-dfs' in variants else '',
                       two_read=(' (`shared-warm-plus-cold/`: each input after ceil(log2 n) + 3 loops, its registered read)'
                                 if 'shared-warm-plus-cold' in variants else ''),
                       two_note=('; and, in two folders, the census block with a cold block beside it, trained on all 30 tasks '
                                 'the same way' if 'shared-warm-plus-cold' in variants else ''),
                       dfs_loops=' (depth-first search in `shared-units-dfs/`: 1 to n)' if 'shared-units-dfs' in variants else '')
    (root / 'README.md').write_text(text)
    print(root / 'README.md')


if __name__ == '__main__':
    main(*sys.argv[1:])
