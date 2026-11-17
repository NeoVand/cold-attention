"""Write the Hugging Face model card (README.md) of the models folder.

  python scripts/model_card.py <models dir> <repo id>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables  # noqa: E402

CARD = """---
license: apache-2.0
library_name: jax
tags:
- algorithmic-reasoning
- clrs-30
- attention
- graph-neural-network
- length-generalization
---

# Attention at absolute zero: cold attention on CLRS-30

The trained models of **cold attention**. An attention layer sums evidence over its witnesses and reads the sum at a
temperature; as the temperature falls to zero, the sum keeps two exact quantities, the score of the best witness and the
number of witnesses that reach it. Cold attention computes them and passes the rest of the network only their
comparisons.

One small block of it (about 150,000 parameters, one configuration) trained separately on each of the 30
algorithms of DeepMind's CLRS-30 benchmark, from inputs of at most 16 items, and tested at 64. This repository holds
every trained model: {count} models, {tasks} tasks, five blocks, three seeds each (five for compare-first quickselect).

**Code, training and the demo:** see the companion repository (link added at release). The demo page runs these
models in the browser.

## The block in one paragraph

Every ordered pair of items (u, v) carries a vector of 128 numbers. In each loop the block asks two learned yes-or-no
questions of every middle item k: a test on the pair (u, k) and a test on the pair (k, v). An item that passes both is
a *witness* for (u, v). Half of the 128 channels count their witnesses exactly and pass on only comparisons between
two counts ("more", "same"); the other half keep the best witness. The block is looped, and its answer is read by a
fixed rule. The block runs in two settings. In the **general** setting the input's numbers enter through the learned
embedding; in the **cold** setting they stay in exact registers and only comparisons of them enter the learned state.

| Block | Setting | Numbers enter | Read |
|---|---|---|---|
| `census` | general | through the learned input embedding | after 8 loops |
| `cold` | cold | none (the task has no input numbers); only bits enter the state | when its bits stop changing (at most n loops) |
| `compare_first` | cold | never: input numbers sit in registers and the state receives only comparisons of them (on depth-first search, the positions; its bits are not kept) | when its bits stop changing (at most n loops) |
| `pair_numbers` | cold | as above, plus one number per pair the block builds (a best path sum or bottleneck); Floyd-Warshall adds via pairs, Prim a second built number (the hops) | when its bits and numbers stop changing (at most 4n loops) |
| `census_bilinear` | general | as `census`; its first 16 count channels count a witness only if it also passes a bilinear test of the two legs' states (rank 2), trained 320,000 steps (segments intersect) | after 8 loops |

## Results

### Best block per task

{best}

### Every block

{tables}

## Files

```
<block>/<task>/seed<k>/
  model.safetensors           parameters (float32), names like blk/a, in30/e1, one30/min
  config.json                 task, block, model and training configuration, test and val scores, parameter count
  reference_predictions.npz   the model's predictions on the official test and val splits (the registered read and
                              every loop), made when it was trained; the code reproduces them (check_release.py)
index.json                    every model with its scores
best.json                     per task, the block with the best mean test score over its seeds (ties: the block the
                              task's symmetry calls for)
```

## Use

```python
from witness_clrs import load_hub, registered_read, score, OUTPUTS
from witness_clrs.data import load_split
params, cfg = load_hub("{repo}", "census/bfs/seed0")
d = load_split("data/bfs/test.npz")                        # official split, written by witness_clrs.generate
outs = [(o["name"], o["kind"]) for o in cfg["outputs"]]
pred, loops = registered_read(params, cfg["model"], outs, d)
print(score(pred, d, outs))                                # 1.0
```

## Training data and evaluation

Data from the library itself (dm-clrs 2.0.3): the official test (64 items, 80 for the string matchers) and val (16)
splits, and training pools drawn as the library's own training script draws them (sizes 4 to 16). No hints
(intermediate steps) are used. Scores use the library's scorer. Each model's training configuration is in its
`config.json`.

## Limits

Tasks whose answers depend on sums of numbers that are not comparisons (optimal BST, the maximum subarray, LCS) remain
below 100%. Depth-first search reaches 100% only with the cold block read after n loops (the census block, read after
8, scores 73%). On strongly connected components the cold block (86 to 90%) is below the census block (about 91%). The
cold-setting blocks cover only the tasks listed.

## License

Apache 2.0.
"""


def best_table(models):
    import statistics as st
    from witness_clrs.tasks import FLOYDNET, NAMES, TASKS
    best = json.loads((Path(models) / 'best.json').read_text())
    rows = ['| Task | FloydNet | Best block | Mean test (seeds) |', '|---|---:|---|---:|']
    for t in TASKS:
        b = best[t]
        rows.append(f'| {NAMES[t]} | {FLOYDNET[t]:.2f} | `{b["block"]}` | {b["test_mean"]:.2f} ({b["seeds"]}) |')
    m = st.mean(best[t]['test_mean'] for t in TASKS)
    ahead = sum(best[t]['test_mean'] >= FLOYDNET[t] - 1e-9 for t in TASKS)
    return (f'Average over the 30 tasks: **{m:.2f}** (FloydNet 96.64); at or above FloydNet on {ahead} of 30. A selection over '
            'blocks per task, so optimistic; the census block alone (one configuration on all 30) averages 97.04. Where '
            'blocks tie, the table names the block the task\'s symmetry calls for (`best.json` lists the others).\n\n' + '\n'.join(rows))


def main(models, repo):
    idx = tables.load(models)
    text = CARD.format(count=len(idx), tasks=len({m['task'] for m in idx}), tables=tables.tables(idx), repo=repo,
                       best=best_table(models))
    (Path(models) / 'README.md').write_text(text)
    print(Path(models) / 'README.md')


if __name__ == '__main__':
    main(*sys.argv[1:])
