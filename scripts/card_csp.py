"""Write the Hugging Face model card (README.md) of the constraint-problem models (2-SAT, two-colouring).

  python scripts/card_csp.py <csp dir> <repo id>
"""
import json
import statistics as st
import sys
from pathlib import Path

CARD = """---
license: apache-2.0
library_name: jax
tags:
- algorithmic-reasoning
- constraint-satisfaction
- 2-sat
- attention
- graph-neural-network
- length-generalization
---

# Attention at absolute zero: 2-SAT and two-colouring

Blocks of **cold attention** ({params:,} parameters) that decide whether a 2-SAT instance or a two-colouring instance
is unsatisfiable, trained on instances of at most 16 variables (2-SAT) or 32 (two-colouring) and read at up to 500.
Three seeds per problem.

The block is the cold setting of the paper's layer: every ordered pair of items (u, v) carries 128 numbers; each loop
asks two learned yes-or-no questions of every middle item k, and an item that passes both is a witness for (u, v). The
block hands on only bits (which of two witness counts is larger, or that they are equal; whether some witness passes
both tests), and the state is rebuilt from the input and these bits every loop. Item positions are withheld: every
position the block receives is 0, so its answer cannot depend on the order of the items.

## The problems

- **2-SAT** on n variables: 2n literal items and one global item. A(a, b) = 1 when literal a implies literal b (each
  clause (p or q) gives not-p implies q and not-q implies p); adj pairs each literal with its negation. Unsatisfiable
  exactly when some literal and its negation imply each other.
- **Two-colouring** on n variables: n variable items and one global item. A relates two variables that must take the
  same colour, adj two that must differ.

The answer (1: unsatisfiable) is read at the global item's own pair (output `unsat`, stored at item `s`; -1 elsewhere).
Half of the inputs at each size are random instances, balanced to half unsatisfiable; the other half come in twin
pairs: an unsatisfiable instance with a planted contradiction, and a satisfiable twin that differs in one constraint and
has the same counts of every pair input.

## Results

Share of inputs answered right (%), weight-averaged parameters, each input read after ceil(log2 n) + 7 loops for n items.

{table}

{first}## Training

{training}

## Files

```
cold/<problem>/seed<k>/                (also cold_first_registered/{{two_sat,two_colouring}}/seed0 and
                                       cold_positions_given/parity/seed0: registered runs the paper reports)
  model.safetensors           weight-averaged parameters (float32)
  config.json                 model, training and read configuration, scores per split, loops used
  reference_predictions.npz   predictions on every split after each loop, at the read, and the loops used
data/<problem>/{{val,test,n256,n500}}.npz   (parity: val, test, n8, n12, cube, petersen, fit) the evaluation splits (inputs A, adj, pos, s, n; answer unsat; kind: 1 = twin,
                                          pair: twin pair id; depth: shallowest contradiction; nv: variables)
index.json
```

## Use

```python
from huggingface_hub import snapshot_download
from witness_clrs import load, registered_read, score
from witness_clrs.data import load_split
root = snapshot_download("{repo}")
params, cfg = load(f"{{root}}/cold/two_sat/seed0")
d = load_split(f"{{root}}/data/two_sat/test.npz")
outs = [(o["name"], o["kind"]) for o in cfg["outputs"]]
pred, loops = registered_read(params, cfg["model"], outs, d, batch=4)   # each input after ceil(log2 n) + 7 loops
print(score(pred, d, outs))                                            # share of inputs whose answer is right
```

The pair state grows with the square of the items: the 500-variable splits were read one input at a time on an 80 GB
GPU (an NVIDIA H100); a 24 GB GPU runs out of memory there. On an H100 with jax 0.8.0 the code reproduces
`reference_predictions.npz` exactly: all {n_models} models, every split, after every loop. The companion repository (link added at release) has the block and the reader; the
instances were generated with a sampler that is not in it, so the evaluation splits are included here.

## Limits

{limits}

## License

Apache 2.0.
"""

NAMES = {'two_sat': '2-SAT', 'two_colouring': 'Two-colouring'}


def main(root, repo):
    root = Path(root)
    rows = ['| Problem | split (variables, items) | seed 0 | seed 1 | seed 2 | mean |', '|---|---|---:|---:|---:|---:|']
    trainings, wrong = [], []
    params = None
    for task in ('two_sat', 'two_colouring'):
        cfgs = [json.loads((root / 'cold' / task / f'seed{k}' / 'config.json').read_text()) for k in range(3)]
        params = cfgs[0]['parameters']
        for sp, info in cfgs[0]['splits'].items():
            d = json.loads(json.dumps(info))
            items = 2 * d['variables'] + 1 if task == 'two_sat' else d['variables'] + 1
            s = [c['scores'][sp] for c in cfgs]
            rows.append(f'| {NAMES[task]} | {sp} ({d["variables"]}, {items}; {d["inputs"]} inputs) | ' + ' | '.join(f'{x:.2f}' for x in s) + f' | {st.mean(s):.2f} |')
            wrong += [(NAMES[task], sp, k, x) for k, x in enumerate(s) if x < 100]
        tr = cfgs[0]['train']
        trainings.append(f'**{NAMES[task]}:** {tr["variables"][0]} to {tr["variables"][1]} variables; {tr["steps"]:,} updates of batch '
                         f'{tr["batch"]}; loops per batch {tr["loops"]}; AdamW (learning rate {tr["lr"]}, {tr["warmup"]:,} warm-up '
                         f'steps, cosine decay to {tr["final_lr_frac"]:.0%}, weight decay {tr["weight_decay"]}); the read uses the '
                         f'average of {tr["weight_averaging"]["snapshots"]} snapshots from update {tr["weight_averaging"]["first_update"]:,} on, '
                         f'one every {tr["weight_averaging"]["every"]:,}.')
    limits = ('Every split is answered right on every seed, except ' + '; '.join(f'{n} {sp}, seed {k}: {x:.2f}' for n, sp, k, x in wrong) + '. '
              if wrong else 'Every split is answered right on every seed. ')
    limits += ('These are two of the bounded-width constraint classes: their contradictions are paths, which the block grows by '
               'joining witnesses. The block computes no more than counting logic over three variables at a time can: an '
               'unsatisfiable parity instance (three-variable equations) on the cube graph cannot be told from a satisfiable '
               'one by any such rule, and the block with random weights gives the two bit-identical answers.')
    first = ''
    fr = root / 'cold_first_registered' / 'two_sat' / 'seed0' / 'config.json'
    if fr.exists():
        c = json.loads(fr.read_text()); tr = c['train']; sc = c['scores']
        first = ('### The first registered 2-SAT run (`cold_first_registered/two_sat/seed0`)\n\n'
                 'The same block and read, from the first registered training of 2-SAT: learning rate ' + str(tr['lr']) + ' and loops ' + tr['loops'] +
                 '. The supplement and Fig. 6 report it beside the three seeds above, which were trained with the smaller learning rate and '
                 'loop budget that the paper adopted after it. Right (%): ' + ', '.join(f"{sp} {sc[sp]:.2f}" for sp in ('val', 'test', 'n256', 'n500')) +
                 '. It fits the training sizes and does not carry to 64 variables and beyond.\n\n')
    fc = root / 'cold_first_registered' / 'two_colouring' / 'seed0' / 'config.json'
    if fc.exists():
        c = json.loads(fc.read_text()); tr = c['train']; sc = c['scores']
        first += ('### The first two-colouring run (`cold_first_registered/two_colouring/seed0`): collapsed\n\n'
                  'A registered failure, published because the paper reports it: the same block and read, trained at the higher learning '
                  'rate (' + str(tr['lr']) + ', loops ' + tr['loops'] + '). It collapsed to one answer, "unsatisfiable", for every input, so it '
                  'is right on half of every split (' + ', '.join(f"{sp} {sc[sp]:.2f}" for sp in ('val', 'test', 'n256', 'n500')) + ').\n\n')
    fp = root / 'cold_positions_given' / 'parity' / 'seed0' / 'config.json'
    if fp.exists():
        c = json.loads(fp.read_text()); tr = c['train']; sc = c['scores']
        first += ('### Parity with positions given (`cold_positions_given/parity/seed0`): did not fit\n\n'
                  'A registered failure, published because the paper reports it. Parity equations with a hidden twist: on a graph whose '
                  'vertices each have three edges, two variables per edge must differ, and each vertex has eight equations over its three '
                  'edges\' variables; the twisted instance (one vertex\'s right-hand sides flipped) is unsolvable, the plain one solvable, and '
                  'every count of the inputs is the same for both. Items: the variables, the edge equations, the vertex equations and one '
                  'global item (12.5 per vertex, plus one); A links each variable to the equations it occurs in, and each equation carries '
                  'its right-hand side on its own pair of adj. Positions are given here, unlike the models above. Trained on graphs of '
                  f"{tr['variables'][0]} to {tr['variables'][1]} vertices with batch {tr['batch']}, it answers \"solvable\" for every input and does not "
                  'fit its own training instances. Right (%): ' + ', '.join(f"{sp} {sc[sp]:.2f}" for sp in sc) +
                  ' (fit: 96 of its training instances, 54% of them unsolvable; cube and petersen: the cube and the Petersen graph).\n\n')
    n_models = len([q for q in root.rglob('config.json') if 'data' not in q.parts])
    text = CARD.format(n_models=n_models, params=params, table='\n'.join(rows), training='\n\n'.join(trainings), repo=repo, limits=limits, first=first)
    (root / 'README.md').write_text(text)
    print(root / 'README.md')


if __name__ == '__main__':
    main(*sys.argv[1:])
