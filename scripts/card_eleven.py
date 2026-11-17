"""Write the Hugging Face model card (README.md) of the eleven-task shared blocks (the paper's depth study).

  python scripts/card_eleven.py <eleven dir> <repo id>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from witness_clrs.tasks import NAMES  # noqa: E402

VARIANTS = {
    'warm': ('The census block alone (`warm/`)',
             'The census block (the general setting: the input\'s numbers enter through the learned embedding), the same architecture as the 30-task shared release, trained on these eleven tasks only.'),
    'warm_plus_warm': ('The census block with a second census block beside it (`warm_plus_warm/`)',
                       'The control for the cold block: a second census block beside the first, with its own embedding and read-outs per task. In every loop the second block runs first and its normalised state reaches the first block\'s state through a linear map that starts at zero; the second block never reads the first. Each answer adds the two blocks\' read-outs. It has 3.4% more parameters in all than the census-plus-cold pair below.'),
    'warm_plus_cold': ('The census block with a cold block beside it (`warm_plus_cold/`)',
                       'A cold block beside the census block. The cold block hands on only bits, and keeps one built number per pair, a best bottleneck (max arithmetic) over the edge weights, grown over the witnesses its bits admit; its state is rebuilt from the input and its bits every loop, so once its bits repeat it rests. Every number leaves its input: weights go to the built number\'s register, positions and other item numbers to three register rows that are only compared. In every loop the cold block runs first and its new bits reach the census state through a linear map that starts at zero; the cold block never reads the census state. Each answer adds the census read-out and the cold read-out.'),
    'warm_plus_cold_tied': ('The census block with a cold block, tied legs (`warm_plus_cold_tied/`)',
                            'The same pair, with the last 8 channels of the cold block\'s best half asking one learned question of both legs, so that each such channel squares its relation at every loop.'),
}
DEPTH_SPLITS = ('test', 'chain64', 'deep128', 'deep256', 'lib128', 'dfs96', 'dfs128', 'dfs256')
SPLIT_NAMES = {'test': 'official test (64 nodes)', 'chain64': 'chains of 64 nodes', 'deep128': 'deep graphs of 128 nodes',
               'deep256': 'deep graphs of 256 nodes', 'lib128': 'library-style graphs of 128 nodes', 'dfs96': 'graphs of 96 nodes',
               'dfs128': 'graphs of 128 nodes', 'dfs256': 'graphs of 256 nodes'}

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

# Attention at absolute zero: shared blocks on eleven tasks (the depth study)

Four shared blocks of **cold attention**, each trained on eleven of DeepMind's CLRS-30 algorithms: the tasks whose
answers need long chains of steps or exact numbers ({tasks}). Each task keeps only its own input encoder and read-outs;
the blocks that run the algorithms are shared. The paper uses them for its study of depth: whether a block keeps
doubling its reach on graphs deeper than any it was trained on. Every input of n items is read after
ceil(log2 n) + 3 loops (the registered read) and four loops later; the paper's depth figure uses the later read. One
seed per block.

## Depth: the four blocks on deeper graphs

Share of answers right (%), read after ceil(log2 n) + 7 loops (the paper's depth read), weight-averaged parameters.
The deeper graphs come from the study's own generator and are in `data/`.

{depth}

{sections}

## Use

```python
from huggingface_hub import hf_hub_download
from witness_clrs import load_shared_hub, registered_read, score
from witness_clrs.data import load_split
params, cfg = load_shared_hub("{repo}", "warm_plus_cold/seed0", "bellman_ford")
d = load_split(hf_hub_download("{repo}", "data/bellman_ford/deep256.npz"))   # or the official split from witness_clrs.generate
outs = [(o["name"], o["kind"]) for o in cfg["outputs"]]
pred, loops = registered_read(params, cfg["model"], outs, d)          # each input after ceil(log2 n) + 3 loops
print(score(pred, d, outs))
```

The code is the companion repository's `witness_clrs` package (link added at release).

## Files

```
<block>/seed<k>/
  model.safetensors           the census block (blk/*) and every task's own parameters (task/<task>/*); for two blocks also
                              the second block (second/blk/*), every task's second embedding and read-outs
                              (second/task/<task>/*) and the map between them (couple/*); float32
  config.json                 every task's outputs, input features, training sizes and scores (registered read and four
                              loops later, every split); training; the read rule
  reference_predictions.npz   every task's predictions on every split after loops 1 to 16 and at the registered read
data/<task>/<split>.npz       the deeper evaluation graphs (format of witness_clrs.data.load_split); <split>_depth.npz:
                              each node's depth in the answer and whether it is reachable
```

## Checks

{checks}

## Limits

One seed per block. The census-plus-census control has 3.4% more parameters in all than the census-plus-cold pair, so
the comparison is not at equal size. The deeper graphs are small fixed sets (32 graphs per split; 16 at 256 nodes; for
depth-first search 8 at 96 and at 128 nodes and 4 at 256), so one graph moves a score by up to several points.

## License

Apache 2.0.
"""


def main(root, repo):
    root = Path(root); sections = []; tasks = None; cfgs = {}
    for v, (title, what) in VARIANTS.items():
        d = root / v
        if not d.exists():
            continue
        c = json.loads(sorted(d.glob('seed*/config.json'))[0].read_text()); cfgs[v] = c; tasks = list(c['tasks'])
        rows = ['| Task | test, read | test, 4 loops later | val |', '|---|---:|---:|---:|']
        for t in tasks:
            sc = c['tasks'][t]
            rows.append(f"| {NAMES[t]} | {sc['scores']['test']:.2f} | {sc.get('scores_4_loops_later', {}).get('test', float('nan')):.2f} | {sc['scores']['val']:.2f} |")
        mean = sum(c['tasks'][t]['scores']['test'] for t in tasks) / len(tasks)
        tr = c['train']
        sections.append(f"## {title}\n\n{what} {c['parameters_shared']:,} shared parameters. Seed 0. Read: {c['read']['note']}, "
                        f"with the weight-averaged parameters. Mean test accuracy over the eleven tasks: **{mean:.2f}**.\n\n" + '\n'.join(rows) +
                        f"\n\n**Training:** {tr['updates']:,} updates; in every update one batch of {tr['batch']} inputs from every task, the losses summed; "
                        f"loops {tr['loops']}; AdamW (learning rate {tr['lr']}, {tr['warmup']:,} warm-up steps, cosine decay to {tr['final_lr_frac']:.0%}, "
                        f"weight decay {tr['weight_decay']}); the released parameters are {tr['parameters']}. Training inputs: the library's own "
                        f"(dm-clrs 2.0.3), except {tr['pools']}." if tr.get('pools') else '')
    vs = [v for v in VARIANTS if v in cfgs]
    depth = ['| Task | graphs | ' + ' | '.join(f'`{v}`' for v in vs) + ' |', '|---|---|' + '---:|' * len(vs)]
    for t in tasks:
        for sp in DEPTH_SPLITS:
            if sp == 'test' and not any(sp2 in cfgs[vs[0]]['tasks'][t]['scores'] for sp2 in DEPTH_SPLITS[1:]):
                continue
            if sp not in cfgs[vs[0]]['tasks'][t]['scores']:
                continue
            f = root / 'data' / t / f'{sp}.npz'
            count = len(np.load(f)['n']) if f.exists() else 32
            depth.append(f'| {NAMES[t]} | {SPLIT_NAMES[sp]}, {count} graphs | ' + ' | '.join(f"{cfgs[v]['tasks'][t]['scores_4_loops_later'][sp]:.1f}" for v in vs) + ' |')
    checks = (root / 'CHECKS.md').read_text().strip() if (root / 'CHECKS.md').exists() else '(pending)'
    (root / 'README.md').write_text(CARD.format(tasks=', '.join(NAMES[t] for t in tasks), sections='\n\n'.join(sections), repo=repo,
                                                checks=checks, depth='\n'.join(depth)))
    print(root / 'README.md')

if __name__ == '__main__':
    main(*sys.argv[1:])
