"""Write the Hugging Face model card (README.md) of the per-task variants the paper reports beside the main release.

  python scripts/card_variants.py <variants dir> <repo id>
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from witness_clrs.tasks import NAMES  # noqa: E402

ORDER = ['positions_compared_only', 'scc_multi_entry_pool', 'random_positions', 'deadline_units', 'n_loops', 'open_start', 'recurrence_given', 'recurrence_given_learned_tests', 'stored_count', 'product_pair_number', 'item_memory', 'first_registered_seeds']
TITLES = {
    'positions_compared_only': 'Positions held in a register and only compared (cold setting)',
    'scc_multi_entry_pool': 'Strongly connected components from a pool rich in multi-entry components',
    'random_positions': 'Task scheduling, compare-first, positions randomised in training',
    'deadline_units': 'Task scheduling, general block, deadlines in units of n',
    'n_loops': 'Depth-first search, general block, read after n loops',
    'open_start': 'Pair numbers from an open start',
    'recurrence_given': 'Dynamic programs with the recurrence given as a fixed program (ceilings)',
    'stored_count': 'String matchers given a stored count as a program (evidence for the cause)',
    'product_pair_number': 'Segment intersection given a product pair number (registered failures)',
    'item_memory': 'Item memory in place of pair memory (all 30 tasks, a control)',
    'recurrence_given_learned_tests': 'Dynamic programs with the recurrence given, admission learned',
    'first_registered_seeds': 'Quickselect: the first registered compare-first seeds (registered failures)',
}
ROLE = {
    'positions_compared_only': 'The paper\'s cold-setting row for topological sort and strongly connected components, and the SCC score that committee-chosen training (`cold-attention-scc-committee`) improves on.',
    'scc_multi_entry_pool': 'The paper\'s test of the failure it names for SCC: components entered from several places are rare in the library\'s training graphs.',
    'random_positions': 'Beside the main release\'s compare-first task scheduling, which trains on the library\'s positions.',
    'deadline_units': 'Beside the cold task scheduling: the general block with the deadline in the same units as the positions.',
    'n_loops': 'The paper\'s general-setting depth-first search read after n loops, and the margins of its carried errors.',
    'open_start': 'Counted among the 41 cold models the paper checks for invariance under recoding of the input numbers.',
    'item_memory': 'The supplement\'s item memory against pair memory: averages 76.76% (a tie counted wrong) against 98.44% for the pair block\'s column, and stays within half a point of it on 12 tasks. The tasks that need no middle item hold; those that need one fall.',
    'product_pair_number': 'The one failure without a fix: given the product of two coordinates as a pair number, the cold block stayed near chance (44.30% over three seeds).',
    'stored_count': 'The paper\'s evidence that string matching needs a count carried along the text: given the count as a fixed program, the two matchers are exact on 6 of 6 runs. Reported as evidence for the cause, not as part of the method.',
    'recurrence_given': 'The supplement\'s dynamic programs with the recurrence given: 7 of the 7 fixed-program models are exact. They enter no score the paper reports as its own.',
    'recurrence_given_learned_tests': 'Beside the ceilings: with the program\'s tests learned from a random start, training stopped short of exactness on all three.',
    'first_registered_seeds': 'The paper reports them as registered failures: right on every validation input (16 items), far from it at 64.',
}


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

# Attention at absolute zero: the per-task variants the paper reports

These are trained blocks of **cold attention** that the paper reports beside its main results on DeepMind's CLRS-30:
controls, alternative training, alternative reads, and registered failures. They are kept apart from the main per-task
release (the companion repository `cold-attention-clrs30`) so that its tables stay the paper's chosen models. Every
model here was trained from inputs of at most 16 items (20 for the string matchers) and is tested on the official test
split (64 items; 80 for the string matchers).

{summary}

{sections}

## Files

```
<variant>/<block>/<task>/seed<k>/
  model.safetensors           parameters (float32)
  config.json                 task, block, model and training configuration, read rule, test and val scores, the variant's note
  reference_predictions.npz   predictions on the official test and val splits, after every loop and at the read
<variant>/index.json
```

## Use

```python
from witness_clrs import load_hub, registered_read, score
from witness_clrs.data import load_split
params, cfg = load_hub("{repo}", "positions_compared_only/compare_first/topological_sort/seed0")
d = load_split("data/topological_sort/test.npz")   # official split, written by witness_clrs.generate
outs = [(o["name"], o["kind"]) for o in cfg["outputs"]]
pred, loops = registered_read(params, cfg["model"], outs, d)   # the read rule in cfg["model"] / cfg["read"]
print(score(pred, d, outs))
```

The code is the companion repository's `witness_clrs` package (link added at release).

## Checks

{checks}

## License

Apache 2.0.
"""


def main(root, repo):
    root = Path(root); summary = ['| Variant | Task | Block | Seeds | Test (mean) |', '|---|---|---|---:|---:|']; sections = []
    for v in [x for x in ORDER if (root / x).exists()]:
        cfgs = [json.loads(q.read_text()) for q in sorted((root / v).rglob('config.json'))]
        by = {}
        for c in cfgs:
            by.setdefault((c['task'], c['block']), []).append(c)
        rows = ['| Task | Block | ' + ' | '.join(['seed', 'test', 'val']) + ' |', '|---|---|---:|---:|---:|']
        for (t, b), cs in sorted(by.items()):
            for c in sorted(cs, key=lambda c: c['name']):
                rows.append(f"| {NAMES[t]} | {b} | {c['name'].rsplit('seed', 1)[1]} | {c['scores']['test']:.2f} | {c['scores']['val']:.2f} |")
            summary.append(f"| `{v}/` | {NAMES[t]} | {b} | {len(cs)} | {st.mean(c['scores']['test'] for c in cs):.2f} |")
        c0 = cfgs[0]; tr = c0['train']
        rd = c0['read'] if isinstance(c0['read'], dict) else {}
        read = rd.get('note') or (f"after {rd.get('loops')} loops" if rd.get('rule') == 'fixed' else rd.get('rule', ''))
        sections.append(f"## {TITLES[v]} (`{v}/`)\n\n{c0['variant_note']}\n\n**In the paper:** {ROLE[v]}\n\n**Read:** {read}.\n\n" + '\n'.join(rows))
    checks = (root / 'CHECKS.md').read_text().strip() if (root / 'CHECKS.md').exists() else '(pending)'
    (root / 'README.md').write_text(CARD.format(summary='\n'.join(summary), sections='\n\n'.join(sections), repo=repo, checks=checks))
    print(root / 'README.md')


if __name__ == '__main__':
    main(*sys.argv[1:])
