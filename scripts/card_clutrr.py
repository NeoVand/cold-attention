"""Write the Hugging Face model card (README.md) of the CLUTRR organs.

  python scripts/card_clutrr.py <clutrr dir> <repo id>
"""
import json
import statistics as st
import sys
from pathlib import Path

SPLITS = {'gen_train234_test2to10': 'first split: trained on stories of 2, 3 and 4 relations', 'gen_train23_test2to10': 'second split: trained on 2 and 3'}

CARD = """---
license: apache-2.0
library_name: numpy
tags:
- multi-hop-reasoning
- clutrr
- kinship
- attention
- frozen-language-model
- qwen2.5
---

# Attention at absolute zero: cold attention beside a frozen language model on CLUTRR

CLUTRR (Sinha and others, 2019) is the field's benchmark for kinship reasoning from text. A short story states a chain
of k family relations, and the task is to name the relation between the first and the last person, out of 20. Systems
train on short stories and are tested on stories of up to 10 relations.

A frozen **Qwen2.5-7B-Instruct** reads each pair of people named close together in the story. Beside it, a block of
**cold attention** chains what it read with a rule table it learned from the stories' answers, and answers. Nothing in
the language model is trained or changed, and no rule is written by hand.

| Test accuracy, all test stories (k = 2 to 10), mean of three seeds | First split (1,048 stories) | Second split (1,146 stories) |
|---|---:|---:|
| **Read-in trained from the answers, cost read** | **{tr234:.1f}%** ({tr234s}) | **{tr23:.1f}%** ({tr23s}) |
| **Zero-shot reading, nothing trained in the reader, cost read** | **{zs234:.1f}%** ({zs234s}) | **{zs23:.1f}%** ({zs23s}) |
| Zero-shot reading, bit block | {bt234:.1f}% | {bt23:.1f}% |
| The same frozen model alone, reasoning step by step | {al234:.1f}% | {al23:.1f}% |
| The same frozen model alone, one word | {aw234:.1f}% | {aw23:.1f}% |

**Where this sits, on the same test files.** On the first split's test file, Path-of-Thoughts (arXiv 2412.17963)
reports GPT-5 (medium reasoning) zero-shot at 73.2% and GPT-4o with a rule module (an answer-set solver with
hand-written rules) at 67.7%; GPT-4o reasoning step by step is at 57.6%. On the second split, DSR-LM (Zhang and
others, 2023; a fine-tuned reader and a differentiable logic engine) reports 60.98%. The organ sits below GPT-5 and
below GPT-4o with hand-written rules, with a frozen 7B open model and no hand-written rule. This is not a best on CLUTRR.

**The limit of the trained read-in: a reasoning shortcut** (Marconato and others, NeurIPS 2023). Trained only from
answers, the read-in does not learn to read better. It reads the relations the stories state less faithfully than the
model's own reading: {faith} right, against 89 to 90%. It uses spare relation names as codes of its own (a stated
mother read as "mother-in-law", a father as "father-in-law"; no CLUTRR story states an in-law), and its rule table
composes the codes into the right answer. The answers improve; the reading does not.

**On the benchmark's own graphs**, with no language model, the block is right on {g234} of 1,048 test stories (first
split) and {g23} of 1,146 (second), on each of three seeds, and right on every test story with 5 to 10 relations
on the first split. The rule table it learns is exactly CLUTRR's 62 rules, no more and no fewer, in all six runs.

## How it reads and answers

- **The people:** the names CLUTRR puts in brackets, in order of first appearance.
- **Candidate pairs:** every ordered pair of people whose names are at most two apart in the story's sequence of
  names, with the sentences around their mentions as context.
- **The zero-shot reading:** the model is given the context in its chat template, with an instruction and one
  made-up example, and asked to complete "[B] is [A]'s ___" with one family word or "unknown". Its word and that
  word's probability are kept. Nothing is trained.
- **The trained read-in:** a linear map (75,285 weights) from the model's last-layer state at that place to 21 scores
  (the 20 relations and "nothing stated"). It starts as a fit to the model's own reading of the training stories and
  is then trained from the stories' answers only, through the cost read; no fact is ever labelled.
- **The block:** for each ordered pair of people, 20 bits, one per relation. A learned rule (a, b gives c) turns bit c
  on at (A, C) when bit a is on at (A, B) and bit b at (B, C) for some person B; bits never turn off. The rule table
  is learned from the answers to 2-relation training stories with one loop, then frozen; 40 numbers (a scale and an
  offset per relation) read the answer at the question's pair.
- **Two different tables.** The graph block learned its table from the benchmark's graphs: exactly CLUTRR's 62 rules,
  in all six runs. The text organs use the **teacher table**, which the bit block learned in the zero-shot stage from
  the frozen model's readings of the training stories; it never saw the graphs. It holds {t_lo} to {t_hi} rules: {tt_lo} to
  {tt_hi} of CLUTRR's 62, plus {te_lo} to {te_hi} extra ones. The trained read-in is trained against this table and read with it.
- **The cost read:** the same table with a number in place of each bit. A read relation costs -log of its probability,
  a derived relation costs its cheapest derivation, and the answer is the cheapest relation at the question's pair.
  Nothing is trained in it. The field calls this the most probable proof. A tie, or a story where nothing is
  derived, takes the bit block's answer.

## Results by number of relations (mean of three seeds, %)

{byk}

## Training

{training}

Every choice (which reading, which thresholds, which table, how long the read-in trains) was made on validation
stories; the test stories were read once per stage. `text/<split>/selection.json` holds those choices and the
validation numbers they were made on.

## Files

```
clutrr.py                                     the code: data loader, candidate pairs, the frozen model's reading and
                                              states (torch, transformers), the bit block and the cost read (numpy), Organ
graph/<split>/seed<k>/                        the block on the benchmark's graphs
  model.safetensors                           W (the rule weights, 20 x 20 x 20; a rule is on when W > 0), g, bias
  config.json                                 the 62 rules, training, scores by k
  reference_answers.json                      the answer per validation and test story
text/<split>/
  readin_distilled.npz                        the read-in fitted to the model's own reading (U, b, mu, sd)
  selection.json                              the validation choices
  zero_shot/seed<k>/                          model.safetensors (the bit block learned from the zero-shot reading; its
                                              rules are the teacher table), config.json, reference_answers.json
  trained/seed<k>/                            readin.npz (U, b, mu, sd), fallback.safetensors (the bit block on the
                                              read-in's facts), config.json, reference_answers.json
index.json                                    every organ's test accuracy; the frozen model alone, by k
```

## Use

```python
from huggingface_hub import snapshot_download
import sys
root = snapshot_download("{repo}")
sys.path.insert(0, root)
import clutrr

rows = clutrr.load_clutrr("gen_train234_test2to10", "test")   # downloads CLUTRR's own file and checks its sha256
organ = clutrr.Organ("trained", "gen_train234_test2to10", seed=0, root=root)   # loads Qwen2.5-7B-Instruct on first use
got = organ.answer(rows[:16])
print([g["answer"] for g in got], [r["target_text"] for r in rows[:16]])

graph = clutrr.Organ("graph", "gen_train234_test2to10", seed=0, root=root)   # no language model
print(sum(g["answer"] == r["target_text"] for g, r in zip(graph.answer(rows), rows)), "of", len(rows))   # {g234}
```

The reading needs the frozen model, best on a GPU in float16 (about 16 GB). The block and the cost read run on a CPU.

## Checks

The released code was checked on Modal against the recorded runs:

{checks}

## Limits

- **Not a best on CLUTRR text** (see above). One model family, one prompt; three seeds of the block, one reading.
- **The trained read-in is a reasoning shortcut**: its reading is less faithful than the model's own, and a table
  relearned from it is not the 62 rules (80 to 113 rules). The bit block on its reading stays near a third right.
- **Bits need a clean reading.** With one bit per relation the block on the zero-shot reading falls under the model
  alone on long stories: one wrong fact derives relations at many pairs, and a bit cannot be outweighed. The cost read
  keeps the same table and tolerates the noise.
- **Half the stories are misread somewhere.** The frozen model reads 88.6% of the stated relations right (first
  split), and every stated relation right in half the test stories; then no chaining can be right by proof.
- **CLUTRR stops at 10 relations**, so the length the block reaches is not measured here.
- **The graph result has a caveat:** the graph trainer went through four builds, and the earlier ones were scored on
  the test set before the final design (learn the table from 2-relation stories with one loop, then freeze it) was
  chosen. The text stages have no such caveat.

## License

Code and weights: Apache 2.0. The frozen model is Qwen2.5-7B-Instruct (Apache 2.0), downloaded from its own repository
at the pinned revision. The CLUTRR data is CC BY-NC 4.0 and is not part of this repository; `clutrr.load_clutrr`
downloads it from the dataset's own repository.
"""


def main(root, repo):
    root = Path(root); idx = json.loads((root / 'index.json').read_text())
    C = lambda name: json.loads((root / name / 'config.json').read_text())
    acc = {}
    for split in SPLITS:
        for organ in ('trained', 'zero_shot'):
            cs = [C(f'text/{split}/{organ}/seed{k}') for k in range(3)]
            acc[(split, organ)] = [c['scores']['cost_read']['test']['all'] for c in cs]
            if organ == 'zero_shot':
                acc[(split, 'bits')] = [c['scores']['bits']['test']['all'] for c in cs]
        acc[(split, 'graph')] = [C(f'graph/{split}/seed{k}')['scores']['test']['all'] for k in range(3)]
    m = lambda xs: 100 * st.mean(x['accuracy'] for x in xs)
    seeds = lambda xs: ' / '.join(f"{100 * x['accuracy']:.1f}" for x in xs)
    al = idx['frozen_model_alone']; A = lambda split, arm: 100 * sum(v['right'] for v in al[split][f'test_{arm}'].values()) / sum(v['questions'] for v in al[split][f'test_{arm}'].values())
    g = {s: acc[(s, 'graph')] for s in SPLITS}; assert all(len({x['right'] for x in g[s]}) == 1 for s in SPLITS)
    byk = []
    for split, what in SPLITS.items():
        ks = [str(k) for k in range(2, 11)]; rows = [f'**{what}**', '', '| | ' + ' | '.join(f'k = {k}' for k in ks) + ' |', '|---|' + '---:|' * len(ks)]
        for label, key in (('Trained read-in, cost read', 'trained'), ('Zero-shot reading, cost read', 'zero_shot'), ('Zero-shot reading, bit block', 'bits'), ('Graph input (no language model)', 'graph')):
            if key == 'graph':
                per = [C(f'graph/{split}/seed{s}')['scores']['test'] for s in range(3)]
            elif key == 'bits':
                per = [C(f'text/{split}/zero_shot/seed{s}')['scores']['bits']['test'] for s in range(3)]
            else:
                per = [C(f'text/{split}/{key}/seed{s}')['scores']['cost_read']['test'] for s in range(3)]
            rows.append(f'| {label} | ' + ' | '.join(f"{100 * st.mean(p[k]['accuracy'] for p in per):.0f}" for k in ks) + ' |')
        rows.append('| The frozen model alone, step by step | ' + ' | '.join(f"{100 * al[split]['test_steps'][k]['accuracy']:.0f}" for k in ks) + ' |')
        byk.append('\n'.join(rows))
    t = C('text/gen_train234_test2to10/trained/seed0')['train']; gtr = C('graph/gen_train234_test2to10/seed0')['train']
    training = (f"**The block** (graph input, and the bit block of each text stage): {gtr['updates']:,} updates of batch {gtr['batch']}; {gtr['phase_1']}; {gtr['phase_2']}; "
                f"{gtr['loss']}; {gtr['optimizer']}; {gtr['weight_averaging']}. About 40 seconds on one CPU thread.\n\n"
                f"**The read-in** (trained stage): first {t['distil']}; then {t['from_answers']}. The update kept, per seed: "
                + '; '.join(f"{s}: " + ', '.join(str(C(f'text/{s}/trained/seed{k}')['train']['kept_at_update']) for k in range(3)) for s in SPLITS) + '.')
    G = {s: {tuple(r) for r in C(f'graph/{s}/seed0')['rules']} for s in SPLITS}; assert all(len(v) == 62 for v in G.values())
    Z = [{tuple(r) for r in C(f'text/{s}/zero_shot/seed{k}')['rules']} for s in SPLITS for k in range(3)]; GZ = [G[s] for s in SPLITS for k in range(3)]
    tt = [len(z & g) for z, g in zip(Z, GZ)]; te = [len(z - g) for z, g in zip(Z, GZ)]
    text = CARD.format(t_lo=min(map(len, Z)), t_hi=max(map(len, Z)), tt_lo=min(tt), tt_hi=max(tt), te_lo=min(te), te_hi=max(te), tr234=m(acc[('gen_train234_test2to10', 'trained')]), tr23=m(acc[('gen_train23_test2to10', 'trained')]),
                       tr234s=seeds(acc[('gen_train234_test2to10', 'trained')]), tr23s=seeds(acc[('gen_train23_test2to10', 'trained')]),
                       zs234=m(acc[('gen_train234_test2to10', 'zero_shot')]), zs23=m(acc[('gen_train23_test2to10', 'zero_shot')]),
                       zs234s=seeds(acc[('gen_train234_test2to10', 'zero_shot')]), zs23s=seeds(acc[('gen_train23_test2to10', 'zero_shot')]),
                       bt234=m(acc[('gen_train234_test2to10', 'bits')]), bt23=m(acc[('gen_train23_test2to10', 'bits')]),
                       al234=A('gen_train234_test2to10', 'steps'), al23=A('gen_train23_test2to10', 'steps'), aw234=A('gen_train234_test2to10', 'word'), aw23=A('gen_train23_test2to10', 'word'),
                       faith='77 to 81% of them (seeds and splits)', g234=f"{g['gen_train234_test2to10'][0]['right']:,}", g23=f"{g['gen_train23_test2to10'][0]['right']:,}",
                       byk='\n\n'.join(byk), training=training, repo=repo, checks=(root / 'CHECKS.md').read_text().strip() if (root / 'CHECKS.md').exists() else '(pending)')
    (root / 'README.md').write_text(text)
    print(root / 'README.md')


if __name__ == '__main__':
    main(*sys.argv[1:])
