"""Write the Hugging Face model card (README.md) of the organs: cold attention beside a frozen language model.

  python scripts/card_organ.py <organ dir> <repo id>
"""
import json
import statistics as st
import sys
from pathlib import Path

CARD = """---
license: apache-2.0
library_name: jax
tags:
- multi-hop-reasoning
- length-generalization
- attention
- frozen-language-model
- qwen2.5
---

# Attention at absolute zero: cold attention beside a frozen language model

A frozen Qwen2.5 model reads the text. Beside it, a block of **cold attention** with {params_lo} to {params_hi}
parameters chains what the model read and answers a yes-or-no question. The block was trained from yes/no answers on
contexts of at most 16 statements and is read at up to 1,000. Nothing in the language model is trained or changed.

With frozen Qwen2.5-7B, every released organ answers every test question right, from 16 to 1,000 statements: 6 of 6
starts over two input layouts, with two of the eight sentence wordings never seen in training. The 7B model alone,
given the whole context and asked, answers {alone_7b_16:.1f}% right at 16 statements and {alone_7b_1000:.1f}% at 1,000.

## The block

The items are the mentions of names: the question's X and Y, then, for each statement, the person named first and the
person named second. Every ordered pair of mentions carries 3 bits. In each loop, bit c of (u, v) turns on when some
mention k passes test c at both (u, k) and (k, v). Bits never turn off. A test is a learned function of the pair's input
planes plus a non-negative mix of the pair's bits, so it can ask for a bit to be on and never for one to be off. The
answer at (X, Y) is a non-negative weighting of its three bits minus a non-negative offset: with no bit on, the answer
is no. The block is read after K + 4 loops, where K = ceil(log2 of the number of mentions) + 3.

The input planes of a pair (a, b) of mentions come from the frozen model, and nothing in this read-in is trained:

- **same person:** the keys of a and b (the sum of the model's input embeddings over the name's tokens), centred and
  scaled with numbers from the training statements, have a cosine above a threshold found without labels (just under
  the cluster of matches among the sorted cosines);
- **the statement's link:** a is named first and b second in one statement of class c, or the reverse (layout
  `within`); or a is named in a statement of class c and the other person named there is b's person (layout
  `partner`);
- **the question's pair**, and four flags (a is in the question; b is in the question; the same mention; both real).

A statement's class depends on the frozen model:

- **Qwen2.5-7B and Qwen2.5-1.5B (pooled reading).** Each sentence is given to the model twice, followed by "Therefore,
  the parent is" and by "Therefore, the child is". The reading compares, across the two, the logit of the first-named
  person's first token with the second-named's. Sentences are grouped by wording through the key of their
  closed-class words. Each wording's direction (is the first-named the parent?) is the majority reading over all of
  that wording's sentences read together. No label or answer is used. A wording not seen in training opens a class of
  its own.
- **Qwen2.5-0.5B (wording classes).** This model reads "X is a child of Y" backwards, so the class is the wording
  itself, and the block learns each wording's direction from the answers. All eight wordings are in training.

## The tasks

**Kinship.** Statements in eight wordings: {wordings}. Each states one parent link, and a wording is used only where
it fits the two people's genders. People are drawn from {people:,} full names (96 first names, 120 surnames). A
context has n statements in shuffled order, as one to three lines of descent (`lineage`) or as a family over
generations where a person may have a stated mother and father (`pedigree`). There are 8 questions per context,
"Is X an ancestor of Y?". Half are yes, through a chain of parent links from X down to Y, spread over chain lengths.
The other half are reversed (Y is an ancestor of X), unrelated, or absent (one name does not occur).

**Comparison.** Facts "* A is older than B." over 1,400 single-token names of the model's vocabulary, and the
question "Is X older than Y?". The read-in is 12 comparators on the model's input embeddings at the name positions,
with thresholds found without labels. It is a simpler organ, beside a frozen Qwen2.5-0.5B.

## Results

Questions answered right at the read, per seed. Test contexts: 24 per size (12 at 1,000), 8 questions each, all eight
wordings. The validation splits have 48 contexts; `val6` uses only the six training wordings.

{tables}

{controls}**Every start, Qwen2.5-0.5B.** Ten starts were trained on the 0.5B read-ins, and five learned. Every start that learned
is right on every question at every size; those five are the default folders. The five that did not learn are released too,
as reported failures (`not_learned_<layout>/`). A start that did not learn is already wrong on the 16-statement validation
split, so starts can be selected without reading the test:

{starts}

## Training

{training}

## Files

```
organ.py                                 the code: generators, read-in (torch, transformers), block (jax), Organ.ask
kinship/<model>/
  readin.npz                             mu_p, sd_p, mu_w, sd_w (centring of the keys), theta_p (same-person threshold),
                                         centres (the training wordings' closed-class keys)
  closed.json                            the closed-class words of statements and of questions
  <layout>/seed<k>/                      within or partner; also per_sentence_<layout>/ (7B) and not_learned_<layout>/ (0.5B)
    model.safetensors                    parameters (float32)
    config.json                          frozen model and revision, layout, classes, read rule, training, scores per
                                         split and size (with the read-in's errors and the pooled readings)
    reference_answers.npz                logits per question after K, K + 4 and K + 16 loops, and the labels
comparison/qwen2.5-0.5b/
  names.json                             the 1,400 names
  seed<k>/  model.safetensors, config.json, reference_answers.npz, readin.npz (comparators, thresholds, centring, whitening)
comparison/controls/
  warm/seed0/                            the warm control: model.safetensors, config.json, readin.npz (seed 0's read-in)
  true_relations/seed0/                  the true-relations control: model.safetensors, config.json (no frozen model)
certificates/
  table.json                             every network's certificate verdict (28)
  networks/<folder>.json                 each network's full certificate
  lean/OrganCert.lean, OrganTables.lean  the Lean 4.35 proofs (layout within) and the table checks
index.json                               every organ's test scores; the frozen models alone; every 0.5B start; the
                                         per-sentence variant
```

## Use

```python
from huggingface_hub import snapshot_download
import sys
root = snapshot_download("{repo}", allow_patterns=["organ.py", "kinship/qwen2.5-7b/*"])
sys.path.insert(0, root)
import organ

o = organ.Organ(f"{{root}}/kinship/qwen2.5-7b/within/seed0")   # the default (organ.Organ() fetches it); loads Qwen/Qwen2.5-7B at the pinned revision on first use
statements = ["Maria Lopez is the mother of Tom Baker.", "Ivan Chen is the son of Tom Baker.", "Ivan Chen's father is Tom Baker."]
print(o.ask([(statements, ["Is Maria Lopez an ancestor of Ivan Chen?", "Is Ivan Chen an ancestor of Maria Lopez?"])]))
# [[True, False]]

c = organ.kinship_split("test1000")[0]                          # the first test context of 1,000 statements
print(o.ask([organ.kinship_text(c)]), c["label"])
```

`python organ.py demo kinship/qwen2.5-7b/within/seed0 --statements 256` prints a generated family and the answers.
The read-in needs the frozen model, best on a GPU in float16 (the 7B model needs about 16 GB). The block itself is
small: at 1,000 statements it holds three bit matrices over 2,002 mentions.

## Checks

The released code was checked against the recorded runs on Modal:

{checks}

## Certificates: exact at every size

{certs}

## Limits

- **Template sentences and one relation.** These are not crowd-written stories, and relations are not composed. The
  organ answers beside the model, not through it.
- **The model is read by two extra passes per sentence.** The direction comes from its next-word choice after a short
  probe, not from its inner states.
- **A wording with a new relation word** would need the closed-class words recomputed on text that contains it. The
  two unseen wordings ("{held1}", "{held2}") reuse known words in a new pattern.
- **Read per sentence instead of pooled**, the 7B organ misses {ps_wrong} questions on the test splits on every one of
  six starts (released in `kinship/qwen2.5-7b/per_sentence_<layout>/`). The model misreads 4 of the 36,192 test sentences, and each wrong answer is a chain
  through a misread sentence. Pooling over a wording removes them, but pooling needs several sentences of each
  wording read together.
- **Starts are a lottery when the directions are learned from answers** (Qwen2.5-0.5B): 5 of 10 learned. With the
  direction taken from the model (7B, 1.5B), every start learned: 15 of 15, counting the six per-sentence starts.
- One model family; three to six starts per frozen model.

## License

Apache 2.0. The frozen models are Qwen2.5 (Apache 2.0); they are downloaded from their own repositories at the pinned
revisions and are not part of this repository.
"""

TEST_SIZES = ['16', '32', '64', '128', '256', '512', '1000']


def table(title, cfgs, alone, kind):
    seeds = [c['name'].split('/')[-1].replace('seed', 'seed ') for c in cfgs]
    head = f'**{title}**\n\n| {"Statements" if kind == "kinship" else "Facts"} | ' + ' | '.join(seeds) + ' | frozen model alone |\n|---|' + '---:|' * (len(seeds) + 1)
    rows = []
    for sp, ns in (('test', TEST_SIZES[:-1]), ('test1000', ['1000'])):
        for n in ns:
            cells = []
            for c in cfgs:
                s = c['scores'][sp][n]; right = s['questions'] - (s['wrong_at_K_plus_4'] if kind == 'kinship' else round(s['questions'] * (100 - s['right_pct']['at_K_plus_4']) / 100))
                cells.append(f"{right} / {s['questions']}")
            rows.append(f'| {int(n):,} | ' + ' | '.join(cells) + f" | {alone[sp][n]:.1f}% |")
    val = []
    for sp in [k for k in ('val6', 'val16', 'val24') if k in cfgs[0]['scores']]:
        for n, s0 in cfgs[0]['scores'][sp].items():
            v = [c['scores'][sp][n]['right_pct']['at_K_plus_4'] for c in cfgs]
            val.append(f'{sp} ({n} statements): ' + ' / '.join(f'{x:g}%' for x in v))
    return head + '\n' + '\n'.join(rows) + '\n\nValidation, right at the read: ' + '; '.join(val) + '.'


def main(root, repo):
    root = Path(root); index = json.loads((root / 'index.json').read_text())
    cfgs = {o['name']: json.loads((root / o['name'] / 'config.json').read_text()) for o in index['organs']}
    from safetensors.numpy import load_file
    params = [sum(v.size for v in load_file(str(root / n / 'model.safetensors')).values()) for n in cfgs]
    alone = index['frozen_model_alone_right_pct']
    groups = [('Kinship, frozen Qwen2.5-7B, pooled reading, layout `within` (the default)', 'kinship/qwen2.5-7b/within/'),
              ('Kinship, frozen Qwen2.5-7B, pooled reading, layout `partner`', 'kinship/qwen2.5-7b/partner/'),
              ('Kinship, frozen Qwen2.5-1.5B, pooled reading, layout `within`', 'kinship/qwen2.5-1.5b/within/'),
              ('Kinship, frozen Qwen2.5-0.5B, wording classes, layout `within` (the 2 of 5 starts that learned)', 'kinship/qwen2.5-0.5b/within/'),
              ('Kinship, frozen Qwen2.5-0.5B, wording classes, layout `partner` (the 3 of 5 starts that learned)', 'kinship/qwen2.5-0.5b/partner/'),
              ('Comparison, frozen Qwen2.5-0.5B', 'comparison/qwen2.5-0.5b/')]
    tables = []
    for title, pre in groups:
        cs = [c for n, c in sorted(cfgs.items()) if n.startswith(pre)]
        tables.append(table(title, cs, alone[pre.rsplit('/', 2)[0] if pre.startswith('kinship') else 'comparison/qwen2.5-0.5b'], cs[0]['task']))
    controls = ''
    cdir = root / 'comparison' / 'controls'
    if cdir.exists():
        cw = json.loads((cdir / 'warm' / 'seed0' / 'config.json').read_text()); ct = json.loads((cdir / 'true_relations' / 'seed0' / 'config.json').read_text())
        cold = [json.loads((root / 'comparison' / 'qwen2.5-0.5b' / f'seed{k}' / 'config.json').read_text()) for k in range(3)]
        rows = ['| Facts | cold organ, seeds 0 / 1 / 2 | warm control (questions) | true-relations control |', '|---:|---|---:|---:|']
        for sp in ('test', 'test1000'):
            for n in cw['scores'][sp]:
                c = ' / '.join(f"{x['scores'][sp][n]['right_pct']['at_K_plus_4']:g}%" for x in cold)
                w = cw['scores'][sp][n]
                rows.append(f"| {int(n):,} | {c} | {w['right_pct']['at_K_plus_4']:g}% ({w['questions']}) | {ct['scores'][sp][n]['right_pct']['at_K_plus_4']:g}% |")
        controls = ('**Controls for the comparison organ** (`comparison/controls/`, seed 0 each). The warm control puts a census block '
                    '(width 128, a state that carries strengths) in place of the cold block, on the same comparator bits as seed 0 '
                    'and trained 25,000 updates; it fits every training and validation question and falls as the contexts grow. '
                    'Its reads at 512 and 1,000 facts were capped at 16 and 4 questions per family of context to bound their cost. '
                    'The true-relations control gives the cold block hand-made "same name" relations in place of the model\'s read '
                    'and fails, about two thirds right at every size; the paper puts this down to how those inputs were built. '
                    'Right at the read (K + 4), test contexts:\n\n' + '\n'.join(rows) + '\n\n')
    starts = ['| Layout | seed | val16 (16 statements) | test, 16 to 512 | test1000 | folder |', '|---|---:|---:|---:|---:|---|']
    for s in index['kinship_qwen2_5_0_5b_every_start']:
        r = s['right_pct_at_read']; t = [r[f'test/{n}'] for n in TEST_SIZES[:-1]]
        starts.append(f"| {s['layout']} | {s['seed']} | {r['val16/16']:g}% | {(f'{min(t):g}%' if min(t) == max(t) else f'{min(t):g} to {max(t):g}%')} | {r['test1000/1000']:g}% | `{s['folder']}/` |")
    import organ as O  # noqa: E402  (organ.py beside the card's folder)
    wordings = ', '.join(f'"{w[0].format(P="P", C="C")}"' for w in O.WORDINGS)
    k7 = cfgs['kinship/qwen2.5-7b/within/seed0']['train']; k5 = cfgs['kinship/qwen2.5-0.5b/within/seed2']['train']; cp = cfgs['comparison/qwen2.5-0.5b/seed0']['train']
    training = (f"Kinship: contexts of {k7['statements'][0]} to {k7['statements'][1]} statements, {k7['contexts_per_size']} per size, "
                f"{k7['questions_per_context']} questions each; for the 7B and 1.5B organs only the six wordings {', '.join(chr(34) + w.format(P='P', C='C') + chr(34) for w in k7['wordings'])}; "
                f"for the 0.5B organs all eight. {k7['updates']:,} updates of batch {k7['batch']} (one size drawn per batch); loops {k7['loops']}; "
                f"{k7['loss']}; {k7['optimizer']}; {k7['weight_averaging']}. Comparison: the same, on contexts of {cp['facts'][0]} to {cp['facts'][1]} facts; "
                f"loops {cp['loops']}. Each start trains in about two minutes on one H100. The read-in is not trained.")
    ps = index['kinship_qwen2_5_7b_per_sentence_reading']
    ps_wrong = sorted({sum(v for k, v in s['wrong_at_read'].items() if k.startswith('test')) for s in ps})
    checks = (root / 'CHECKS.md').read_text().strip() if (root / 'CHECKS.md').exists() else '(pending)'
    k = cfgs['kinship/qwen2.5-7b/within/seed0']['scores']
    cert_dir = root / 'certificates'; certs = '(none)'
    if cert_dir.exists():
        tab = json.loads((cert_dir / 'table.json').read_text()); ok = [r['network'] for r in tab if r['exact']]; bad = [r['network'] for r in tab if not r['exact']]
        learned = {o['name'] for o in index['organs'] if 'not_learned' not in o['name']}
        assert set(ok) == learned and all('not_learned' in b for b in bad), (set(ok) ^ learned, bad)
        walkers = [r['network'] for r in tab if r['chain_class'] == 'walker']
        certs = (f"`certificates/` holds a finite certificate for each of the {len(tab)} networks here. A trained block is a finite table: for "
                 "each pattern of a pair of mentions and each set of kept bits, which tests fire. Two finite checks on the table decide "
                 "whether the network, given its input bits, answers every question right in every family of any size at the registered "
                 "number of loops: (1) **no false yes at any size** (every \"no\" context maps into one small context of kinds of mention, "
                 "and the kept bits the table allows there never make the read-out say yes); (2) **every true yes in time, at every length** "
                 "(bare chains of up to 48 items computed in classes of pairs, longer chains by an induction on distance).\n\n"
                 f"- `table.json`: each network's verdict. `exact` is true for all {len(ok)} networks that learned and false for the "
                 f"{len(bad)} that did not. Beside it: `no_false_yes`, `chains_in_loops`, `chain_class` (`in_time`; `walker`: right at "
                 "rest, too slow for the registered loops; `never`), the loops each chain length needs against the registered ones, "
                 "spot checks on random contexts, and `engine_differences` (the certificate's own engine against the trained code).\n"
                 + (f"- The check sees the loop count, which a run-to-rest test cannot: {', '.join('`' + w + '`' for w in walkers)} is a walker "
                    "(it answers a chain of k statements at about loop 2k), right on every context run to rest and rejected here.\n" if walkers else '')
                 + "- `networks/`: each network's full certificate.\n"
                 "- `lean/`: the result machine-checked in Lean 4.35 (core, no Mathlib) for the nine trained networks of layout `within` "
                 "with two direction classes (Qwen2.5-7B `within` and `per_sentence_within`, Qwen2.5-1.5B `within`; seeds 0 to 2): "
                 "`correct_<network>` (right on every well-formed instance of every size) and `no_false_yes_<network>`. Four `partner` "
                 "networks pass the two table checks there; the lemma placing their patterns on a shortest chain is measured, not "
                 "proved. Compile with `lean -o OrganCert.olean OrganCert.lean` and then `LEAN_PATH=. lean OrganTables.lean`; every "
                 "theorem depends only on `propext`, `Classical.choice` and `Quot.sound`. Not proved in Lean: that each table equals "
                 "the float network (it is the network's forward pass evaluated on every case of its finite domain).")
    text = CARD.format(certs=certs, params_lo=min(params), params_hi=max(params), alone_7b_16=alone['kinship/qwen2.5-7b']['test']['16'], alone_7b_1000=alone['kinship/qwen2.5-7b']['test1000']['1000'],
                       wordings=wordings, people=O.N_PEOPLE, tables='\n\n'.join(tables), controls=controls, starts='\n'.join(starts), training=training, repo=repo, checks=checks,
                       held1=O.WORDINGS[1][0].format(P='P', C='C'), held2=O.WORDINGS[6][0].format(P='P', C='C'), ps_wrong=' or '.join(map(str, ps_wrong)))
    assert all(s['right_pct']['at_K_plus_4'] == 100 for sp in ('test', 'test1000') for s in k[sp].values())
    (root / 'README.md').write_text(text)
    print(root / 'README.md')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
    main(*sys.argv[1:])
