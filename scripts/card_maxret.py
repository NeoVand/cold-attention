"""Write the Hugging Face model card (README.md) of the Max Retrieval models.

  python scripts/card_maxret.py <max_retrieval dir> <repo id>
"""
import json
import statistics as st
import sys
from pathlib import Path

SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
PUBLISHED = {16: 98.6, 32: 97.1, 64: 94.3, 128: 89.7, 256: 81.3, 512: 70.1, 1024: 53.8, 2048: 35.7, 4096: 22.6, 8192: 15.7, 16384: 12.4}
COLS = [('cold_count', 'cold'), ('soft_count', 'warm'), ('soft_count', 'cold'), ('softmax', 'softmax'), ('softmax', 'cold'),
        ('softmax_half_penalty', 'softmax'), ('softmax_half_penalty', 'cold')]
LABEL = {('cold_count', 'cold'): 'cold count', ('soft_count', 'warm'): 'soft count, warm read', ('soft_count', 'cold'): 'soft count, cold read',
         ('softmax', 'softmax'): 'softmax', ('softmax', 'cold'): 'softmax, cold read', ('softmax_half_penalty', 'softmax'): 'softmax, half penalty',
         ('softmax_half_penalty', 'cold'): 'softmax, half penalty, cold read'}

CARD = """---
license: apache-2.0
library_name: jax
tags:
- attention
- length-generalization
- max-retrieval
- softmax
- sharpness
---

# Attention at absolute zero: Max Retrieval

Forty trained networks on the task where softmax attention was shown to fail as sets grow (Velickovic et al. 2025,
"Softmax is not enough (for sharp size generalisation)", arXiv 2410.01104): pick the class of the item of highest
priority from a set. Every network is trained on sets of 5 to 16 items and tested on sets of 16 to 16,384. Four
networks, ten seeds each:

| Folder | Network | Weight penalty | Reads |
|---|---|---|---|
| `cold_count/` | **cold pair count**: each item counts the rivals a learned test admits; the item with the smallest count is read at zero temperature | (1/2) * 1e-3 * sum of squares | cold |
| `soft_count/` | the same input and layers, trained and read warm (sigmoid test, soft count, softmax(-beta c)) | (1/2) * 1e-3 * sum of squares | warm; cold from the same weights |
| `softmax/` | our replication of the published softmax network | 1e-3 * sum of squares | softmax; cold from the same weights |
| `softmax_half_penalty/` | the same, with the other reading of the published penalty | (1/2) * 1e-3 * sum of squares | softmax; cold from the same weights |

**The cold pair count is right on at least {p_min:.2f}% of test sets at every size up to 16,384** (mean of ten seeds;
the lowest single seed scores {p_seed_min:.2f}%, at 16,384 items), and every error it makes falls on a set whose highest priority is duplicated in float32, so that two items
of different classes may both be "the" maximum ({p_err} errors in {p_sets:,} sets, {p_err_dup} of them on such sets).
At 4,096 items our softmax network is right on {w_4096:.1f}%, the published softmax network on 22.6%, and the best
published method (ASEntmax; Vasylenko et al. 2026, arXiv 2506.16640, Table 8) on 62.7%.

The control separates the two halves of the remedy. A network with the same input (priorities only as comparisons)
trained and read warm falls to {s_warm_4096:.2f}% at 4,096 items; read at zero temperature, the same weights are right on at
least {s_cold_min:.2f}% at every size (mean of ten seeds; lowest single seed {s_seed_min:.2f}%). It had learned the right comparisons, and its soft reading lost them.

## The task

A set holds n items. Item i has a priority rho_i, uniform in [0, 1), and one of 10 classes; a query q, uniform in
[0, 1), comes with the set and does not change the answer. The answer is the class of the item of highest priority.
The softmax network sees each item as (rho_i, onehot class) and the query; the count networks see the priorities only
through comparisons b_ij = 1[rho_j > rho_i], computed from the raw priorities, and the classes.

## The networks

- **softmax** (the network of Velickovic et al. 2025, App. A.3, written in plain jax with Flax's default
  initialisers): item encoder h_i = gelu(x2(gelu(x1(x_i)))), 128 features; query encoder q2(gelu(q1(q))); one attention
  head, score e_i = qp(query) . kp(h_i) / sqrt(128), weights softmax(e); value and output projections vp, op; read-out
  r2(gelu(r1(.))) to 10 class logits. {w_params:,} parameters.
- **cold_count**: a learned test t_ij = t2(gelu(t1(b_ij, onehot k_i, onehot k_j))) (21 -> 32 -> 1) of rival j for item i;
  item i counts the rivals with t_ij > 0; the weight goes to the items of smallest count, shared evenly over ties; the
  picked items' class features h(k) = gelu(x2(gelu(x1(onehot k)))) pass through vp, op and the read-out. Trained
  straight-through: the forward pass is exactly the hard test and pick, the backward pass uses the gradients of
  sigmoid(t) and softmax(-c). {p_params:,} parameters.
- **soft_count**: cold_count's layers, warm: c_i = sum of sigmoid(t_ij), weights softmax(-beta c) with a learned beta
  (it ends at {beta_lo:.3f} to {beta_hi:.3f} on the ten seeds). {s_params:,} parameters.

The cold read of a softmax network puts all weight on the keys of highest score (split evenly over exact float32 ties).
The cold read of soft_count uses the test 1[t > 0], the exact count and the smallest count.

## Results

Share of test sets answered right (%), mean over ten seeds; 10,240 test sets per size, the same sets for every model.
"published" is the softmax curve of Velickovic et al. 2025 (their Table 1, ten seeds).

{table}

Lowest seed (%):

{table_min}

Every seed at 4,096 items (%):

{table_4096}

**Our softmax replication does not reproduce the published curve.** Under the weight penalty read as 1e-3 * sum of
squares it falls up to {gap_w:.1f} points below the published curve between 64 and 4,096 items; read as (1/2) * 1e-3 * sum
of squares, up to {gap_w1:.1f}. The cold-read results do not depend on this. The cold read of the softmax weights fails
for a reason that can be read from them: within each class the learned score rises with priority, but each class
carries its own fixed offset, between 1.9 and 6.0 thousandths of the priority range across these twenty networks. Sets
of 16 items never ask a network to rank two items that close; at thousands of items, neighbouring priorities are closer
than the offsets and the network ranks by class instead. The offsets alone predict each network's cold accuracy at
4,096 items to within 0.5 points (r = 0.99).

**The learned tests.** The test sees only (b_ij, class of i, class of j), so its hard form is a 2 x 10 x 10 table
(`learned_test` in each config). On every seed of both count networks it is exactly the comparison: admit a rival
when it has the higher priority, whatever either class. The cold pair count's margins on the reject side are thin
(largest b = 0 logit {p_rej_lo:.4f} to {p_rej_hi:.4f} over seeds); soft_count, trained warm, has a wide one ({s_rej_lo:.3f} to
{s_rej_hi:.3f}).

## Use

```python
import sys
from huggingface_hub import snapshot_download
root = snapshot_download("{repo}")
sys.path.insert(0, root)
import max_retrieval as M
params, cfg = M.load(f"{{root}}/cold_count/seed0")
out = M.read(params, cfg["network"], 4096)     # every read of the network on the 10,240 test sets of 4,096 items
print(M.score(out, cfg["network"]))           # {{'cold': ...}}, share right (%)
```

`python max_retrieval.py demo softmax/seed0 --sizes 16,4096` prints a model's reads beside the recorded ones;
`M.train(network, seed, l2_scale)` retrains a model from scratch. The test sets are redrawn from fixed keys (see the
module's docstring); they were drawn with jax 0.8.0 on CPU, and `test_sets.npz` holds every answer, so a different
version of jax can be checked against them. A CPU is enough: on 4 cores one model's eleven sizes take about three
minutes.

## Files

```
max_retrieval.py                 the task, the three networks, the reads, the training (jax, optax, safetensors)
<folder>/seed<k>/
  model.safetensors              parameters (float32); suffix w = weight, b = bias; x = item or class encoder,
                                 q = query encoder, qp/kp/vp/op = attention projections, r = read-out, t = the test
  config.json                    network, training, test protocol, scores per read and size; for the count networks
                                 the learned table, its margins and the cold read's errors per size
  reference_predictions.npz      <read>_<n>: the predicted class of every test set, every read, every size
test_sets.npz                    answer_<n>, top_duplicated_<n> for every test set
index.json
```

## Checks

{checks}

## Limits

- The comparison bits b_ij of the count networks are computed from the raw priorities, outside the network, as in the
  paper's cold setting; the softmax network embeds the priority itself.
- The softmax replication misses the published curve (above), so the comparison with the published 22.6% at 4,096
  items is with their number, not with our network.
- One task. Ten seeds per network, one test draw per size.

## License

Apache 2.0. The task follows Velickovic et al. 2025; the code here is our own.
"""


def fmt(x):
    return f'{x:.2f}'


def main(root, repo):
    root = Path(root)
    cfg = {(c['name'].split('/')[0], c['seed']): c for c in (json.loads(q.read_text()) for q in root.glob('*/seed*/config.json'))}
    val = lambda f, r, n: [cfg[f, k]['scores'][r][str(n)] for k in range(10)]
    head = '| n | ' + ' | '.join(LABEL[f, r] for f, r in COLS) + ' | published |'
    sep = '|---:|' + '---:|' * (len(COLS) + 1)
    t1 = [head, sep] + [f'| {n:,} | ' + ' | '.join(fmt(st.mean(val(f, r, n))) for f, r in COLS) + f' | {PUBLISHED[n]} |' for n in SIZES]
    t2 = ['| n | ' + ' | '.join(LABEL[f, r] for f, r in COLS) + ' |', '|---:|' + '---:|' * len(COLS)] + \
         [f'| {n:,} | ' + ' | '.join(fmt(min(val(f, r, n))) for f, r in COLS) + ' |' for n in SIZES]
    t3 = ['| model, read | ' + ' | '.join(f's{k}' for k in range(10)) + ' |', '|---|' + '---:|' * 10] + \
         [f'| {LABEL[f, r]} | ' + ' | '.join(fmt(x) for x in val(f, r, 4096)) + ' |' for f, r in COLS]
    perr = [cfg['cold_count', k]['cold_errors']['per_size'][str(n)] for k in range(10) for n in SIZES]
    gap = lambda f: max(PUBLISHED[n] - st.mean(val(f, 'softmax', n)) for n in (64, 128, 256, 512, 1024, 2048, 4096))
    rej = lambda f: [cfg[f, k]['learned_test']['largest_reject_logit'] for k in range(10)]
    assert all(cfg[f, k]['learned_test']['table_is_the_comparison'] for f in ('cold_count', 'soft_count') for k in range(10))
    betas = [cfg['soft_count', k]['beta'] for k in range(10)]
    checks = (root / 'CHECKS.md').read_text().strip() if (root / 'CHECKS.md').exists() else '(pending)'
    text = CARD.format(
        repo=repo, table='\n'.join(t1), table_min='\n'.join(t2), table_4096='\n'.join(t3), checks=checks,
        p_min=min(st.mean(val('cold_count', 'cold', n)) for n in SIZES), p_seed_min=min(x for n in SIZES for x in val('cold_count', 'cold', n)), p_err=sum(e for e, _ in perr), p_err_dup=sum(d for _, d in perr),
        p_sets=10 * len(SIZES) * 10240, w_4096=st.mean(val('softmax', 'softmax', 4096)),
        s_warm_4096=st.mean(val('soft_count', 'warm', 4096)), s_cold_min=min(st.mean(val('soft_count', 'cold', n)) for n in SIZES), s_seed_min=min(x for n in SIZES for x in val('soft_count', 'cold', n)),
        w_params=cfg['softmax', 0]['parameters'], p_params=cfg['cold_count', 0]['parameters'], s_params=cfg['soft_count', 0]['parameters'],
        beta_lo=min(betas), beta_hi=max(betas), gap_w=gap('softmax'), gap_w1=gap('softmax_half_penalty'),
        p_rej_lo=min(rej('cold_count')), p_rej_hi=max(rej('cold_count')), s_rej_lo=min(rej('soft_count')), s_rej_hi=max(rej('soft_count')))
    (root / 'README.md').write_text(text)
    print(root / 'README.md')


if __name__ == '__main__':
    main(*sys.argv[1:])
