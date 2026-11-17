"""Train one block on one task, with no hints, and score it on the official splits.

  python -m witness_clrs.train <config.json> <data dir> <out dir> [seed]

The config (configs/<block>/<task>.json) holds the model and the schedule. Each step draws a loop count K from
train.loops and a batch of training inputs (sizes 4 to 16; positions randomised as the library does), runs the block K
times and takes the cross-entropy of the outputs read after loop K. AdamW with a warm-up and cosine decay, gradients
clipped to norm 1. The census block trains for train.steps; the resting blocks stop at the first validation check that
follows `patience` checks in a row at 100% for every loop count from loops_from to loops_to.

The constraint problems (configs/csp) use three more keys. train.loop_rule {c, extra}, in place of train.loops: each
batch draws delta from 0..extra and every input is trained after its own ceil(log2 n) + c + delta loops (n items).
train.big_pool {sizes, share}: a second training pool (big_L<size>.npz), from which a batch is drawn with probability
share. train.weight_averaging {from, every}: the running mean of the parameters at every update u >= from * steps with
u % every == 0 (and the last), saved to <out dir>/weight_averaged with its own scores. A model config with a read rule
(read: log2 or n) is scored at that read; train.extra_splits adds evaluation splits; train.eval_nodes caps an
evaluation batch at eval_nodes // (largest n) inputs.

Writes to <out dir>: model.safetensors and config.json (with scores), predictions.npz (registered read and every loop on
test and val), progress.jsonl. Training on one NVIDIA L4 takes about 1.5 to 2 hours for the census block (80,000 steps)
and minutes for the resting blocks.
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import evaluate as E
from .data import feature_counts, load_split, model_inputs, targets
from .model import count_params, forward_rule, init_params, loss_fn, outputs_loss
from .tasks import OUTPUTS
from .weights import save


def decay_mask(params):
    """AdamW weight decay on every parameter except the best half's mixing weights, the via read-out gains, the triangle
    pairs' learned form (read divided by its norm) and the item read's gains."""
    mask = jax.tree.map(lambda _: True, params)
    mask['blk']['th_t'] = False
    if 'via' in mask:
        mask['via'] = False
    if 'tri_m' in mask['blk']:
        mask['blk']['tri_m'] = False
    for q in mask.get('ihit', {}).values():
        q['g'] = False
    return mask


def main(config, data_dir, out_dir, seed=0):
    cfg = config if isinstance(config, dict) else json.loads(Path(config).read_text())
    task, mc, tc = cfg['task'], cfg['model'], cfg['train']
    seed = int(seed)
    outs = OUTPUTS[task]
    data, dest = Path(data_dir) / task, Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    parts = [load_split(data / f'train_L{L}.npz') for L in tc['sizes']]
    train = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    big = tc.get('big_pool')
    if big:
        bparts = [load_split(data / f'big_L{L}.npz') for L in big['sizes']]
        btrain = {k: np.concatenate([q[k] for q in bparts]) for k in bparts[0]}
    splits = {s: load_split(data / f'{s}.npz') for s in ('val', 'test') + tuple(tc.get('extra_splits', ()))}
    feats = feature_counts(splits['val'])
    NP, B, steps = len(train['n']), tc['batch'], tc['steps']
    sched = optax.warmup_cosine_decay_schedule(0., tc['lr'], tc['warmup'], steps, tc['lr'] * tc['final_lr_frac'])
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sched, weight_decay=tc['weight_decay'], mask=decay_mask))

    def make_step(K):
        @jax.jit
        def step(p, s, x, y):
            v, g = jax.value_and_grad(loss_fn)(p, mc, x, y, K, outs)
            u, s = tx.update(g, s, p)
            return optax.apply_updates(p, u), s, v
        return step
    rule = tc.get('loop_rule')
    if rule:   # the loop counts a batch can run, per pool and delta (static for the compiled step)
        pools = [train] + ([btrain] if big else [])
        lohi = {(i, dl): (int(E.log2_loops(int(q['n'].min()), rule['c'])[0]) + dl, int(E.log2_loops(int(q['n'].max()), rule['c'])[0]) + dl)
                for i, q in enumerate(pools) for dl in range(rule['extra'] + 1)}

        def make_rule_step(lo, hi):
            @jax.jit
            def step(p, s, x, y, kv):
                v, g = jax.value_and_grad(lambda p: outputs_loss(forward_rule(p, mc, x, kv, lo, hi, outs), x['m'], y, outs))(p)
                u, s = tx.update(g, s, p)
                return optax.apply_updates(p, u), s, v
            return step
        rule_steps = {}
    else:
        steps_fn = {K: make_step(K) for K in tc['loops']}
    kmax, EB = tc.get('eval_loops', 16), tc.get('eval_batch', 16)
    stop = tc.get('stop')

    rng = np.random.default_rng(seed + 100)
    p = init_params(jax.random.PRNGKey(seed), mc, outs, feats)
    s = tx.init(p)
    swa = tc.get('weight_averaging')
    if swa:
        swa_from, swa_n = int(np.ceil(swa['from'] * steps)), 0
        A = jax.tree.map(jnp.zeros_like, p)
    t0, streak, stopped_at = time.time(), 0, None
    log = (dest / 'progress.jsonl').open('a')
    for it in range(1, steps + 1):
        src, NS, pool = train, NP, 0
        if big and rng.random() < big['share']:
            src, NS, pool = btrain, len(btrain['n']), 1
        if rule:
            dl = int(rng.integers(rule['extra'] + 1))
            lo, K = lohi[(pool, dl)]
        else:
            K = int(tc['loops'][rng.integers(len(tc['loops']))])
        idx = rng.integers(NS, size=B)
        x = model_inputs(mc, src, idx, None if tc.get('fixed_positions') else rng)
        y = targets(src, idx, outs)
        x, y = {k: jnp.array(a) for k, a in x.items()}, {k: jnp.array(a) for k, a in y.items()}
        if rule:
            if (lo, K) not in rule_steps:
                rule_steps[(lo, K)] = make_rule_step(lo, K)
            kv = (E.log2_loops(src['n'][idx], rule['c']) + dl).astype(np.int32)
            p, s, v = rule_steps[(lo, K)](p, s, x, y, jnp.array(kv))
        else:
            p, s, v = steps_fn[K](p, s, x, y)
        if swa and it >= swa_from and (it % swa['every'] == 0 or it == steps):
            swa_n += 1
            A = jax.tree.map(lambda a, w: a + (w - a) / swa_n, A, p)
        if it % tc.get('check_every', 250) == 0 or it == steps:
            row = dict(seed=seed, step=it, loss=float(v), K=K, sec=round(time.time() - t0, 1))
            if stop and it % stop['every'] == 0:  # validation check: the score after each loop 1..kmax
                row['val'] = E.score_loops(E.predict_loops(p, mc, outs, splits['val'], kmax, EB), splits['val'], outs)
                fit = all(q == 1. for q in row['val'][stop['loops_from'] - 1:stop['loops_to']])
                streak = streak + 1 if fit else 0
                if streak >= stop['patience']:
                    stopped_at = it
            log.write(json.dumps(row) + '\n'); log.flush()
            if not np.isfinite(float(v)) or stopped_at:
                break
    log.close()

    train_sec = round(time.time() - t0, 1)
    rule_read = mc.get('read', {}).get('rule') in ('log2', 'n')

    def evaluate(q):
        scores, arrays, loops = {}, {}, {}
        for sp, d in splits.items():
            eb = max(1, min(EB, tc['eval_nodes'] // int(d['n'].max()))) if tc.get('eval_nodes') else EB
            if rule_read:
                read, K, per = (E.predict_log2 if mc['read']['rule'] == 'log2' else E.predict_n)(q, mc, outs, d, eb)
                arrays[f'{sp}/loops'] = K.astype(np.int16)
            elif mc['block'] == 'census':
                per = E.predict_loops(q, mc, outs, d, kmax, eb)
                read = {k: v[E.CENSUS_LOOPS - 1] for k, v in per.items()}
            else:
                r = E.predict_rest(q, mc, outs, d, eb)
                per, read = r['per_loop'], r['pred']
                arrays[f'{sp}/loops'] = r['loops'].astype(np.int16)
                loops[sp] = dict(median=float(np.median(r['loops'])), max=int(r['loops'].max()), rested=float(r['stable'].mean()))
            arrays |= {f'{sp}/per_loop/{k}': v.astype(np.int16) for k, v in per.items()}
            arrays |= {f'{sp}/read/{k}': v.astype(np.int16) for k, v in read.items()}
            scores[sp] = round(100 * E.score(read, d, outs), 4)
        return scores, arrays, loops

    finals = [(dest, p, {})] + ([(dest / 'weight_averaged', A, dict(snapshots=swa_n))] if swa and swa_n else [])
    for where, q, extra in finals:
        where.mkdir(parents=True, exist_ok=True)
        scores, arrays, loops = evaluate(q)
        np.savez_compressed(where / 'predictions.npz', **arrays)
        out = dict(cfg, name=f"{mc['block']}/{task}/seed{seed}" + ('/weight_averaged' if extra else ''), block=mc['block'], features=feats,
                   outputs=[dict(name=n, kind=k) for n, k in outs], parameters=count_params(q), scores=scores,
                   steps_done=it, stopped_at=stopped_at, train_seconds=train_sec, **({'loops_used': loops} if loops else {}))
        out['train'] = dict(tc, seed=seed, **extra)
        save(where, q, out)
        if not extra:
            result = {k: out[k] for k in ('name', 'parameters', 'scores', 'steps_done')}
    return result

if __name__ == '__main__':
    print(json.dumps(main(*sys.argv[1:])))
