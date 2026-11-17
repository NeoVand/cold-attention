"""The shared trainer with each input read at its own loop count, and an optional second block beside the general one:
the eleven-task depth study and the general-and-cold shared block (train_shared.main calls this when the config's train
has loop_rule).

  python -m witness_clrs.train_shared <config.json> <data dir> <out dir> [seed]

Each update draws delta from 0..loop_rule.extra (one random integer), then, with big_pool, one uniform: below
big_pool.share every task in big_pool.tasks takes this update's batch from its second pool (big_L<size>.npz, padded to the
largest size), the others from their own pools; then one batch per task, in the config's task order. Every input is read
after its own K(n) + delta loops, K(n) = ceil(log2 n) + loop_rule.c for n items. The block runs once per group of tasks
with the same number of items, on their batches stacked, for the group's static loop range (scanned loops); each input's
logits are taken at its own loop.

second (config key): a second block of the given kind beside the general block, shared by all tasks, each task with its own
second embedding and read-outs. kind 'cold': the cold block with a fixed register layout (model.two_layout), fed by the
cold inputs (data.two_inputs); its new bits enter the general state through a linear map that starts at zero. kind 'warm':
a second general block whose normalised state enters the same way. The answer is the sum of the two read-outs. Its initial
parameters are drawn from key fold 86, so the general block's draws are those without it. tie_legs (cold): the last
tie_legs exists channels test both legs with the left leg's test.

During training the parameters keep the training code's layout ('blk', 'task' and, for the second block, 'cblk' / 'ctask'
(cold) or 'blk2' / 'task2' (warm), and 'couple'), so the clipped gradient norm sums them in the same order; they are saved
in the released layout ('blk', 'task', 'second': {'blk', 'task'}, 'couple').
"""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import evaluate as E
from .data import feature_counts, load_split, model_inputs, targets
from .model import census_loop, count_params, embed, init_params, lin, norm, outputs_loss, readout, rebuild, two_cold_start, two_cold_step
from .tasks import NAMES, OUTPUTS
from .weights import save


def rule_K(n, c):
    return E.log2_loops(n, c)


def second_names(kind):
    return ('cblk', 'ctask') if kind == 'cold' else ('blk2', 'task2')


def init_second(seed, tasks, cfgs, outs, feats, sc, P):
    """The second block: one block for all tasks (checked equal across tasks' initial draws), each task's own second
    embedding and read-outs, and the zero map into the general state. Key fold 86."""
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 86)
    bn, tn = second_names(sc['kind'])
    ref, own = None, {}
    for t in tasks:
        c2 = dict(cfgs[t], block='cold' if sc['kind'] == 'cold' else 'census', width=sc['width'], mlp=sc['mlp'])
        for k in ('weight_scale',) if sc['kind'] == 'cold' else ():
            c2.pop(k, None)
        p = init_params(key, c2, outs[t], feats[t])
        blk = p.pop('blk')
        if ref is None:
            ref = blk
        assert jax.tree.structure(ref) == jax.tree.structure(blk) and all(bool((np.asarray(a) == np.asarray(b)).all())
                                                                         for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(blk))), t
        own[t] = p
    D = cfgs[tasks[0]]['width']
    return dict(P, **{bn: ref, tn: own}, couple=dict(w=jnp.zeros((sc['width'], D)), b=jnp.zeros((D,))))


def decay_mask(params):
    """AdamW weight decay on everything except the best half's mixing weights of both blocks."""
    mask = jax.tree.map(lambda _: True, params)
    for b in ('blk', 'cblk', 'blk2'):
        if b in mask:
            mask[b]['th_t'] = False
    return mask


def rule_logits(P, tasks, outs, xs, kvs, gl, sc, remat=False):
    """Every task's logits, each input read at its own loop kvs[t] [B]. gl: {group: (tasks, lo, hi)}, static."""
    kind = sc['kind'] if sc else None
    ties = int(sc.get('tie_legs', 0)) if sc else 0
    view = lambda t: dict(P['task'][t], blk=P['blk'])
    z = {t: embed(view(t), xs[t]) for t in tasks}
    if kind == 'warm':
        ec = {t: embed(dict(P['task2'][t], blk=P['blk2']), xs[t]) for t in tasks}
    elif kind == 'cold':
        ec = {t: embed(dict(P['ctask'][t], blk=P['cblk']), xs[t]['cold']) for t in tasks}
    res = {}
    for _, (ts, lo, hi) in gl.items():
        cat = (lambda d: d[ts[0]]) if len(ts) == 1 else (lambda d: jnp.concatenate([d[t] for t in ts]))
        sizes = [xs[t]['m'].shape[0] for t in ts]
        cut = (lambda a: [a]) if len(ts) == 1 else (lambda a: jnp.split(a, np.cumsum(sizes)[:-1]))
        zw, m = cat(z), cat({t: xs[t]['m'] for t in ts})
        pair = (m[:, :, None] * m[:, None, :])[:, None]
        if kind == 'warm':
            e, xg = cat(ec), None
            s = (e,)
        elif kind == 'cold':
            e = cat(ec)
            xg = {k: cat({t: xs[t]['cold'][k] for t in ts}) for k in ('reg', 'W', 'm')}
            s = two_cold_start(P['cblk'], e, xg)
        else:
            e = xg = s = None

        def body(c):
            zw, s = c
            if kind == 'warm':
                s = (census_loop(P['blk2'], s[0], m),)
                zw = zw + lin(norm(s[0]), P['couple']['w'], P['couple']['b']) * pair
            elif kind == 'cold':
                s = two_cold_step(P['cblk'], ties, e, xg, s)
                zw = zw + lin(s[0], P['couple']['w'], P['couple']['b']) * pair
            return census_loop(P['blk'], zw, m), s
        if remat:
            body = jax.checkpoint(body)

        def read(c):
            zs = cut(c[0])
            ss = cut(c[1][0]) if kind else [None] * len(ts)
            es = cut(e) if kind == 'cold' else [None] * len(ts)
            out = {}
            for i, t in enumerate(ts):
                lg = {name: readout(view(t), zs[i], xs[t]['m'], k, name) for name, k in outs[t]}
                if kind == 'warm':
                    lg = {name: lg[name] + readout(dict(P['task2'][t], blk=P['blk2']), ss[i], xs[t]['m'], k, name) for name, k in outs[t]}
                elif kind == 'cold':
                    zc = rebuild(P['cblk'], es[i], ss[i])
                    lg = {name: lg[name] + readout(dict(P['ctask'][t], blk=P['cblk']), zc, xs[t]['m'], k, name) for name, k in outs[t]}
                out[t] = lg
            return out
        carry, _ = jax.lax.scan(lambda c, _: (body(c), None), (zw, s), None, length=lo)
        out = read(carry)
        if hi > lo:
            def b2(cc, step):
                c, o = cc
                c = body(c)
                new = read(c)
                o = {t: {nm: jnp.where((kvs[t] == step).reshape((-1,) + (1,) * (v.ndim - 1)), v, o[t][nm]) for nm, v in new[t].items()} for t in ts}
                return (c, o), None
            (carry, out), _ = jax.lax.scan(b2, (carry, out), jnp.arange(lo + 1, hi + 1))
        res.update(out)
    return res


def make_rule_step(tx, tasks, outs, gl, sc, remat):
    @jax.jit
    def step(P, S, xs, kvs, ys):
        def lossf(P):
            lg = rule_logits(P, tasks, outs, xs, kvs, gl, sc, remat)
            per = {t: outputs_loss(lg[t], xs[t]['m'], ys[t], outs[t]) for t in tasks}
            tot = per[tasks[0]]
            for t in tasks[1:]:
                tot = tot + per[t]
            return tot, per
        (v, per), g = jax.value_and_grad(lossf, has_aux=True)(P)
        u, S = tx.update(g, S, P)
        return optax.apply_updates(P, u), S, v, per
    return step


def released(P, sc):
    """The training layout -> the released layout."""
    out = dict(blk=P['blk'], task=P['task'])
    if sc:
        bn, tn = second_names(sc['kind'])
        out |= dict(second=dict(blk=P[bn], task=P[tn]), couple=P['couple'])
    return out


def task_model(cfg, t, sc):
    """One task's model config for the reads: the general block, or the two blocks."""
    mc = dict(cfg['model'], **cfg.get('task_model', {}).get(t, {}))
    if sc:
        mc = dict(mc, block='two_blocks', second=sc['kind'], second_width=sc['width'], second_mlp=sc['mlp'], tie_legs=int(sc.get('tie_legs', 0)))
    return mc


def main(cfg, data_dir, out_dir, seed=0):
    from .train_shared import build, task_config
    tasks, tc, seed = list(cfg['tasks']), cfg['train'], int(seed)
    rule, big, sc, swa = tc['loop_rule'], tc.get('big_pool'), cfg.get('second'), tc.get('weight_averaging')
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cfgs = {t: task_config(cfg, t) for t in tasks}
    icfg = {t: task_model(cfg, t, sc) for t in tasks}          # inputs: with the cold inputs when the second block is cold
    outs = {t: OUTPUTS[t] for t in tasks}
    root = lambda t: Path(data_dir) / cfg.get('task_data', {}).get(t, '.') / t
    train, splits, feats, NP, bigp, NB = {}, {}, {}, {}, {}, {}
    for t in tasks:
        parts = [load_split(root(t) / f'train_L{L}.npz') for L in cfg.get('task_sizes', {}).get(t, tc['sizes'])]
        train[t] = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        splits[t] = {s: load_split(root(t) / f'{s}.npz') for s in ('val', 'test') + tuple(tc.get('extra_splits', ()))}
        for name in tc.get('task_extra_splits', {}).get(t, ()):
            splits[t][name] = load_split(Path(data_dir) / t / f'{name}.npz')
        feats[t] = feature_counts(splits[t]['val'])
        NP[t] = len(train[t]['n'])
        if big and t in big['tasks']:
            parts = [load_split(Path(data_dir) / big.get('dir', '.') / t / f'big_L{L}.npz') for L in big['sizes']]
            bigp[t] = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
            NB[t] = len(bigp[t]['n'])
    kr = lambda d: (int(rule_K(int(d['n'].min()), rule['c'])[0]), int(rule_K(int(d['n'].max()), rule['c'])[0]))
    krange = {(t, 'main'): kr(train[t]) for t in tasks} | {(t, 'big'): kr(bigp[t]) for t in bigp}
    B, steps, kmax, EB = tc['batch'], tc['updates'], tc.get('eval_loops', 16), tc.get('eval_batch', 16)
    sched = optax.warmup_cosine_decay_schedule(0., tc['lr'], tc['warmup'], steps, tc['lr'] * tc['final_lr_frac'])
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sched, weight_decay=tc['weight_decay'], mask=decay_mask))
    P = build(seed, tasks, cfgs, outs, feats)
    if sc:
        P = init_second(seed, tasks, cfgs, outs, feats, sc, P)
    S = tx.init(P)
    if swa:
        swa_from, swa_n = int(np.ceil(swa['from'] * steps)), 0
        A = jax.tree.map(jnp.zeros_like, P)
    steps_fn = {}
    rng = np.random.default_rng(seed + 100)
    t0 = time.time()
    log = (dest / 'progress.jsonl').open('a')
    for it in range(1, steps + 1):
        K = int(rng.integers(rule['extra'] + 1))
        isbig = bool(big) and bool(rng.random() < big['share'])
        xs, ys, kvs, gl = {}, {}, {}, {}
        for t in tasks:
            src, nps, pool = (bigp[t], NB[t], 'big') if (isbig and t in bigp) else (train[t], NP[t], 'main')
            idx = rng.integers(nps, size=B)
            x = model_inputs(icfg[t], src, idx, rng)
            xs[t] = jax.tree.map(jnp.array, x)
            ys[t] = {k: jnp.array(a) for k, a in targets(src, idx, outs[t]).items()}
            kvs[t] = jnp.array(rule_K(x['m'].sum(1).astype(int), rule['c']) + K, jnp.int32)
            lo, hi = krange[(t, pool)]
            g = gl.setdefault(int(x['m'].shape[1]), [[], 10 ** 9, 0])
            g[0].append(t); g[1], g[2] = min(g[1], lo + K), max(g[2], hi + K)
        gl = {nn: (tuple(g[0]), g[1], g[2]) for nn, g in sorted(gl.items())}
        key = (K, isbig)
        if key not in steps_fn:
            steps_fn[key] = make_rule_step(tx, tasks, outs, gl, sc, bool(tc.get('remat')))
        P, S, v, per = steps_fn[key](P, S, xs, kvs, ys)
        if swa and it >= swa_from and (it % swa['every'] == 0 or it == steps):
            swa_n += 1
            A = jax.tree.map(lambda a, w: a + (w - a) / swa_n, A, P)
        if it % tc.get('check_every', 500) == 0 or it == steps:
            log.write(json.dumps(dict(seed=seed, step=it, loss=float(v), delta=K, big=isbig, sec=round(time.time() - t0, 1),
                                      tasks={t: float(per[t]) for t in tasks})) + '\n'); log.flush()
            if not np.isfinite(float(v)):
                break
    log.close()
    train_sec = round(time.time() - t0, 1)

    finals = [('', P)] + ([('weight_averaged', A)] if swa and swa_n else [])
    result = {}
    for sub, Q in finals:
        where = dest / sub if sub else dest
        where.mkdir(parents=True, exist_ok=True)
        Q = released(Q, sc)
        arrays, tasks_cfg = {}, {}
        for t in tasks:
            mc = dict(icfg[t], read=dict(rule='log2', plus=rule['c']))
            p = dict(Q['task'][t], blk=Q['blk'])
            if sc:
                p |= dict(second=dict(Q['second']['task'][t], blk=Q['second']['blk']), couple=Q['couple'])
            scores, later = {}, {}
            for sp, d in splits[t].items():
                Kn = rule_K(d['n'], rule['c'])
                eb = max(1, min(EB, tc['eval_nodes'] // int(d['n'].max()))) if tc.get('eval_nodes') else EB
                per_loop = E.predict_loops(p, mc, outs[t], d, max(kmax, int(Kn.max()) + 4), eb)
                i = np.arange(len(Kn))
                read, read4 = {k: v[Kn - 1, i] for k, v in per_loop.items()}, {k: v[Kn + 3, i] for k, v in per_loop.items()}
                arrays |= {f'{t}/{sp}/per_loop/{k}': v.astype(np.int16) for k, v in per_loop.items()}
                scores[sp], later[sp] = round(100 * E.score(read, d, outs[t]), 4), round(100 * E.score(read4, d, outs[t]), 4)
            tasks_cfg[t] = dict(task_name=NAMES[t], outputs=[dict(name=n, kind=k) for n, k in outs[t]], features=feats[t],
                                model=cfg.get('task_model', {}).get(t, {}), parameters_own=count_params(Q['task'][t]),
                                scores=scores, scores_4_loops_later=later)
        np.savez_compressed(where / 'predictions.npz', **arrays)
        model = dict(cfg['model'], **({'block': 'two_blocks', 'second': sc['kind'], 'second_width': sc['width'], 'second_mlp': sc['mlp'],
                                        'tie_legs': int(sc.get('tie_legs', 0))} if sc else {}))
        out = dict(name=cfg.get('name', 'shared') + (f'/{sub}' if sub else ''), block=model['block'], model=model,
                   read=dict(rule='log2', plus=rule['c']), parameters_shared=count_params(Q['blk']) + (count_params(Q['second']['blk']) + count_params(Q['couple']) if sc else 0),
                   parameters_total=count_params(Q), train=dict(tc, seed=seed, steps_done=it, train_seconds=train_sec,
                   **({'snapshots': swa_n} if sub else {})), tasks=tasks_cfg)
        save(where, Q, out)
        result[sub or 'final'] = {t: tasks_cfg[t]['scores'] for t in tasks}
    return result
