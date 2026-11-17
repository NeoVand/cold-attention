"""Train one shared general block on many CLRS-30 tasks at once, with no hints.

  python -m witness_clrs.train_shared <config.json> <data dir> <out dir> [seed]

Every task keeps its own input embedding and read-outs; the looped block is one, shared by all tasks. Each task's
initial parameters are init_params(PRNGKey(seed), its config), so every task draws the same block, which the model keeps
once: params = {'blk': the block, 'task': {task: its own parameters}}.

Each update draws one loop count K from train.loops and one batch of train.batch inputs from every task (in the config's
task order, one random stream), runs the block K times on each batch, and sums the tasks' cross-entropies; one gradient
then updates the block and every task's parameters (AdamW with warm-up and cosine decay over train.updates, gradients
clipped to norm 1 over the whole tree). The block runs once per group of tasks whose inputs have the same number of
items, on their batches stacked; the values are those of separate runs.

Per-task settings (config 'task_model', 'task_sizes'): a task's own model keys (weight_scale, scale_by_n, read) and its
own training sizes. train.task_loops {task: [lo, hi]}: that task trains at lo + round(i (hi - lo) / (len(loops) - 1))
loops, i the index of the update's draw from train.loops (loops 4..8 and [8, 32]: 8, 14, 20, 26, 32), in its own group.
train.weight_averaging {from, every}: the running mean of the parameters at every update u >= from * updates with
u % every == 0 (and the last), saved beside the final parameters.

Writes to <out dir>: model.safetensors and config.json (the final parameters, with each task's scores at its registered
read), weight_averaged/ (the same for the mean, when asked), predictions.npz (per task and split: the registered read and
every loop) and progress.jsonl. The paper's shared block (configs/shared/units_dfs.json) trains 160,000 updates in about
13 hours on one NVIDIA B200.
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
from .model import census_loop, count_params, decide, embed, init_params, outputs_loss, readout
from .tasks import NAMES, OUTPUTS
from .train import decay_mask
from .weights import save, shared_task


def task_config(cfg, task):
    return dict(cfg['model'], **cfg.get('task_model', {}).get(task, {}))


def same(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        x.shape == y.shape and x.dtype == y.dtype and bool((np.asarray(x) == np.asarray(y)).all()) for x, y in zip(la, lb))


def build(seed, tasks, cfgs, outs, feats):
    """{'blk': the shared block, 'task': {t: own parameters}}; every task's initial block is checked to be the same."""
    key = jax.random.PRNGKey(seed)
    P = {'blk': None, 'task': {}}
    for t in tasks:
        p = init_params(key, cfgs[t], outs[t], feats[t])
        blk = p.pop('blk')
        if P['blk'] is None:
            P['blk'] = blk
        assert same(blk, P['blk']), f'{t}: its initial block differs from {tasks[0]}\'s'
        P['task'][t] = p
    return P


def view(P, t):
    """Task t's parameters in the one-task layout (the same arrays)."""
    return dict(P['task'][t], blk=P['blk'])


def task_loops(tc, i):
    """{task: loops} of an update whose draw from train.loops has index i (train.task_loops)."""
    n = len(tc['loops'])
    return {t: int(lo) + int(i * (int(hi) - int(lo)) / (n - 1) + .5) for t, (lo, hi) in tc.get('task_loops', {}).items()}


def grouped_logits(P, tasks, outs, xs, K, Kt):
    """Every task's logits after K loops (a task in Kt after its own loops). The block runs once per group of tasks with
    the same number of items (a task with its own loops forms its own group), on their batches stacked."""
    views = {t: view(P, t) for t in tasks}
    z = {t: embed(views[t], xs[t]) for t in tasks}
    groups = {}
    for t in tasks:
        groups.setdefault(z[t].shape[-1] if t not in Kt else (z[t].shape[-1], Kt[t]), []).append(t)
    for ts in groups.values():
        if len(ts) == 1:
            zz, mm = z[ts[0]], xs[ts[0]]['m']
        else:
            zz, mm = jnp.concatenate([z[t] for t in ts]), jnp.concatenate([xs[t]['m'] for t in ts])
        for _ in range(Kt.get(ts[0], K)):
            zz = census_loop(P['blk'], zz, mm)
        if len(ts) == 1:
            z[ts[0]] = zz
        else:
            for t, q in zip(ts, jnp.split(zz, np.cumsum([z[t].shape[0] for t in ts])[:-1])):
                z[t] = q
    return {t: {name: readout(views[t], z[t], xs[t]['m'], kind, name) for name, kind in outs[t]} for t in tasks}


def make_step(tx, tasks, outs, K, Kt):
    @jax.jit
    def step(P, S, xs, ys):
        def lossf(P):
            lg = grouped_logits(P, tasks, outs, xs, K, Kt)
            per = {t: outputs_loss(lg[t], xs[t]['m'], ys[t], outs[t]) for t in tasks}
            tot = per[tasks[0]]
            for t in tasks[1:]:
                tot = tot + per[t]
            return tot, per
        (v, per), g = jax.value_and_grad(lossf, has_aux=True)(P)
        u, S = tx.update(g, S, P)
        return optax.apply_updates(P, u), S, v, per
    return step


def make_read_loops(cfg, outs):
    """f(p, x, loops) -> per-loop predictions, one compiled loop at a time (a task read after n loops)."""
    first = jax.jit(lambda p, x: embed(p, x))

    @jax.jit
    def one(p, x, z):
        z = census_loop(p['blk'], z, x['m'])
        return z, {name: decide(readout(p, z, x['m'], kind, name), kind) for name, kind in outs}

    def f(p, x, loops):
        z, seq = first(p, x), []
        for _ in range(loops):
            z, pr = one(p, x, z)
            seq.append(pr)
        return seq
    return f


def predict(P, t, cfg, outs, d, kmax, batch, loop_fn=None):
    """{name: [loops, inputs, ...]} after every loop: kmax loops, or, for a task read after n loops, max(kmax, largest n)."""
    if loop_fn is None:
        return E.predict_loops(view(P, t), cfg, outs, d, kmax, batch)
    L = max(kmax, int(d['n'].max()))
    n, chunks = len(d['n']), []
    for i in range(0, n, batch):
        x = model_inputs(cfg, d, np.arange(i, min(i + batch, n)))
        chunks.append(loop_fn(view(P, t), {k: jnp.array(a) for k, a in x.items()}, L))
    return {name: np.concatenate([np.stack([np.asarray(r[name]) for r in c]) for c in chunks], 1) for name, _ in outs}


def registered(per, d, cfg):
    """The registered read of per-loop predictions: after n loops (read rule 'n') or after 8."""
    if cfg.get('read', {}).get('rule') == 'n':
        n = d['n'].astype(int)
        return {k: np.stack([v[n[i] - 1, i] for i in range(len(n))]) for k, v in per.items()}
    return {k: v[E.CENSUS_LOOPS - 1] for k, v in per.items()}


def main(config, data_dir, out_dir, seed=0):
    cfg = config if isinstance(config, dict) else json.loads(Path(config).read_text())
    if 'loop_rule' in cfg['train']:   # each input read at its own loop count, optionally with a second block
        from .train_shared_rule import main as rule_main
        return rule_main(cfg, data_dir, out_dir, seed)
    tasks, tc, seed = list(cfg['tasks']), cfg['train'], int(seed)
    assert len(tasks) > 1 and len(set(tasks)) == len(tasks), tasks
    assert cfg['model']['block'] == 'census', 'the shared trainer trains the general block'
    assert set(tc.get('task_loops', {})) <= set(tasks)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cfgs = {t: task_config(cfg, t) for t in tasks}
    outs = {t: OUTPUTS[t] for t in tasks}
    train, splits, feats, NP = {}, {}, {}, {}
    for t in tasks:
        data = Path(data_dir) / t
        parts = [load_split(data / f'train_L{L}.npz') for L in cfg.get('task_sizes', {}).get(t, tc['sizes'])]
        train[t] = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        splits[t] = {s: load_split(data / f'{s}.npz') for s in ('val', 'test') + tuple(tc.get('extra_splits', ()))}
        feats[t] = feature_counts(splits[t]['val'])
        NP[t] = len(train[t]['n'])
    B, steps, kmax, EB = tc['batch'], tc['updates'], tc.get('eval_loops', 16), tc.get('eval_batch', 16)
    sched = optax.warmup_cosine_decay_schedule(0., tc['lr'], tc['warmup'], steps, tc['lr'] * tc['final_lr_frac'])
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sched, weight_decay=tc['weight_decay'], mask=decay_mask))
    P = build(seed, tasks, cfgs, outs, feats)
    S = tx.init(P)
    steps_fn = {}
    swa = tc.get('weight_averaging')
    if swa:
        swa_from, swa_n = int(np.ceil(swa['from'] * steps)), 0
        A = jax.tree.map(jnp.zeros_like, P)

    rng = np.random.default_rng(seed + 100)
    t0 = time.time()
    log = (dest / 'progress.jsonl').open('a')
    for it in range(1, steps + 1):
        i = int(rng.integers(len(tc['loops'])))
        K, Kt = int(tc['loops'][i]), task_loops(tc, i)
        xs, ys = {}, {}
        for t in tasks:
            idx = rng.integers(NP[t], size=B)
            xs[t] = {k: jnp.array(a) for k, a in model_inputs(cfgs[t], train[t], idx, rng).items()}
            ys[t] = {k: jnp.array(a) for k, a in targets(train[t], idx, outs[t]).items()}
        if K not in steps_fn:
            steps_fn[K] = make_step(tx, tasks, outs, K, Kt)
        P, S, v, per = steps_fn[K](P, S, xs, ys)
        if swa and it >= swa_from and (it % swa['every'] == 0 or it == steps):
            swa_n += 1
            A = jax.tree.map(lambda a, w: a + (w - a) / swa_n, A, P)
        if it % tc.get('check_every', 500) == 0 or it == steps:
            row = dict(seed=seed, step=it, loss=float(v), K=K, sec=round(time.time() - t0, 1), tasks={t: float(per[t]) for t in tasks})
            log.write(json.dumps(row) + '\n'); log.flush()
            if not np.isfinite(float(v)):
                break
    log.close()
    train_sec = round(time.time() - t0, 1)

    loop_fn = {t: make_read_loops(cfgs[t], outs[t]) for t in tasks if cfgs[t].get('read', {}).get('rule') == 'n'}
    finals = [('', P)] + ([('weight_averaged', A)] if swa and swa_n else [])
    result = {}
    for sub, Q in finals:
        where = dest / sub if sub else dest
        where.mkdir(parents=True, exist_ok=True)
        arrays, tasks_cfg = {}, {}
        for t in tasks:
            scores = {}
            for sp, d in splits[t].items():
                per_loop = predict(Q, t, cfgs[t], outs[t], d, kmax, EB, loop_fn.get(t))
                read = registered(per_loop, d, cfgs[t])
                arrays |= {f'{t}/{sp}/per_loop/{k}': v.astype(np.int16) for k, v in per_loop.items()}
                arrays |= {f'{t}/{sp}/read/{k}': v.astype(np.int16) for k, v in read.items()}
                scores[sp] = round(100 * E.score(read, d, outs[t]), 4)
            tasks_cfg[t] = dict(task_name=NAMES[t], outputs=[dict(name=n, kind=k) for n, k in outs[t]], features=feats[t],
                                model=cfg.get('task_model', {}).get(t, {}), train_sizes=cfg.get('task_sizes', {}).get(t, tc['sizes']),
                                parameters_own=count_params(Q['task'][t]), scores=scores)
        np.savez_compressed(where / 'predictions.npz', **arrays)
        out = dict(name=cfg.get('name', 'shared') + (f'/{sub}' if sub else ''), block=cfg['model']['block'], model=cfg['model'],
                   read=dict(rule='fixed', loops=E.CENSUS_LOOPS), parameters_shared=count_params(Q['blk']),
                   parameters_total=count_params(Q), train=dict(tc, seed=seed, steps_done=it, train_seconds=train_sec,
                   **({'snapshots': swa_n} if sub else {})), tasks=tasks_cfg)
        save(where, Q, out)
        result[sub or 'final'] = {t: tasks_cfg[t]['scores'] for t in tasks}
    return result


if __name__ == '__main__':
    print(json.dumps(main(*sys.argv[1:])))
