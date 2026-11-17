"""Prediction, the registered read-out rule, and a copy of the library's scorer.

The read-out rule (fixed before any test score is seen):
  census          the prediction after 8 loops (a task of a shared block may name its own rule: 'n', each input read
                  after its own n loops, as depth-first search in the shared block with deadline units).
  cold            the prediction at the first loop whose bits equal the previous loop's (cap: n loops for n items).
  compare_first   the same.
  pair_numbers    the same, with the built numbers (R, and H) also repeating (cap: 4 n loops).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .data import model_inputs
from .model import decide, forward, make_rest_read, read_all, start, step
from .scoring import score  # noqa: F401 (re-exported)

CENSUS_LOOPS = 8


def _jx(x):
    return {k: _jx(v) if isinstance(v, dict) else jnp.array(v) for k, v in x.items()}


def _per_loop(cfg, outs, loops):
    """The start and one loop each compiled on their own, the loop called once per loop: the two-block models, and a model
    whose read says compile 'per_loop' (each read as when it was trained; one compiled call over all loops rounds a few near
    ties differently)."""
    st = jax.jit(lambda p, x: start(p, cfg, x))

    @jax.jit
    def one(p, x, e, s):
        s = step(p, cfg, x, e, s)
        lg = read_all(p, cfg, x, e, s, outs)
        return s, {name: decide(lg[name], kind) for name, kind in outs}

    def f(p, x):
        e, s = st(p, x)
        seq = []
        for _ in range(loops):
            s, pr = one(p, x, e, s)
            seq.append(pr)
        return seq
    return f


def predict_loops(p, cfg, outs, d, loops=16, batch=16):
    """{name: [loops, samples, ...]}: the predicted indices after each loop 1..loops, in batches."""
    one_loop_at_a_time = cfg['block'] == 'two_blocks' or cfg.get('read', {}).get('compile') == 'per_loop'
    f = _per_loop(cfg, outs, loops) if one_loop_at_a_time else jax.jit(lambda p, x: forward(p, cfg, x, loops, outs, every=True))
    n = len(d['n'])
    chunks = [f(p, _jx(model_inputs(cfg, d, np.arange(i, min(i + batch, n))))) for i in range(0, n, batch)]
    return {name: np.concatenate([np.stack([np.asarray(r[name]) for r in c]) for c in chunks], 1) for name, _ in outs}


def predict_rest(p, cfg, outs, d, batch=16, read=None):
    """The resting blocks' registered read on a split -> dict(pred {name: [samples, ...]}, loops [samples],
    stable [samples], per_loop {name: [L, samples, ...]}) with per_loop padded to the split's cap by repeating an input's
    last loop (exact for an input at rest)."""
    read = read or make_rest_read(cfg, outs)
    n = len(d['n'])
    L = (4 if cfg['block'] in ('pair_numbers', 'dp_program') else 1) * int(d['n'].max())
    acc = []
    for i in range(0, n, batch):
        r = read(p, _jx(model_inputs(cfg, d, np.arange(i, min(i + batch, n)))))
        ran = r['changed'].shape[0]
        pad = lambda a: a if ran == L else np.concatenate([a, np.repeat(a[-1:], L - ran, 0)])
        acc.append(dict(per={k: pad(v) for k, v in r['per_loop'].items()}, pred=r['pred'], loops=r['loops'], stable=r['stable']))
    cat = lambda key, ax=0: {k: np.concatenate([a[key][k] for a in acc], ax) for k in acc[0][key]}
    return dict(pred=cat('pred'), per_loop=cat('per', 1), loops=np.concatenate([a['loops'] for a in acc]),
                stable=np.concatenate([a['stable'] for a in acc]))


def log2_loops(n, plus):
    """The per-input loop count ceil(log2 n) + plus, for inputs of n items."""
    return np.array([int(q - 1).bit_length() + plus for q in np.ravel(n)])


def predict_log2(p, cfg, outs, d, batch=16):
    """Inputs each read after their own ceil(log2 n) + plus loops (cfg['read'] = {'rule': 'log2', 'plus': ...}) -> pred,
    loops, per_loop {name: [max loops, samples, ...]}."""
    K = log2_loops(d['n'], cfg['read']['plus'])
    per = predict_loops(p, cfg, outs, d, int(K.max()), batch)
    i = np.arange(len(K))
    return {k: v[K - 1, i] for k, v in per.items()}, K, per


def predict_n(p, cfg, outs, d, batch=16):
    """Inputs each read after their own n loops (n items; cfg['read'] = {'rule': 'n'}) -> pred, loops, per_loop."""
    K = np.ravel(d['n']).astype(int)
    per = predict_loops(p, cfg, outs, d, int(K.max()), batch)
    i = np.arange(len(K))
    return {k: v[K - 1, i] for k, v in per.items()}, K, per


def registered_read(p, cfg, outs, d, batch=16):
    """{name: [samples, ...]}: the prediction the paper scores, by the registered rule; and the loops used per input."""
    if cfg.get('read', {}).get('rule') == 'log2':
        pred, K, _ = predict_log2(p, cfg, outs, d, batch)
        return pred, K
    if cfg.get('read', {}).get('rule') == 'n':
        pred, K, _ = predict_n(p, cfg, outs, d, batch)
        return pred, K
    if cfg['block'] == 'census':
        per = predict_loops(p, cfg, outs, d, CENSUS_LOOPS, batch)
        return {k: v[-1] for k, v in per.items()}, np.full(len(d['n']), CENSUS_LOOPS)
    r = predict_rest(p, cfg, outs, d, batch)
    return r['pred'], r['loops']


def score_ties(pred, d, outs):
    """The tie-aware score (item_memory's registered score): as score, but an argmax entry is right only if its largest logit
    is unique (pred['<name>#unique']) and at the right index. Mask outputs are scored as score scores them."""
    sc = []
    for name, kind in outs:
        if kind in ('nmask', 'emask', 'gmask'):
            sc.append(score({name: pred[name]}, d, [(name, kind)])); continue
        t, q, u = d[name], pred[name], pred[name + '#unique'] > 0
        if kind == 'node':
            sc.append(float(np.mean((q[:, :t.shape[1]] == t) & u[:, :t.shape[1]])))
        elif kind == 'ecat':
            ok = t >= 0; sc.append(float(((q == t) & u)[ok].sum() / ok.sum()))
        else:
            sc.append(float(np.mean((q == t) & u)))
    return float(np.mean(sc))


def score_loops(per, d, outs):
    """score after each loop: per {name: [L, samples, ...]} -> [L] scores."""
    L = next(iter(per.values())).shape[0]
    return [score({k: v[K] for k, v in per.items()}, d, outs) for K in range(L)]
