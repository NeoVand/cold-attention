"""Max Retrieval with cold attention: the released networks, their test sets, their reads and their training (jax 0.8.0).

The task is the one of Velickovic et al. 2025, "Softmax is not enough (for sharp size generalisation)", arXiv 2410.01104,
Sec. 4.1 and App. A. A set holds n items. Item i has a priority rho_i, uniform in [0, 1), and one of 10 classes k_i; a query
q, uniform in [0, 1), comes with the set and does not change the answer. The answer is the class of the item of highest
priority. Every network here is trained on sets of 5 to 16 items and tested on sets of 16 to 16,384.

Three networks. All start from Flax's default initialisers (lecun_normal weights, zero biases). Parameter names are the ones
stored in model.safetensors; suffix w is a weight matrix, b its bias.

  softmax      the network of the paper above (its App. A.3), in plain jax:
                 h_i = gelu(x2(gelu(x1(x_i))))           x_i = (rho_i, onehot k_i); 128 features
                 u   = q2(gelu(q1(q)))
                 e_i = qp(u) . kp(h_i) / sqrt(128)       one attention head (computed as h_i . (kp_w qp(u)) / sqrt(128) +
                                                         kp_b . qp(u) / sqrt(128), the same function)
                 y   = r2(gelu(r1(op(vp(sum_i a_i h_i)))))    a = softmax(e)
  cold_count   priorities enter only as comparisons, b_ij = 1[rho_j > rho_i] (rival j beats item i); rho is never embedded:
                 t_ij = t2(gelu(t1(b_ij, onehot k_i, onehot k_j)))    a learned test of rival j for item i (21 -> 32 -> 1)
                 c_i  = sum over j != i of 1[t_ij > 0]                 the count of admitted rivals
                 w    = 1 on the items of smallest count, shared evenly over ties     (zero temperature)
                 y    = r2(gelu(r1(op(vp(sum_i w_i h_i)))))            h_i = gelu(x2(gelu(x1(onehot k_i))))
               Trained straight-through: the forward pass is exactly the hard test and pick; the backward pass uses the
               gradients of sigmoid(t) and softmax(-c).
  soft_count   cold_count's input and layers, warm: a_ij = sigmoid(t_ij), c_i = sum over j != i of a_ij,
               w = softmax(-beta c) with a learned beta (starts at 1); trained end to end like the softmax network.

Reads (READS):
  softmax      'softmax' as trained; 'cold': all weight on the keys of highest score e_i, split evenly over exact float32 ties.
  cold_count   'cold'.
  soft_count   'warm' as trained; 'cold' from the same weights: the test 1[t > 0], the exact count, the smallest count.
The learned test sees only (b, k_i, k_j), so its hard form is a 2 x 10 x 10 table and its soft form sigmoid(t) a table of the
same shape. At test the counts come from the table and one sort per set, O(n log n) per set, the same counts as the pairwise
layer (check_table compares the two).

Test sets (test_keys): for size n, 10,240 sets, drawn in chunks of m sets; chunk c from fold_in(PRNGKey(2**20 + n), c), where
m is the largest divisor of 10,240 that is at most 2**20 // n. The same sets for every network and seed. They were drawn with
jax 0.8.0 on CPU; another version of jax may draw other sets (test_sets.npz holds every answer, to check).

Training (train): 100,000 Adam steps, learning rate 1e-3, batches of 128 sets, one set size per batch, uniform in 5..16
(numpy default_rng(seed)); cross-entropy + l2_scale * 1e-3 * (sum of squared parameters, every parameter); init key and data
key = split(PRNGKey(seed)); batch t from fold_in(data key, t). cold_count stops early once the training loss is at its floor
(five 1,000-step windows in a row with no training error and mean losses within 2% of each other); on all ten seeds it never
did, and every model ran 100,000 steps.

  python max_retrieval.py demo [cold_count/seed0] [--sizes 16,1024]       (downloads the model; a CPU is enough)

Apache 2.0.
"""
import json
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

REPO = 'mohsenvand/cold-attention-max-retrieval'
D, C, B, H, LR, LAM = 128, 10, 128, 32, 1e-3, 1e-3
TEST_KEY_BASE = 2 ** 20
N_SETS, CHUNK_ITEMS = 10240, 2 ** 20
SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
NETWORKS = ('softmax', 'cold_count', 'soft_count')
READS = {'softmax': ('softmax', 'cold'), 'cold_count': ('cold',), 'soft_count': ('warm', 'cold')}
gelu = partial(jax.nn.gelu, approximate=True)   # flax.linen.gelu's default


# ========================================================================================================== the task
def make_batch(key, m, n):
    """m sets of n items: x (m, n, 11) = (rho, onehot class); q (m, 1); answer y (m,); rho (m, n); index of the top item."""
    k1, k2, k3 = jax.random.split(key, 3)
    rho = jax.random.uniform(k1, (m, n))
    cls = jax.random.randint(k2, (m, n), 0, C)
    q = jax.random.uniform(k3, (m, 1))
    x = jnp.concatenate([rho[..., None], jax.nn.one_hot(cls, C)], -1)
    ti = rho.argmax(-1)
    y = jnp.take_along_axis(cls, ti[:, None], 1)[:, 0]
    return x, q, y, rho, ti


def test_keys(n, n_sets=N_SETS, chunk_items=CHUNK_ITEMS):
    """The chunk size m and the key of every chunk of the test sets of size n."""
    m = max(1, min(n_sets, chunk_items // n))
    while n_sets % m:
        m -= 1
    return m, [jax.random.fold_in(jax.random.PRNGKey(TEST_KEY_BASE + n), c) for c in range(n_sets // m)]


def classes(x):
    return x[..., 1:].argmax(-1)


# ======================================================================================================= the networks
SHAPES = {'softmax': dict(x1=(C + 1, D), x2=(D, D), q1=(1, D), q2=(D, D), qp=(D, D), kp=(D, D), vp=(D, D), op=(D, D), r1=(D, D), r2=(D, C)),
          'cold_count': dict(t1=(1 + 2 * C, H), t2=(H, 1), x1=(C, D), x2=(D, D), vp=(D, D), op=(D, D), r1=(D, D), r2=(D, C))}
SHAPES['soft_count'] = SHAPES['cold_count']


def init_params(network, key):
    shapes = SHAPES[network]
    init = jax.nn.initializers.lecun_normal()
    p = {}
    for k, (name, s) in zip(jax.random.split(key, len(shapes)), shapes.items()):
        p[name + 'w'] = init(k, s, jnp.float32)
        p[name + 'b'] = jnp.zeros((s[1],), jnp.float32)
    if network == 'soft_count':
        p['beta'] = jnp.array(1.0, jnp.float32)
    return p


def head(p, hbar):   # value and output projections, then the read-out (10 class logits)
    v = hbar @ p['vpw'] + p['vpb']
    o = v @ p['opw'] + p['opb']
    return gelu(o @ p['r1w'] + p['r1b']) @ p['r2w'] + p['r2b']


# ---- softmax
def embed(p, x, q):   # item features h (m, n, 128) and key scores e (m, n)
    h = gelu(gelu(x @ p['x1w'] + p['x1b']) @ p['x2w'] + p['x2b'])
    qq = gelu(q @ p['q1w'] + p['q1b']) @ p['q2w'] + p['q2b']
    qv = qq @ p['qpw'] + p['qpb']
    w = (qv @ p['kpw'].T) / math.sqrt(D)
    c = (qv @ p['kpb']) / math.sqrt(D)
    e = jnp.einsum('bnd,bd->bn', h, w) + c[:, None]
    return h, e


# ---- cold_count and soft_count
def test_logit(p, b, oi, oj):
    return (gelu(jnp.concatenate([b[..., None], oi, oj], -1) @ p['t1w'] + p['t1b']) @ p['t2w'] + p['t2b'])[..., 0]


def embed_class(p, oh):
    return gelu(gelu(oh @ p['x1w'] + p['x1b']) @ p['x2w'] + p['x2b'])


def st(hard, soft):   # straight-through: the forward value is exactly hard (counts stay integers)
    return hard + (soft - jax.lax.stop_gradient(soft))


def pair_counts(p, rho, oh):   # the pairwise cold layer: rho (m, n), oh (m, n, 10) -> counts of admitted rivals (m, n)
    n = rho.shape[1]
    b = (rho[:, None, :] > rho[:, :, None]).astype(jnp.float32)   # b[i, j] = 1[rho_j > rho_i]
    t = test_logit(p, b, jnp.broadcast_to(oh[:, :, None, :], b.shape + (C,)), jnp.broadcast_to(oh[:, None, :, :], b.shape + (C,)))
    a = st((t > 0).astype(jnp.float32), jax.nn.sigmoid(t)) * (1 - jnp.eye(n))
    return a.sum(-1)


def pick(cnt):   # zero temperature: weight on the smallest count, shared over ties; softmax(-c) gradient backward
    hard = (cnt == cnt.min(-1, keepdims=True)).astype(jnp.float32)
    return st(hard / hard.sum(-1, keepdims=True), jax.nn.softmax(-cnt, -1))


def soft_counts(p, rho, oh):   # the pairwise warm layer of soft_count
    n = rho.shape[1]
    b = (rho[:, None, :] > rho[:, :, None]).astype(jnp.float32)
    t = test_logit(p, b, jnp.broadcast_to(oh[:, :, None, :], b.shape + (C,)), jnp.broadcast_to(oh[:, None, :, :], b.shape + (C,)))
    return (jax.nn.sigmoid(t) * (1 - jnp.eye(n))).sum(-1)


def truth_table(p):
    """The learned test on all 200 inputs: T[b, k_i, k_j] = 1[t > 0] (int32) and the logits t."""
    e = jnp.eye(C)
    bb, ii, jj = jnp.meshgrid(jnp.arange(2.0), jnp.arange(C), jnp.arange(C), indexing='ij')
    t = test_logit(p, bb, e[ii], e[jj])
    return (t > 0).astype(jnp.int32), t


def table_counts(T, rho, cls):
    """Counts from a 2 x 10 x 10 table (hard T or soft sigmoid(t)), in ascending-priority order, from one sort per set:
    c_i = sum over classes k' of T[1, k_i, k'] #{j: k_j = k', rho_j > rho_i} + T[0, k_i, k'] #{j != i: k_j = k', rho_j <= rho_i}."""
    o = jnp.argsort(rho, -1)
    rs, cs = jnp.take_along_axis(rho, o, -1), jnp.take_along_axis(cls, o, -1)
    oh = jax.nn.one_hot(cs, C, dtype=jnp.int32)
    cum = jnp.cumsum(oh, 1)
    idx = jax.vmap(lambda a: jnp.searchsorted(a, a, side='right'))(rs) - 1   # last index with rho <= rho_i
    le = jnp.take_along_axis(cum, idx[..., None], 1)
    gt = cum[:, -1:, :] - le
    cnt = (T[1][cs] * gt).sum(-1) + (T[0][cs] * (le - oh)).sum(-1)
    return cnt, o, rs, cs, oh


def tables(p, network):
    """The tables a read uses: the hard table T, and for soft_count also the soft table sigmoid(t)."""
    T, tl = truth_table(p)
    return (T, jax.nn.sigmoid(tl)) if network == 'soft_count' else (T,)


# =========================================================================================================== the reads
@partial(jax.jit, static_argnums=(2, 3))
def _chunk_softmax(p, key, n, m):
    x, q, y, rho, ti = make_batch(key, m, n)
    h, e = embed(p, x, q)
    soft = jax.nn.softmax(e, -1)
    top = e == e.max(-1, keepdims=True)
    cnt = top.sum(-1)
    hard = top.astype(e.dtype) / cnt[:, None]
    ys = head(p, jnp.einsum('bn,bnd->bd', soft, h)).argmax(-1)
    yh = head(p, jnp.einsum('bn,bnd->bd', hard, h)).argmax(-1)
    dup = (rho == jnp.take_along_axis(rho, ti[:, None], 1)).sum(-1) > 1
    return dict(softmax=ys, cold=yh, answer=y, top_duplicated=dup)


@partial(jax.jit, static_argnums=(3, 4))
def _chunk_cold_count(p, T, key, n, m):
    x, q, y, rho, ti = make_batch(key, m, n)
    cnt, o, rs, cs, oh = table_counts(T, rho, classes(x))
    mn = cnt.min(-1, keepdims=True); top = cnt == mn; nmin = top.sum(-1)
    w = (top[..., None] * oh).sum(1).astype(jnp.float32) / nmin[:, None]   # class mixture of the picked items
    pred = head(p, w @ embed_class(p, jnp.eye(C))).argmax(-1)
    return dict(cold=pred, answer=y, top_duplicated=(rs == rs[:, -1:]).sum(-1) > 1)


@partial(jax.jit, static_argnums=(4, 5))
def _chunk_soft_count(p, T, Sg, key, n, m):
    x, q, y, rho, ti = make_batch(key, m, n)
    cls = classes(x)
    cs_, o, rs, cs, oh = table_counts(Sg, rho, cls)   # soft counts
    ch, *_ = table_counts(T, rho, cls)                # exact hard counts
    Hc = embed_class(p, jnp.eye(C))
    a = jax.nn.softmax(-p['beta'] * cs_, -1)
    pw = head(p, (a[..., None] * oh).sum(1) @ Hc).argmax(-1)
    mn = ch.min(-1, keepdims=True); top = ch == mn; nmin = top.sum(-1)
    pc = head(p, ((top[..., None] * oh).sum(1).astype(jnp.float32) / nmin[:, None]) @ Hc).argmax(-1)
    return dict(warm=pw, cold=pc, answer=y, top_duplicated=(rs == rs[:, -1:]).sum(-1) > 1)


def read(params, network, n, n_sets=N_SETS, chunk_items=CHUNK_ITEMS):
    """Every read of the network on the test sets of size n: {read: predicted class (n_sets,), 'answer': ..., 'top_duplicated': ...}."""
    p = {k: jnp.asarray(v) for k, v in params.items()}
    m, keys = test_keys(n, n_sets, chunk_items)
    if network == 'softmax':
        f = lambda k: _chunk_softmax(p, k, n, m)
    elif network == 'cold_count':
        T, = tables(p, network); f = lambda k: _chunk_cold_count(p, T, k, n, m)
    else:
        T, Sg = tables(p, network); f = lambda k: _chunk_soft_count(p, T, Sg, k, n, m)
    parts = [jax.device_get(f(k)) for k in keys]
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}


def score(out, network):
    """Share of test sets answered right (%) by each read."""
    return {r: 100 * float((out[r] == out['answer']).mean()) for r in READS[network]}


@partial(jax.jit, static_argnums=(2, 3))
def _check_chunk(p, T, n, m):
    x, q, y, rho, ti = make_batch(jax.random.PRNGKey(7), m, n)
    pw = pair_counts(p, rho, x[..., 1:])
    cnt, o, *_ = table_counts(T, rho, classes(x))
    return jnp.abs(jnp.take_along_axis(pw, o, -1) - cnt).max()


def check_table(params, network, n=64, m=64):
    """Largest difference between the pairwise layer's counts and the table's, on m sets of n items (0 for cold_count)."""
    p = {k: jnp.asarray(v) for k, v in params.items()}
    T = tables(p, network)[-1]
    if network == 'soft_count':
        x, q, y, rho, ti = make_batch(jax.random.PRNGKey(7), m, n)
        cnt, o, *_ = table_counts(T, rho, classes(x))
        return float(jnp.abs(jnp.take_along_axis(soft_counts(p, rho, x[..., 1:]), o, -1) - cnt).max())
    return float(_check_chunk(p, T, n, m))


# ========================================================================================================== training
def _l2(p):
    return sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(p))


def _loss_softmax(p, x, q, y, l2s):
    h, e = embed(p, x, q)
    logits = head(p, jnp.einsum('bn,bnd->bd', jax.nn.softmax(e, -1), h))
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return ce + l2s * LAM * _l2(p), (ce, (logits.argmax(-1) == y).mean())


def _loss_cold_count(p, x, y, l2s):
    rho, oh = x[..., 0], x[..., 1:]
    w = pick(pair_counts(p, rho, oh))
    logits = head(p, jnp.einsum('bn,bnd->bd', w, embed_class(p, oh)))
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return ce + l2s * LAM * _l2(p), (ce, (logits.argmax(-1) == y).mean())


def _loss_soft_count(p, x, y, l2s):
    rho, oh = x[..., 0], x[..., 1:]
    w = jax.nn.softmax(-p['beta'] * soft_counts(p, rho, oh), -1)
    logits = head(p, jnp.einsum('bn,bnd->bd', w, embed_class(p, oh)))
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return ce + l2s * LAM * _l2(p), (ce, (logits.argmax(-1) == y).mean())


def _make_step(network):
    opt = optax.adam(LR)

    @partial(jax.jit, static_argnums=(3, 4), donate_argnums=(0, 1))
    def step(p, s, key, n, l2s):
        x, q, y, _, _ = make_batch(key, B, n)
        if network == 'softmax':
            (l, (ce, acc)), g = jax.value_and_grad(_loss_softmax, has_aux=True)(p, x, q, y, l2s)
        elif network == 'cold_count':
            (l, (ce, acc)), g = jax.value_and_grad(_loss_cold_count, has_aux=True)(p, x, y, l2s)
        else:
            (l, (ce, acc)), g = jax.value_and_grad(_loss_soft_count, has_aux=True)(p, x, y, l2s)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l, ce, acc
    return opt, step


def train(network, seed, l2_scale, steps=100000, every=1000, log=None):
    """Train one network from scratch; returns (params, history of 1,000-step windows, steps run)."""
    opt, step = _make_step(network)
    k_init, k_data = jax.random.split(jax.random.PRNGKey(seed))
    p = init_params(network, k_init); s = opt.init(p)
    sizes = np.random.default_rng(seed).integers(5, 17, steps)
    acc_l = acc_ce = acc_a = 0.0; hist = []; ran = steps
    for t in range(steps):
        p, s, l, ce, a = step(p, s, jax.random.fold_in(k_data, t), int(sizes[t]), float(l2_scale))
        acc_l, acc_ce, acc_a = acc_l + l, acc_ce + ce, acc_a + a
        if (t + 1) % every == 0 or t + 1 == steps:
            k = (t % every) + 1
            rec = dict(step=t + 1, loss=float(acc_l) / k, ce=float(acc_ce) / k, acc=float(acc_a) / k)
            hist.append(rec); acc_l = acc_ce = acc_a = 0.0
            if log:
                log(rec)
            last = hist[-5:]
            if network == 'cold_count' and len(last) == 5 and all(h['acc'] == 1.0 for h in last) and \
                    (max(h['loss'] for h in last) - min(h['loss'] for h in last)) <= 0.02 * min(h['loss'] for h in last):
                ran = t + 1
                break
    return jax.device_get(p), hist, ran


# ========================================================================================================= loading
def load(folder):
    """params (dict of numpy arrays) and config of one released model folder."""
    from safetensors.numpy import load_file
    folder = Path(folder)
    return load_file(str(folder / 'model.safetensors')), json.loads((folder / 'config.json').read_text())


def load_hub(name, repo=REPO):
    """Download one model, e.g. load_hub('cold_count/seed0')."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, allow_patterns=[f'{name}/*'])
    return load(Path(root) / name)


def demo(name='cold_count/seed0', sizes=(16, 1024)):
    params, cfg = load(name) if Path(name, 'config.json').exists() else load_hub(name)
    for n in sizes:
        out = read(params, cfg['network'], n)
        print(f"{cfg['name']}  n = {n:>6,}  " + '  '.join(f'{r}: {v:.2f}%' for r, v in score(out, cfg['network']).items()) +
              f"   (recorded: " + ', '.join(f"{r} {cfg['scores'][r][str(n)]:.2f}" for r in READS[cfg['network']]) + ')')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['demo']:
        a = [x for x in args[1:] if not x.startswith('--')]
        sz = (16, 1024)
        if '--sizes' in args:
            sz = tuple(int(v) for v in args[args.index('--sizes') + 1].split(','))
            a = [x for x in a if x != args[args.index('--sizes') + 1]]
        demo(*(a[:1] or ['cold_count/seed0']), sizes=sz)
    else:
        print(__doc__)
