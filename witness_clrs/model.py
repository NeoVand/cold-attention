"""One looped pair block, in four variants.

THE STATE. Every ordered pair of items (u, v) carries a vector z(u, v) of D = 128 numbers. Arrays are channel-first:
z has shape [B, D, N, N] for B inputs of N items (padded items are masked by m [B, N], 1 for a real item).

ONE LOOP. The block reads the normalised state t = norm(z) through two learned linear maps, giving every channel c a
left score xa_c(u, k) and a right score xc_c(k, v). A middle item k is a WITNESS for the pair (u, v) in channel c when it
passes both yes/no tests, xa_c(u, k) > 0 and xc_c(k, v) > 0. The two halves of the channels then read the witnesses cold
(exactly, with no temperature):

  count half (channels 0..63)   channel pairs (2i, 2i+1) count their witnesses exactly, N_2i(u, v) and N_2i+1(u, v),
                                and hand on only the comparison: more = [N_2i > N_2i+1] and same = [N_2i = N_2i+1].
  best half  (channels 64..127) the best witness: cost(k) = max(x_a, x_c) + theta * min(x_a, x_c) with x = -log sigmoid
                                of the score, and the channel hands on exp(-min_k cost(k)), the score of the best witness.

Forward the tests are hard (0 or 1, so the counts are whole numbers); backward they pass the slope of a sigmoid
(straight-through). The four variants differ in how numbers enter and how the state is updated:

  census          numbers enter through the learned input embedding. The loop adds a gated linear map of the 128
                  outputs to the state, then an MLP. Read after a fixed 8 loops.
  cold            the plain cold block (no numbers in the input). Both halves hand on bits (the count half more / same;
                  the best half [some witness passes both tests]) and the state is rebuilt every loop from the input
                  embedding and these bits, so when the bits repeat every later loop repeats. Read when the bits stop
                  changing, or after n loops for n items.
  compare_first   numbers never enter the state. Each input number rides along in a register; a count-half channel pair
                  that carries a number compares the numbers of its admitted witnesses (the lowest value and how many
                  reach it) and hands on only more / same. The best half hands on a bit, [some witness passes both
                  tests], and a bit once on stays on. The state is rebuilt every loop from the input embedding and these
                  bits, so when the bits repeat, every later loop repeats: the block is read when its bits stop changing.
  pair_numbers    compare_first on graphs, plus one number per pair that the block builds and keeps, R(u, v): a best
                  path sum ('plus': shortest paths) or a best bottleneck ('max': spanning trees), grown one edge at a time
                  over the witnesses the state admits. The state receives only comparisons of R and of the edge weights.
                  Read when the bits and R stop changing. Two more register tables: 'via' (Floyd-Warshall: two pairs
                  compare R(u, k) + W(k, v) with R(u, v) per admitted k, and a pair's pointer to a third item adds a
                  learned gain where they are equal) and 'hops' (Prim: a second built number H, the hops along edges
                  that are their own best path, with its own grow and comparison pairs); self_loops: a weighted loop is
                  an edge.

Parameter names follow one fixed layout (in the safetensors files: 'blk/a', 'in30/e1', 'one30/min', ...).
"""
import numpy as np
import jax
import jax.numpy as jnp

from .tasks import MASKS, ECAT

jax.config.update('jax_default_matmul_precision', 'highest')  # no TF32: pointer targets are argmins of sums of weights

NE, NF = 7, 2                # graph tasks: pair features (A, A^T, [A != 0] both ways, adj both ways, [u = v]); node features (pos, source)
PREG_BIG, NPREG, PAIR_NUMBER_CAP = 1e30, 8, 4   # pair_numbers: 'no edge'; register channel pairs (the last 8); read cap, loops per item
PREG_TABLE = ('grow', 'edge', 'fwd', 'bwd', 'fwd', 'bwd', 'fwd', 'bwd')
REG_TABLES = {'paths': PREG_TABLE,
              'via': ('grow', 'edge', 'via', 'via', 'fwd', 'bwd', 'fwd', 'bwd'),
              'hops': ('grow', 'edge', 'hgrow', 'hbwd', 'hfwd', 'hbwd', 'fwd', 'bwd')}
VIA_GAIN = 8.                # 'via' table: the starting read-out gain of each via pair (not weight-decayed)


# ------------------------------------------------------------------------------------------------------------ helpers
def norm(t):
    """Normalise every pair's vector over its D channels."""
    return (t - t.mean(1, keepdims=True)) * jax.lax.rsqrt(t.var(1, keepdims=True) + 1e-6)


def lin(t, w, b=None):
    """A linear map over channels, applied to every pair: [B, C, N, N] -> [B, D, N, N]."""
    y = jnp.einsum('bcij,cd->bdij', t, w)
    return y if b is None else y + b[None, :, None, None]


def soft_clip(x, lo, hi):
    """clip(x, lo, hi) in value; the gradient passes as if there were no clip."""
    return x + jax.lax.stop_gradient(jnp.clip(x, lo, hi) - x)


def st_hard(hard, smooth):
    """The hard value forward, the smooth one's slope backward."""
    return hard.astype(smooth.dtype) + (smooth - jax.lax.stop_gradient(smooth))


@jax.custom_vjp
def only_if_wrong(s, hard):
    """s forward; backward the slope passes only where it would change the bit `hard`."""
    return s


only_if_wrong.defvjp(lambda s, hard: (s, hard),
                     lambda hard, ct: (jnp.where(hard > 0, jnp.maximum(ct, 0.), jnp.minimum(ct, 0.)), jnp.zeros_like(hard)))


def half(b):
    """Number of channels in each half (D / 2)."""
    return b['th_t'].shape[0]


def theta(b):
    """The best half's mixing weights, one per channel, in [0, 1]."""
    return soft_clip(b['th_t'], 0., 1.)


def mlp_update(b, z):
    return z + lin(jax.nn.gelu(lin(norm(z), b['f1'], b['bf1'])), b['f2'], b['bf2'])


# ------------------------------------------------------------------------------------------------------------ parameters
def init_params(key, cfg, outs, feats=None):
    """Initial parameters. cfg: 'block', 'width' (128), 'mlp' (312), 'edge_mlp' (32); feats: None for a graph task, else the
    input feature counts F (per item) and G (per input). Keys are drawn in a fixed order, so a seed gives the paper's initial
    weights; draws for heads a task does not use are made and discarded."""
    D, M, W = cfg['width'], cfg['mlp'], cfg['edge_mlp']
    ks = iter(jax.random.split(key, 32))
    g = lambda *s: jax.random.normal(next(ks), s) * s[0] ** -.5
    z = lambda *s: jnp.zeros(s)
    p = dict(e1=g(NE, W), eb1=z(W), e2=g(W, D), E=g(NE, D), eb=z(D), R=g(NF, D), C=g(NF, D))
    for _ in range(7):
        next(ks)
    ka = iter(jax.random.split(jax.random.fold_in(key, 20), 8))
    ga = lambda *s: jax.random.normal(next(ka), s) * s[0] ** -.5
    p['blk'] = dict(a=ga(D, D), b=ga(D, D), g=ga(D, D), o=ga(D, D), f1=ga(D, M), f2=ga(M, D),
                    ba=z(D), bb=z(D), bg=z(D), bo=z(D), bf1=z(M), bf2=z(D), th_t=jnp.linspace(0., 1., D - D // 2))
    if cfg['block'] != 'census':  # the rebuilt state has no gate
        del p['blk']['g'], p['blk']['bg']
    if cfg.get('bilinear_tests'):  # bilinear witness tests on the first count channels (census block), their own key
        cb, rb = int(cfg['bilinear_tests']), int(cfg.get('bilinear_rank', 2))
        assert cfg['block'] == 'census' and cb % 2 == 0 and 0 < cb <= D // 2
        kb = jax.random.split(jax.random.fold_in(key, 47), 2)
        p['blk'] |= dict(bl_p=jax.random.normal(kb[0], (D, cb, rb)) * D ** -.5, bl_q=jax.random.normal(kb[1], (D, cb, rb)) * D ** -.5)
    p['w_node'] = jax.random.normal(next(ks), (D,)) * D ** -.5
    p['w_one'] = jax.random.normal(next(ks), (D,)) * D ** -.5
    p |= dict(P=g(D, D), bP=z(D), Rr=g(D, D), bR=z(D))
    if any(kind in MASKS for _, kind in outs):
        p |= dict(w_mask=jax.random.normal(jax.random.fold_in(key, 7), (D,)) * D ** -.5, b_mask=jnp.zeros(()))
    if cfg.get('weight_scale') == 'graph':  # graph tasks: 4 more pair features (unit_features, both directions)
        assert feats is None
        kw = jax.random.split(jax.random.fold_in(key, 23), 2)
        p['e1'] = jnp.concatenate([p['e1'], jax.random.normal(kw[0], (4, W)) * 4 ** -.5])
        p['E'] = jnp.concatenate([p['E'], jax.random.normal(kw[1], (4, D)) * 4 ** -.5])
    if feats is not None:  # the non-graph tasks: their own input embedding and heads
        for k in ('e1', 'eb1', 'e2', 'E', 'eb', 'R', 'C'):
            del p[k]
        F, G = feats['F'], feats['G']
        ne = 2 + G
        ks = iter(jax.random.split(jax.random.fold_in(key, 30), 16))
        g = lambda *s: jax.random.normal(next(ks), s) * s[0] ** -.5
        p['in30'] = dict(e1=g(ne, W), eb1=z(W), e2=g(W, D), E=g(ne, D), eb=z(D), n1=g(F, W), nb1=z(W), R=g(F + W, D), C=g(F + W, D))
        ones = [name for name, kind in outs if kind == 'one']
        if ones:
            p['one30'] = {name: g(D) for name in ones}
        if any(kind == 'ecat' for _, kind in outs):
            p['ec30'] = dict(w=g(D, ECAT), b=z(ECAT))
        if any(kind == 'gmask' for _, kind in outs):
            p['gm30'] = dict(w=g(D), b=z())
    nv = sum(t == 'via' for t in reg_table(cfg))
    if nv:  # the 'via' table: one read-out gain per via pair, no random draw
        p['via'] = jnp.full((nv,), VIA_GAIN, jnp.float32)
    if cfg.get('open_start'):  # training only: these tests start open for every witness (weights 0, bias +6) and stay learned
        h = half(p['blk'])
        if cfg['block'] == 'pair_numbers':     # the grow pair
            c = pn_layout(h)[0] + 2 * PREG_TABLE.index('grow')
        elif cfg['block'] == 'dp_program':     # the program pairs
            c = np.arange(h - 2 * NPREG, h, 2)
        else:                                  # the triangle pairs (compare_first)
            assert cfg.get('triangle_pairs'), 'open_start: pair_numbers, dp_program or triangle pairs'
            c = np.arange(h - 2 * int(cfg['triangle_pairs']), h, 2)
        b = p['blk']
        p['blk'] = dict(b, a=b['a'].at[:, c].set(0.), b=b['b'].at[:, c].set(0.), ba=b['ba'].at[c].set(6.), bb=b['bb'].at[c].set(6.))
    if cfg.get('triangle_pairs') and str(cfg.get('triangle_form', 'cross')).startswith('learned'):   # the learned 2 x 2 form, its own key
        p['blk'] = dict(p['blk'], tri_m=jax.random.normal(jax.random.fold_in(key, 71), (TRI_ROWS, TRI_ROWS)))
    if cfg['block'] == 'item_memory':
        p = item_init(key, cfg, p, outs, feats)
    return prune(p, outs)


def item_init(key, cfg, p, outs, feats):
    """The item block's own parameters, all from key fold 76 (every other draw is as without them): the test's edge bias
    (ie [F, D], ib [D], in blk) and the read-out maps (iro) of the output kinds the task has; with item_read 'hit', ITEM_HIT
    read channels per pointer or edge-mask output (ihit)."""
    D, F = cfg['width'], item_nfeat(cfg, feats)
    kinds, hit = {kind for _, kind in outs}, cfg['item_read'] == 'hit'
    ks = iter(jax.random.split(jax.random.fold_in(key, 76), 16))
    g = lambda *s: jax.random.normal(next(ks), s) * s[0] ** -.5
    z = lambda *s: jnp.zeros(s)
    p = dict(p, blk=dict(p['blk'], ie=g(F, D), ib=z(D)))
    ro, bil = {}, kinds & ({'edge', 'ecat'} if hit else {'node', 'emask', 'edge', 'ecat'})
    if bil: ro |= dict(p=g(D, D), bp=z(D), q=g(D, D), bq=z(D))
    if not hit and kinds & {'node', 'emask'}: ro['e'] = g(F)
    if 'edge' in kinds: ro |= dict(p2=g(D, D), bp2=z(D), ea=g(F), eb=g(F))
    if 'ecat' in kinds: ro['ec'] = g(F, ECAT)
    if ro: p = dict(p, iro=ro)
    names = [name for name, kind in outs if kind in ('node', 'emask')]
    if hit and names:
        kh = iter(jax.random.split(jax.random.fold_in(jax.random.fold_in(key, 76), 1), 4 * len(names)))
        gh = lambda *s: jax.random.normal(next(kh), s) * s[0] ** -.5
        H = ITEM_HIT
        p = dict(p, ihit={name: dict(a=gh(D, H), ba=z(H), b=gh(D, H), bb=z(H), u=gh(F, H), w=z(H), g=jnp.full((H,), VIA_GAIN), b0=jnp.full((), -VIA_GAIN / 2))
                         for name in names})
    return p


def prune(p, outs):
    """Keep only the parameters the task's forward pass reads (the others get no gradient and change nothing)."""
    kinds = {kind for _, kind in outs}
    drop = set()
    if 'node' not in kinds: drop.add('w_node')
    if 'one' not in kinds or 'one30' in p: drop.add('w_one')
    if 'edge' not in kinds: drop |= {'P', 'bP', 'Rr', 'bR'}
    return {k: v for k, v in p.items() if k not in drop}


def count_params(p):
    return int(sum(np.asarray(w).size for w in jax.tree.leaves(p)))


# ------------------------------------------------------------------------------------------------------------ input embedding
def masked_median(v, ok):
    """Median of v [B, N] over the entries where ok; second output: whether any entry is ok."""
    N = v.shape[-1]
    cnt = ok.sum(-1)
    srt = jnp.sort(jnp.where(ok, v, jnp.inf), -1)
    lo = jnp.take_along_axis(srt, jnp.maximum((cnt - 1) // 2, 0)[:, None], 1)[:, 0]
    hi = jnp.take_along_axis(srt, jnp.minimum(cnt // 2, N - 1)[:, None], 1)[:, 0]
    return (lo + hi) / 2, cnt > 0


def graph_scale(A, m):
    """A graph's own unit of weight: the median over items of the lightest edge at the item. A graph with no edge between
    different items takes the median of its self-loop weights; a graph with no weight at all takes 1."""
    N = A.shape[-1]
    pair = (m[:, :, None] * m[:, None, :] > 0) & ~jnp.eye(N, dtype=bool)
    W = jnp.where(A > 0, A, jnp.inf)
    W = jnp.where(pair, jnp.minimum(W, W.swapaxes(1, 2)), jnp.inf)
    l1 = W.min(-1)
    s_edge, any_edge = masked_median(l1, jnp.isfinite(l1))
    d = jnp.diagonal(A, axis1=1, axis2=2)
    s_loop, any_loop = masked_median(d, (d > 0) & (m > 0))
    return jax.lax.stop_gradient(jnp.where(any_edge, s_edge, jnp.where(any_loop, s_loop, 1.)))


def unit_features(A, s):
    """weight_scale = 'graph': x = A / s in the graph's own units, read as x / (1 + x / 16), clip(ln x, -4, 4) and exp(-x)."""
    has = A > 0
    x = jnp.where(has, A / s[:, None, None], 1.)
    lx = jnp.log(x)
    z = lambda t: jnp.where(has, t, 0.)
    return [z(x / (1 + x / 16)), z(jnp.clip(lx, -4., 4.)), z(jnp.exp(-x))]


def embed_graph(p, x):
    """Graph tasks. x: A, adj [B, N, N]; pos, s1, m [B, N] -> z [B, D, N, N]. With weight_scale = 'graph' (read from the
    parameter shape: 4 more pair features) the weights enter in the graph's own units."""
    A, adj, m = x['A'], x['adj'], x['m']
    N = A.shape[-1]
    has = (A != 0).astype(A.dtype)
    eye = jnp.broadcast_to(jnp.eye(N, dtype=A.dtype), A.shape)
    T = lambda t: t.swapaxes(1, 2)
    if p['e1'].shape[0] == NE + 4:
        f = unit_features(A, graph_scale(A, m))
        e = jnp.stack([f[0], T(f[0]), has, T(has), adj, T(adj), eye] + [g for t in f[1:] for g in (t, T(t))], -1)
    else:
        e = jnp.stack([A, T(A), has, T(has), adj, T(adj), eye], -1)
    h = jax.nn.gelu(e @ p['e1'] + p['eb1']) @ p['e2'] + e @ p['E'] + p['eb']
    f = jnp.stack([x['pos'], x['s1']], -1)
    z = h + (f @ p['R'])[:, :, None] + (f @ p['C'])[:, None]
    z = z * (m[:, :, None] * m[:, None, :])[..., None]
    return z.transpose(0, 3, 1, 2)


def embed_items(p, x):
    """Non-graph tasks. x: f [B, N, F] item features, g [B, G] per-input numbers, m [B, N] -> z [B, D, N, N].
    Pair features: [u, v both real], [u = v], and the per-input numbers on every pair."""
    q, f, m = p['in30'], x['f'], x['m']
    B, N, _ = f.shape
    mm = m[:, :, None] * m[:, None, :]
    cols = [mm, jnp.broadcast_to(jnp.eye(N, dtype=f.dtype), mm.shape)]
    cols += [jnp.broadcast_to(x['g'][:, k, None, None], mm.shape) for k in range(x['g'].shape[1])]
    e = jnp.stack(cols, -1)
    h = jax.nn.gelu(e @ q['e1'] + q['eb1']) @ q['e2'] + e @ q['E'] + q['eb']
    u = jnp.concatenate([f, jax.nn.gelu(f @ q['n1'] + q['nb1'])], -1)
    z = h + (u @ q['R'])[:, :, None] + (u @ q['C'])[:, None]
    z = z * mm[..., None]
    return z.transpose(0, 3, 1, 2)


def embed(p, x):
    return embed_items(p, x) if 'f' in x else embed_graph(p, x)


# ------------------------------------------------------------------------------------------------------------ the two halves
def count_compare(xa, xc, m):
    """Count half. xa [B, h, u, k], xc [B, h, k, v] test scores -> [B, h, u, v]: channel 2i is more = [N_2i > N_2i+1],
    channel 2i+1 is same = [N_2i = N_2i+1], where N_c(u, v) counts the real k with xa_c(u, k) > 0 and xc_c(k, v) > 0."""
    A, Cc = st_hard(xa > 0, jax.nn.sigmoid(xa)), st_hard(xc > 0, jax.nn.sigmoid(xc))
    N = jnp.matmul(A * m[:, None, None, :], Cc * m[:, None, :, None])
    d = N[:, 0::2] - N[:, 1::2]
    s = jax.nn.sigmoid(d)
    more, same = st_hard(d > 0, s), st_hard(d == 0, 4. * s * (1. - s))
    return jnp.stack([more, same], 2).reshape(N.shape)


def bilinear_counts(b, t, xa, xc, m):
    """Bilinear tests (model key bilinear_tests): counts for the first P = bl_p.shape[1] count channels,
    N_c(u, v) = the real k with xa_c(u, k) > 0, xc_c(k, v) > 0 and s_c(u, k, v) > 0, where s_c = sum_r (p_cr . t_uk)(q_cr . t_kv)
    is a bilinear form of the two legs' states (rank bilinear_rank; a cross product of two pair differences is one of rank 2)."""
    pu, qv = jnp.einsum('bduk,dcr->bcruk', t, b['bl_p']), jnp.einsum('bdkv,dcr->bcrkv', t, b['bl_q'])
    s = jnp.einsum('bcruk,bcrkv->bcukv', pu, qv)
    A, Cc, S = st_hard(xa > 0, jax.nn.sigmoid(xa)), st_hard(xc > 0, jax.nn.sigmoid(xc)), st_hard(s > 0, jax.nn.sigmoid(s))
    return (A[..., None] * Cc[:, :, None] * S * m[:, None, None, :, None]).sum(3)


def census_count_half(b, t, xa, xc, m):
    """The census block's count half; with bilinear tests (bl_p in b) the first bl_p.shape[1] channels count with them."""
    if 'bl_p' not in b:
        return count_compare(xa, xc, m)
    P = b['bl_p'].shape[1]
    A, Cc = st_hard(xa[:, P:] > 0, jax.nn.sigmoid(xa[:, P:])), st_hard(xc[:, P:] > 0, jax.nn.sigmoid(xc[:, P:]))
    N = jnp.concatenate([bilinear_counts(b, t, xa[:, :P], xc[:, :P], m),
                         jnp.matmul(A * m[:, None, None, :], Cc * m[:, None, :, None])], 1)
    d = N[:, 0::2] - N[:, 1::2]
    s = jax.nn.sigmoid(d)
    return jnp.stack([st_hard(d > 0, s), st_hard(d == 0, 4. * s * (1. - s))], 2).reshape(N.shape)


def best_witness(xa, xc, mk, t):
    """Best half. xa, xc [B, C, N, N] scores, t [C] -> exp(-min_k [max(x_a, x_c) + t min(x_a, x_c)]), x = -log sigmoid(score).
    One channel at a time (memory B N^3)."""
    ca, cc = jax.nn.softplus(-xa), jax.nn.softplus(-xc)
    mid = mk[:, None, :, None] > 0

    def f(args):
        a, c, tt = args
        A, Cc = a[:, :, :, None], c[:, None, :, :]
        return jnp.where(mid, jnp.maximum(A, Cc) + tt * jnp.minimum(A, Cc), jnp.inf).min(2)
    return jnp.exp(-jax.lax.map(f, (ca.swapaxes(0, 1), cc.swapaxes(0, 1), t)).swapaxes(0, 1))


def exists_bits(xa, xc, m, t):
    """Best half as bits (compare_first, pair_numbers): 1 where some real k passes both tests. Backward: the slope of the
    best witness's score."""
    sg = jax.lax.stop_gradient
    ca, cc = jax.nn.softplus(-xa), jax.nn.softplus(-xc)
    mid = m[:, None, :, None] > 0
    T_ = lambda q: q.swapaxes(0, 1)

    def f(args):
        a, c, pa, pc, tt = args
        A, Cc = a[:, :, :, None], c[:, None, :, :]
        s = jnp.exp(-jnp.where(mid, jnp.maximum(A, Cc) + tt * jnp.minimum(A, Cc), jnp.inf))
        adm = pa[:, :, :, None] & pc[:, None, :, :] & mid
        return s.max(2), (s * adm).sum(2), adm.any(2)
    best, _, hard = [T_(q) for q in jax.lax.map(f, (T_(ca), T_(cc), T_(sg(xa) > 0), T_(sg(xc) > 0), sg(t)))]
    return st_hard(hard, best)


# ------------------------------------------------------------------------------------------------------------ census block
def census_loop(b, z, m):
    """One loop of the census block."""
    t = norm(z)
    xa, xc = lin(t, b['a'], b['ba']), lin(t, b['b'], b['bb'])
    h = half(b)
    y = jnp.concatenate([census_count_half(b, t, xa[:, :h], xc[:, :h], m), best_witness(xa[:, h:], xc[:, h:], m, theta(b))], 1)
    z = z + jax.nn.sigmoid(lin(t, b['g'], b['bg'])) * lin(y, b['o'], b['bo'])
    return mlp_update(b, z)


# ------------------------------------------------------------------------------------------------------------ registers
def register_wiring(h, kinds, occurrence=False):
    """compare_first: the register each count channel reads, [h] integers, 0 = a plain fact (count_compare). kinds: groups of
    register rows (from 1) measured in the same units. The first half of the channel pairs are facts; the rest cycle through
    every ordered pair of registers within a group."""
    P = h // 2
    combos = [(a, b) for g in kinds for i, a in enumerate(g) for b in g[i:]]
    nf = P // 2 if combos else P
    wire = np.array([(0, 0)] * nf + [combos[i % len(combos)] for i in range(P - nf)], np.int32).reshape(-1)
    return (wire, np.array([-1] * nf + [i // len(combos) for i in range(P - nf)])) if occurrence else wire


def own_flags(h, kinds):
    """compare_first: channels with fixed tests that compare the pair's own two numbers. 0 = learned tests, 1 = the witness is
    the pair's first item, 2 = its second item. The first occurrence of each register pair is (1, 2), the second (2, 1)."""
    wire, occ = register_wiring(h, kinds, occurrence=True)
    f = np.zeros(h, np.int32)
    for i in np.nonzero(occ == 0)[0]: f[2 * i], f[2 * i + 1] = 1, 2
    for i in np.nonzero(occ == 1)[0]: f[2 * i], f[2 * i + 1] = 2, 1
    return f


def level_counts(adm1, adm2, e1, e2, rule='sens'):
    """One register channel pair: the joint leading value E* (the lowest register value among both channels' admitted
    witnesses) and how many admitted witnesses of each channel reach it."""
    sg = jax.lax.stop_gradient
    x1, x2 = e1[:, None, :, None], e2[:, None, :, None]
    o1, o2 = sg(adm1) > 0, sg(adm2) > 0
    Es = jnp.minimum(jnp.where(o1, x1, jnp.inf).min(2), jnp.where(o2, x2, jnp.inf).min(2))[:, :, None, :]
    i1, i2 = (x1 <= Es, x2 <= Es) if rule == 'sens' else (x1 == Es, x2 == Es) if rule == 'level' else ((x1 == Es) & o1, (x2 == Es) & o2)
    return (adm1 * sg(i1.astype(adm1.dtype))).sum(2), (adm2 * sg(i2.astype(adm2.dtype))).sum(2), Es[:, :, 0, :]


def compare_bits(adm1, adm2, e1, e2):
    """more = [n1 > n2], same = [n1 = n2] on the level counts. Backward: a bit passes a slope only where the loss asks it to
    change, with the counts at exactly the leading level as the slope's source."""
    sg, sig = jax.lax.stop_gradient, jax.nn.sigmoid
    n1, n2, Es = level_counts(adm1, adm2, e1, e2, 'sens')
    d = n1 - n2
    dv = sg(d)
    hm, hs = (dv > 0).astype(d.dtype), (dv == 0).astype(d.dtype)
    dv = jnp.where(((sg(n1) > 0) & (sg(n2) > 0)) | ~jnp.isfinite(Es), dv, jnp.clip(dv, -1., 1.))
    m1, m2, _ = level_counts(adm1, adm2, e1, e2, 'level')
    ds = m1 - m2
    s = sig(dv)
    return (st_hard(d > 0, only_if_wrong(s * (1. - s) * d, hm)),
            st_hard(d == 0, only_if_wrong(4. * s * (1. - s) * (1. - 2. * s) * ds, hs)))


def lead_numeric(A, Cc, e, wire, w):
    """Register channel pairs. A [B, h, u, k], Cc [B, h, k, v] tests (0/1 forward); e [B, T+1, N] register values by row; w [B, N]."""
    wire = np.asarray(wire)
    c1, c2 = np.arange(0, len(wire), 2), np.arange(1, len(wire), 2)
    T_ = lambda q: q.swapaxes(0, 1)

    def f(args):
        a1, k1, a2, k2, e1, e2 = args
        ww = w[:, None, :, None]
        return compare_bits(a1[:, :, :, None] * k1[:, None, :, :] * ww, a2[:, :, :, None] * k2[:, None, :, :] * ww, e1, e2)
    more, same = jax.lax.map(f, (T_(A[:, c1]), T_(Cc[:, c1]), T_(A[:, c2]), T_(Cc[:, c2]), T_(e[:, wire[c1]]), T_(e[:, wire[c2]])))
    return jnp.stack([T_(more), T_(same)], 2).reshape(A.shape[0], len(wire), A.shape[2], Cc.shape[3])


def lead_compare(xa, xc, m, reg, wire, own=None):
    """The count half with registers -> [B, h, u, v]. Fact pairs run count_compare; register pairs compare the register values
    of their admitted witnesses. own: channels whose tests are fixed (the witness is one of the pair's own two items)."""
    sig = jax.nn.sigmoid
    wire = np.asarray(wire)
    if own is not None:
        o = np.asarray(own)[None, :, None, None]
        dg = jnp.where(jnp.eye(xa.shape[-1], dtype=bool), 30., -30.)[None, None]
        xa = jnp.where(o == 1, dg, jnp.where(o == 2, 30., xa))
        xc = jnp.where(o == 1, 30., jnp.where(o == 2, dg, xc))
    isnum = (wire[0::2] > 0) | (wire[1::2] > 0)
    ch = lambda ix: np.stack([2 * ix, 2 * ix + 1], 1).reshape(-1)
    fact, num = ch(np.nonzero(~isnum)[0]), ch(np.nonzero(isnum)[0])
    parts = []
    if len(fact):
        parts.append(count_compare(xa[:, fact], xc[:, fact], m))
    if len(num):
        e = jnp.concatenate([jnp.zeros_like(m)[:, None, :], reg.astype(m.dtype)], 1)
        parts.append(lead_numeric(st_hard(xa[:, num] > 0, sig(xa[:, num])), st_hard(xc[:, num] > 0, sig(xc[:, num])), e, wire[num], m))
    return jnp.concatenate(parts, 1)[:, np.argsort(np.concatenate([fact, num]))]


# ------------------------------------------------------------------------------------------------------------ resting blocks
def rebuild(b, e, y):
    """The state of a resting block: the input embedding e, plus a linear map of the bits y, plus the MLP."""
    s = e + lin(y, b['o'], b['bo'])
    return mlp_update(b, s)


def read_half(b, z, m, part, reg=None, wire=None, own=None):
    """One half of a resting block's channels under the state z: 'count' or 'exists', [B, D/2, N, N]."""
    t = norm(z)
    h = half(b)
    if part == 'count':
        la, lc = lin(t, b['a'][:, :h], b['ba'][:h]), lin(t, b['b'][:, :h], b['bb'][:h])
        return count_compare(la, lc, m) if wire is None else lead_compare(la, lc, m, reg, wire, own=own)
    ua, uc = lin(t, b['a'][:, h:], b['ba'][h:]), lin(t, b['b'][:, h:], b['bb'][h:])
    return exists_bits(ua, uc, m, theta(b))


def cf_layout(b, cfg):
    h = half(b)
    return register_wiring(h, cfg['registers']), own_flags(h, cfg['registers'])


TRI_ROWS = 2                          # triangle pairs: the two number registers (the last rows of reg) they read
TRI_CROSS = ((0., 1.), (-1., 0.))     # the cross form, P(a, b) = x_a y_b - y_a x_b


def tri_pairs(xa, xc, m, tr, M=None):
    """Triangle pairs on a product pair number P(a, b) = tr_a . M tr_b (M: the learned 2 x 2 form divided by its norm; None:
    the cross form). Even pairs: [some admitted third item k: P(u, v) + P(k, u) < P(k, v)] (and level); odd pairs:
    [some admitted k: P(v, u) + P(k, v) < P(k, u)]. k = u and k = v are never admitted, and a pair with u = v gets no bit."""
    sg, sig = jax.lax.stop_gradient, jax.nn.sigmoid; learn = M is not None
    Mn = M * jax.lax.rsqrt((M ** 2).sum() + 1e-12) if learn else jnp.array(TRI_CROSS, tr.dtype)
    P = jnp.einsum('bra,rs,bsc->bac', tr, Mn, tr); Z = jnp.zeros_like(P); T_ = lambda q: q.swapaxes(1, 2); ne = ~jnp.eye(P.shape[-1], dtype=bool)
    mid = (m[:, None, :, None] > 0) & ne[None, :, :, None] & ne[None, None, :, :]; off = ne[None].astype(P.dtype)
    comb = lambda z, a, c: z[:, :, None, :] + a[:, :, :, None] + c[:, None, :, :]; hard = lambda q: st_hard(q > 0, sig(q)); out = []
    for i in range(xa.shape[1] // 2):
        adm = (hard(xa[:, 2 * i])[:, :, :, None] * hard(xc[:, 2 * i])[:, None, :, :]) * mid
        V1, V2 = (comb(P, T_(P), Z), comb(Z, Z, P)) if i % 2 == 0 else (comb(T_(P), Z, P), comb(Z, T_(P), Z))
        less, level = reg_any(adm, V1, V2)
        if learn:
            best = jnp.where(sg(adm) > 0, V2 - V1, -PREG_BIG).max(2); best = jnp.where(best > -PREG_BIG / 2, best, 0.)
            q = only_if_wrong(best, sg(less)); less = less + (q - sg(q))
        out += [less * off, level * off]
    return jnp.stack(out, 1)


def cf_count(b, cfg, z, x, wire, own):
    """compare_first's count half; with triangle_pairs T, its last 2 T channels are triangle pairs on the last TRI_ROWS register rows."""
    T = int(cfg.get('triangle_pairs', 0))
    if not T:
        return read_half(b, z, x['m'], 'count', x['reg'], wire, own)
    t = norm(z); h = half(b); nt = 2 * T
    la, lc = lin(t, b['a'][:, :h], b['ba'][:h]), lin(t, b['b'][:, :h], b['bb'][:h])
    return jnp.concatenate([lead_compare(la[:, :h - nt], lc[:, :h - nt], x['m'], x['reg'][:, :-TRI_ROWS], wire[:h - nt], own=own[:h - nt]),
                            tri_pairs(la[:, h - nt:], lc[:, h - nt:], x['m'], x['reg'][:, -TRI_ROWS:], b.get('tri_m'))], 1)


def cf_start(p, cfg, x):
    """compare_first: -> e, and the first bits y_0: zero, except the fixed comparisons of each pair's own numbers (input).
    Without kept bits (kept_bits: false), y_0 is zero."""
    sg = jax.lax.stop_gradient
    e = embed(p, x)
    if not cfg.get('kept_bits', True):
        return e, jnp.zeros_like(e)
    b = p['blk']
    h = half(b)
    wire, own = cf_layout(b, cfg)
    y = jnp.zeros_like(e)
    c0 = cf_count(b, cfg, rebuild(b, e, y), x, wire, own)
    y = jnp.concatenate([sg(c0) * jnp.array(own > 0, e.dtype)[None, :, None, None], y[:, h:]], 1)
    return e, y


def cf_step(p, cfg, x, e, y):
    """compare_first, one loop. The kept bits L only turn on; the count half reads the input and L only. Without kept bits
    (kept_bits: false; depth-first search with positions in a register) both halves read the state rebuilt from all the
    bits, as the cold block does, the count half comparing the register at its witnesses."""
    b = p['blk']
    h = half(b)
    wire, own = cf_layout(b, cfg)
    if not cfg.get('kept_bits', True):
        t = norm(rebuild(b, e, y))
        xa, xc = lin(t, b['a'], b['ba']), lin(t, b['b'], b['bb'])
        return jnp.concatenate([lead_compare(xa[:, :h], xc[:, :h], x['m'], x['reg'], wire, own=own),
                                exists_bits(xa[:, h:], xc[:, h:], x['m'], theta(b))], 1)
    L = jnp.maximum(y[:, h:], read_half(b, rebuild(b, e, y), x['m'], 'exists'))
    keep = y[:, :h] * jnp.array(own > 0, y.dtype)[None, :, None, None]
    cb = cf_count(b, cfg, rebuild(b, e, jnp.concatenate([keep, L], 1)), x, wire, own)
    return jnp.concatenate([cb, L], 1)


def cold_start(p, cfg, x):
    """cold: -> e, and the first bits y_0 = 0."""
    e = embed(p, x)
    return e, jnp.zeros_like(e)


def cold_step(p, cfg, x, e, y):
    """cold, one loop: both halves read the state rebuilt from the bits y; the count half hands on more / same, the best
    half [some witness passes both tests]."""
    b = p['blk']
    h = half(b)
    t = norm(rebuild(b, e, y))
    xa, xc = lin(t, b['a'], b['ba']), lin(t, b['b'], b['bb'])
    return jnp.concatenate([count_compare(xa[:, :h], xc[:, :h], x['m']), exists_bits(xa[:, h:], xc[:, h:], x['m'], theta(b))], 1)


# --------------------------------------------------------------------------------- pair numbers
def present(v):
    return v < PREG_BIG / 2


def reg_table(cfg):
    """The 8 register pairs' kinds of a pair_numbers config ('paths' unless the config names 'via' or 'hops')."""
    return REG_TABLES[cfg.get('table', 'paths')] if cfg.get('block') == 'pair_numbers' else ()


def pn_v2(cfg):
    """pair_numbers with the 'via' or 'hops' table or self loops: the loop state is (bits, R, H, hits)."""
    return cfg['block'] == 'pair_numbers' and (cfg.get('table', 'paths') != 'paths' or bool(cfg.get('self_loops')))


def pn_layout(h, table=PREG_TABLE):
    """pair_numbers: the count half's layout. 16 pairs of facts, 8 pairs that compare positions (the first two with fixed
    tests: the pair's own two items), then the NPREG = 8 register pairs. -> base (channels before the register pairs), wire
    and own of the position part, ownbit [h] (bits that depend on no learned test)."""
    P = h // 2
    NNODE = P // 4
    base = 2 * (P - NPREG)
    wire = np.array([0] * (2 * (P - NPREG - NNODE)) + [1] * (2 * NNODE), np.int32)
    own = np.zeros(base, np.int32)
    s = 2 * (P - NPREG - NNODE)
    own[s:s + 4] = (1, 2, 2, 1)
    ownbit = np.concatenate([own > 0, np.repeat(np.array([t == 'edge' for t in table]), 2)])
    return base, wire, own, ownbit


def pn_start_reg(x, kind):
    """R at the start: the edge weights W, with 0 ('plus') or -BIG ('max') on the diagonal of real items."""
    N = x['W'].shape[-1]
    eye = jnp.eye(N, dtype=bool)[None]
    real = (x['m'][:, :, None] * x['m'][:, None, :]) > 0
    return jnp.where(eye & real, 0. if kind == 'plus' else -PREG_BIG, x['W'])


def reg_min(adm, V):
    """adm [B, u, k, v] admitted witnesses; V [B, u, k, v] values -> L [B, u, v], the lowest admitted value (BIG if none).
    Backward: slope 1 to the value at the best admitted witness, -1 to the admission of that witness and of every closed
    witness whose value is below L."""
    sg = jax.lax.stop_gradient
    op = sg(adm) > 0
    L = jnp.where(op & present(V), V, PREG_BIG).min(2)
    Vs, Ls = sg(V), sg(L)[:, :, None, :]
    w = present(Vs) & ((op & (Vs <= Ls)) | (~op & (Vs < Ls)))
    proxy = -(adm * w.astype(adm.dtype)).sum(2)
    return L + (proxy - sg(proxy))


def reg_bits(L1, L2):
    """less = [L1 < L2], level = [L1 = L2, present]."""
    sg = jax.lax.stop_gradient
    less, level = sg(L1) < sg(L2), (sg(L1) == sg(L2)) & present(sg(L1))
    d = L2 - L1
    return st_hard(less, only_if_wrong(d, less.astype(d.dtype))), st_hard(level, only_if_wrong(-jnp.abs(d), level.astype(d.dtype)))


def reg_any(adm, V1, V2):
    """Per-witness comparison: less = [some admitted k: V1 < V2], level = [some admitted k: V1 = V2, both present]."""
    sg = jax.lax.stop_gradient
    ok = present(sg(V1)) & present(sg(V2))
    lt = (sg(V1) < sg(V2)) & present(sg(V1))
    eq = (sg(V1) == sg(V2)) & ok
    op = sg(adm) > 0
    n_lt, n_eq = (adm * lt.astype(adm.dtype)).sum(2), (adm * eq.astype(adm.dtype)).sum(2)
    gap = jnp.where(op & ok, jnp.abs(V1 - V2), PREG_BIG).min(2)
    gap = jnp.where(present(sg(gap)), gap, 0.)
    less, level = sg(n_lt) > 0, sg(n_eq) > 0
    return st_hard(less, only_if_wrong(n_lt, less.astype(n_lt.dtype))), st_hard(level, only_if_wrong(n_eq - gap, level.astype(n_eq.dtype)))


def reg_pairs(xa, xc, m, W, R, kind):
    """The NPREG register pairs -> bits [B, 2 NPREG, u, v] (less, level per pair, in PREG_TABLE's order) and the new R.
      grow     L(u, v) = min over admitted k of R(u, k) + W(k, v) ('plus') or max(R(u, k), R(k, v)) ('max'); kept in R;
               bits compare L with R(u, v) ("improved").
      edge     R(u, v) against W(u, v) (the edge is its own best path), no learned test.
      fwd x3   per admitted k: W(u, v) (+) R(k, u) against R(k, v).    bwd x3: W(v, u) (+) R(k, v) against R(k, u)."""
    sig = jax.nn.sigmoid
    N = W.shape[-1]
    mid = (m[:, None, :, None] > 0)
    eye = jnp.eye(N, dtype=W.dtype)[None]
    plus = kind == 'plus'
    I = jnp.zeros_like(W) if plus else jnp.full_like(W, -PREG_BIG)
    T_ = lambda q: q.swapaxes(1, 2)
    if plus:
        comb = lambda z, a, c: z[:, :, None, :] + a[:, :, :, None] + c[:, None, :, :]
    else:
        comb = lambda z, a, c: jnp.maximum(z[:, :, None, :], jnp.maximum(a[:, :, :, None], c[:, None, :, :]))
    hard = lambda s: st_hard(s > 0, sig(s))
    learned = lambda i: (hard(xa[:, i])[:, :, :, None] * hard(xc[:, i])[:, None, :, :]) * mid
    end2 = (jnp.ones_like(W)[:, :, :, None] * eye[:, None, :, :]) * mid   # fixed test: the witness is the pair's second item
    own = lambda X: reg_min(end2, comb(I, X, I))
    out, Rn = [], R
    for i, t in enumerate(PREG_TABLE):
        c1 = 2 * i
        if t == 'grow':
            L1 = reg_min(learned(c1), comb(I, R, W if plus else R))
            Rn = jnp.minimum(Rn, L1)
            out += list(reg_bits(L1, own(R)))
        elif t == 'edge':
            out += list(reg_bits(own(R), own(W)))
        elif t == 'fwd':
            out += list(reg_any(learned(c1), comb(W, T_(R), I), comb(I, I, R)))
        else:
            out += list(reg_any(learned(c1), comb(T_(W), I, R), comb(I, T_(R), I)))
    return jnp.stack(out, 1), Rn


def pn_count_half(b, z, m, cfg, x, R):
    """pair_numbers: the count half under the state z: the position comparisons and facts, then the register pairs."""
    t = norm(z)
    h = half(b)
    base, wire, own, _ = pn_layout(h)
    xa, xc = lin(t, b['a'][:, :h], b['ba'][:h]), lin(t, b['b'][:, :h], b['bb'][:h])
    y1 = lead_compare(xa[:, :base], xc[:, :base], m, x['reg'], wire, own=own)
    y2, Rn = reg_pairs(xa[:, base:], xc[:, base:], m, x['W'], R, cfg['arithmetic'])
    return jnp.concatenate([y1, y2], 1), Rn


def pn_start(p, cfg, x):
    """pair_numbers: -> e, and the first (bits, R): kept bits at 0, the own bits of the first pass already in."""
    sg = jax.lax.stop_gradient
    e = embed(p, x)
    b = p['blk']
    h = half(b)
    y = jnp.zeros_like(e)
    R = pn_start_reg(x, cfg['arithmetic'])
    ownbit = jnp.array(pn_layout(h)[3], e.dtype)[None, :, None, None]
    c0, _ = pn_count_half(b, rebuild(b, e, y), x['m'], cfg, x, R)
    return e, (jnp.concatenate([sg(c0) * ownbit, y[:, h:]], 1), R)


def pn_step(p, cfg, x, e, s):
    """pair_numbers, one loop: kept bits, then R, then the count half on the R the loop started with."""
    y, R = s
    b = p['blk']
    h = half(b)
    ownbit = jnp.array(pn_layout(h)[3], e.dtype)[None, :, None, None]
    L = jnp.maximum(y[:, h:], read_half(b, rebuild(b, e, y), x['m'], 'exists'))
    cb, Rn = pn_count_half(b, rebuild(b, e, jnp.concatenate([y[:, :h] * ownbit, L], 1)), x['m'], cfg, x, R)
    yn = jnp.concatenate([cb, L], 1)
    return (yn, Rn)


def pn2_start_reg(x, kind):
    """'via' / 'hops' / self loops: as pn_start_reg, but an item with a weighted loop starts at that weight."""
    N = x['W'].shape[-1]
    eye = jnp.eye(N, dtype=bool)[None]
    real = (x['m'][:, :, None] * x['m'][:, None, :]) > 0
    return jnp.where(eye & real, jnp.where(present(x['W']), x['W'], 0. if kind == 'plus' else -PREG_BIG), x['W'])


def pn2_start_hops(x, R):
    """H at the start: 0 on the diagonal, 1 where the edge is its own best path (R = W), absent elsewhere."""
    N = R.shape[-1]
    eye = jnp.eye(N, dtype=bool)[None]
    real = (x['m'][:, :, None] * x['m'][:, None, :]) > 0
    return jnp.where(eye & real, 0., jnp.where(present(x['W']) & (R == x['W']) & real & ~eye, 1., PREG_BIG)).astype(R.dtype)


def reg_pairs2(xa, xc, m, W, R, H, kind, table):
    """The 8 register pairs under a table -> bits [B, 16, u, v], the new R, the hop candidate (H or better), the via hits
    [B, u, k, v] of each via pair. Beyond reg_pairs' kinds:
      via      per admitted k: R(u, k) + W(k, v) against R(u, v); hit = admitted and equal.
      hgrow    min over admitted k of H(u, k) + H(k, v), kept in H; bits compare it with H(u, v).
      hfwd     per admitted k: H(k, u) against H(k, v).    hbwd: H(k, v) against H(k, u)."""
    sg, sig = jax.lax.stop_gradient, jax.nn.sigmoid
    N = W.shape[-1]
    mid = (m[:, None, :, None] > 0)
    eye = jnp.eye(N, dtype=W.dtype)[None]
    plus = kind == 'plus'
    I = jnp.zeros_like(W) if plus else jnp.full_like(W, -PREG_BIG)
    Z0 = jnp.zeros_like(W)
    T_ = lambda q: q.swapaxes(1, 2)
    cplus = lambda z, a, c: z[:, :, None, :] + a[:, :, :, None] + c[:, None, :, :]
    comb = cplus if plus else (lambda z, a, c: jnp.maximum(z[:, :, None, :], jnp.maximum(a[:, :, :, None], c[:, None, :, :])))
    hard = lambda s: st_hard(s > 0, sig(s))
    learned = lambda i: (hard(xa[:, i])[:, :, :, None] * hard(xc[:, i])[:, None, :, :]) * mid
    end2 = (jnp.ones_like(W)[:, :, :, None] * eye[:, None, :, :]) * mid
    own = lambda X: reg_min(end2, comb(I, X, I))
    ownh = lambda X: reg_min(end2, cplus(Z0, X, Z0))
    out, Rn, Hc, hits = [], R, H, []
    for i, t in enumerate(table):
        c1 = 2 * i
        if t == 'grow':
            L1 = reg_min(learned(c1), comb(I, R, W if plus else R))
            Rn = jnp.minimum(Rn, L1)
            out += list(reg_bits(L1, own(R)))
        elif t == 'edge':
            out += list(reg_bits(own(R), own(W)))
        elif t == 'fwd':
            out += list(reg_any(learned(c1), comb(W, T_(R), I), comb(I, I, R)))
        elif t == 'bwd':
            out += list(reg_any(learned(c1), comb(T_(W), I, R), comb(I, T_(R), I)))
        elif t == 'via':
            adm = learned(c1)
            A1, A2 = comb(I, R, W), comb(R, I, I)
            out += list(reg_any(adm, A1, A2))
            hits.append(adm * ((sg(A1) == sg(A2)) & present(sg(A1)) & present(sg(A2))).astype(adm.dtype))
        elif t == 'hgrow':
            L1 = reg_min(learned(c1), cplus(Z0, H, H))
            Hc = jnp.minimum(Hc, L1)
            out += list(reg_bits(L1, ownh(H)))
        elif t == 'hfwd':
            out += list(reg_any(learned(c1), cplus(Z0, T_(H), Z0), cplus(Z0, Z0, H)))
        elif t == 'hbwd':
            out += list(reg_any(learned(c1), cplus(Z0, Z0, H), cplus(Z0, T_(H), Z0)))
        else:
            raise ValueError(t)
    return jnp.stack(out, 1), Rn, Hc, hits


def pn2_count_half(b, z, m, cfg, x, R, H):
    """The count half under the state z with a register table -> bits, new R, hop candidate, hits."""
    t = norm(z)
    h = half(b)
    table = reg_table(cfg)
    base, wire, own, _ = pn_layout(h, table)
    xa, xc = lin(t, b['a'][:, :h], b['ba'][:h]), lin(t, b['b'][:, :h], b['bb'][:h])
    y1 = lead_compare(xa[:, :base], xc[:, :base], m, x['reg'], wire, own=own)
    y2, Rn, Hc, hits = reg_pairs2(xa[:, base:], xc[:, base:], m, x['W'], R, H, cfg['arithmetic'], table)
    return jnp.concatenate([y1, y2], 1), Rn, Hc, hits


def pn2_start(p, cfg, x):
    """-> e, and the first (bits, R, H, hits)."""
    sg = jax.lax.stop_gradient
    e = embed(p, x)
    b = p['blk']
    h = half(b)
    y = jnp.zeros_like(e)
    R = pn2_start_reg(x, cfg['arithmetic'])
    H = pn2_start_hops(x, R) if cfg.get('table') == 'hops' else jnp.zeros_like(R)
    ownbit = jnp.array(pn_layout(h, reg_table(cfg))[3], e.dtype)[None, :, None, None]
    c0, _, _, hits0 = pn2_count_half(b, rebuild(b, e, y), x['m'], cfg, x, R, H)
    return e, (jnp.concatenate([sg(c0) * ownbit, y[:, h:]], 1), R, H, hits0)


def pn2_step(p, cfg, x, e, s):
    """One loop: kept bits; R grows; H restarts from the new R when R moved, else grows; the count half's bits on the R
    (and H) the loop started with."""
    sg = jax.lax.stop_gradient
    y, R, H, _ = s
    b = p['blk']
    h = half(b)
    ownbit = jnp.array(pn_layout(h, reg_table(cfg))[3], e.dtype)[None, :, None, None]
    L = jnp.maximum(y[:, h:], read_half(b, rebuild(b, e, y), x['m'], 'exists'))
    cb, Rn, Hc, hits = pn2_count_half(b, rebuild(b, e, jnp.concatenate([y[:, :h] * ownbit, L], 1)), x['m'], cfg, x, R, H)
    if cfg.get('table') == 'hops':
        pair = (x['m'][:, :, None] * x['m'][:, None, :]) > 0
        moved = ((sg(Rn) != sg(R)) & pair).any((1, 2))
        Hn = jnp.where(moved[:, None, None], pn2_start_hops(x, sg(Rn)), Hc)
    else:
        Hn = H
    return (jnp.concatenate([cb, L], 1), Rn, Hn, hits)



# --------------------------------------------------------------------------------- dynamic programs given as a program
# dp_program: compare_first's node registers and the cold kept-bit loop, and on the last NPREG channel pairs a fixed
# program over built pair numbers (a task's recurrence written as register moves). Each pair names its move:
#   ('grow', t, z, a, c)   t(u, v) <- min(t(u, v), min over admitted k of z(u, v) + a(u, k) + c(k, v)); hands on improved / level
#   ('own', t, a)          t <- min(t, a) at the pair itself;   ('cmp', a, c): compares a with c at the pair (no test)
#   ('any', (z, a, k), (z2, a2, k2))   per admitted k, compares the two sums;   ('via', t, z, a, c): as 'any' against t, and
#                          a pair's pointer to k is read with a gain where the two are equal;   ('none',): two zero channels.
# Register names: '0'; an input (W, N, M); a built number; 'Xt' its transpose; 'X+' read one item later in the first
# index; 'Wo' W without its loops. prog_fixed (model key fixed): every real middle item is admitted, no learned test.
_DG = lambda t, z, x, y: ('grow', t, z, x, y)
_DOWN = lambda t, x: ('own', t, x)
_DVIA = lambda t, z, x, y: ('via', t, z, x, y)
_DNONE = ('none',)
DP_PROGRAMS = {
    'optimal_bst': dict(kinds=[[1], [2, 3]], built=('R', 'E'), pairs=[_DG('R', '0', 'R', 'Wo'), _DG('E', 'R', 'E', 'E+'), _DVIA('E', 'R', 'E', 'E+'), _DVIA('E', 'R', 'E', 'E+'), ('cmp', 'E', 'R'), _DNONE, _DNONE, _DNONE]),
    'find_maximum_subarray_kadane': dict(kinds=[[1], [2]], built=('R', 'B', 'G'), pairs=[_DG('R', '0', 'R', 'Wo'), _DG('B', '0', 'R', '0'), _DG('G', '0', '0', 'B'), ('any', ('0', 'R', '0'), ('G', '0', '0')), ('any', ('0', '0', 'R'), ('G', '0', '0')),
                                                                                 ('any', ('0', 'R', '0'), ('G', '0', '0')), ('any', ('0', '0', 'R'), ('G', '0', '0')), _DNONE]),
    'lcs_length': dict(kinds=[[1]], built=('C', 'D', 'L'), pairs=[_DG('D', '0', 'Nt', 'C'), _DG('L', '0', 'C', 'N'), _DOWN('C', 'D'), _DOWN('C', 'L'), _DG('C', 'M', 'D', 'N'), ('cmp', 'D', 'L'), _DNONE, _DNONE])}
# The string matchers given a stored count: H = steps along each string (grown by doubling), K = rank in the string, M = the
# text item and the pattern item hold the same letter.
STR_TASKS = ('naive_string_matcher', 'kmp_matcher')
_DANY = lambda a, b: ('any', a, b)
_DSTR = dict(kinds=[[1]], built=('H', 'K'), pairs=[_DG('H', '0', 'H', 'H'), _DG('K', '0', 'K', 'H'), _DANY(('0', 'H', 'M'), ('0', '0', 'K')), _DANY(('0', 'H', '0'), ('0', '0', 'K')),
                                                     _DANY(('0', 'H', 'M'), ('0', '0', 'K')), _DANY(('0', 'H', '0'), ('0', '0', 'K')), ('cmp', 'H', 'K'), _DNONE])
DP_PROGRAMS |= {t: _DSTR for t in STR_TASKS}


def dp_start_regs(x, cfg):
    """The built registers at the start."""
    task = cfg['program']; N = x['m'].shape[1]; eye = jnp.eye(N, dtype=bool)[None]
    real = (x['m'][:, :, None] * x['m'][:, None, :]) > 0; big = jnp.full(real.shape, PREG_BIG, jnp.float32)
    if task == 'optimal_bst':
        d = jnp.where(eye & real, x['W'], big)
        return dict(R=d, E=d)
    if task == 'find_maximum_subarray_kadane':
        return dict(R=jnp.where(eye & real, x['W'], big), B=big, G=big)
    if task in STR_TASKS:   # H = 0 on the diagonal and 1 from an item to its next item; K = 0 at each string's first item
        f = x['first'] > 0
        return dict(H=jnp.where(eye & real, 0., jnp.where((x['nxt'] > 0) & real, 1., PREG_BIG)).astype(jnp.float32), K=jnp.where(f[:, None, :] & real, 0., PREG_BIG).astype(jnp.float32))
    f = x['first'] > 0
    return dict(C=jnp.where(present(x['M']), x['M'], jnp.where(real, 0., PREG_BIG)).astype(jnp.float32),
                D=jnp.where(f[:, :, None] & real, 0., PREG_BIG).astype(jnp.float32), L=jnp.where(f[:, None, :] & real, 0., PREG_BIG).astype(jnp.float32))


def dp_lookup(name, x, regs):
    Z0 = jnp.zeros_like(x['nxt'])
    if name == '0':
        return Z0
    if name == 'Wo':
        return jnp.where(jnp.eye(Z0.shape[-1], dtype=bool)[None], PREG_BIG, x['W'])
    if name.endswith('t') and name[:-1] in regs or name in ('Nt', 'Wt'):
        return dp_lookup(name[:-1], x, regs).swapaxes(1, 2)
    if name.endswith('+'):
        A = dp_lookup(name[:-1], x, regs); nx = x['nxt']
        return jnp.where(nx.sum(2)[:, :, None] > 0, jnp.einsum('brk,bkv->brv', nx, jnp.where(present(A), A, 0.)) +
                         jnp.einsum('brk,bkv->brv', nx, (~present(A)).astype(A.dtype)) * PREG_BIG, PREG_BIG)
    return regs[name] if name in regs else x[name]


def dp_reg_program(xa, xc, m, x, regs, P, fixed=False):
    """The 8 register pairs of program P -> bits [B, 16, u, v], the new registers, the via hits."""
    sg, sig = jax.lax.stop_gradient, jax.nn.sigmoid
    N = m.shape[1]; mid = (m[:, None, :, None] > 0); eye = jnp.eye(N, dtype=jnp.float32)[None]; Z0 = jnp.zeros((1, N, N), jnp.float32)
    comb = lambda z, a, c: (a[:, :, :, None] + c[:, None, :, :]) + z[:, :, None, :]          # (left + right) + pair term
    learned = lambda i: st_hard((xa[:, i] > 0)[:, :, :, None] & (xc[:, i] > 0)[:, None, :, :], sig(xa[:, i])[:, :, :, None] * sig(xc[:, i])[:, None, :, :]) * mid
    if fixed:
        every = jnp.ones((m.shape[0], N, N, N), jnp.float32) * mid
        learned = lambda i: every
    end2 = (jnp.ones((m.shape[0], N, N), jnp.float32)[:, :, :, None] * eye[:, None, :, :]) * mid
    own = lambda X: reg_min(end2, comb(Z0, X, Z0))
    L_ = lambda n: dp_lookup(n, x, regs)
    out = []; new = dict(regs); hits = []; zero2 = [jnp.zeros((m.shape[0], N, N), jnp.float32)] * 2
    for i, pr in enumerate(P['pairs']):
        c1 = 2 * i; kind = pr[0]
        if kind == 'grow':
            _, t, z, a, c = pr; L1 = reg_min(learned(c1), comb(L_(z), L_(a), L_(c))); new[t] = jnp.minimum(new[t], L1); out += list(reg_bits(L1, own(regs[t])))
        elif kind == 'own':
            _, t, a = pr; L1 = own(L_(a)); new[t] = jnp.minimum(new[t], L1); out += list(reg_bits(L1, own(regs[t])))
        elif kind == 'cmp':
            out += list(reg_bits(own(L_(pr[1])), own(L_(pr[2]))))
        elif kind == 'any':
            (z1, a1, k1), (z2, a2, k2) = pr[1], pr[2]
            out += list(reg_any(learned(c1), comb(L_(z1), L_(a1), L_(k1)), comb(L_(z2), L_(a2), L_(k2))))
        elif kind == 'via':
            _, t, z, a, c = pr; adm = learned(c1); A1 = comb(L_(z), L_(a), L_(c)); A2 = comb(regs[t], Z0, Z0)
            out += list(reg_any(adm, A1, A2)); hits.append(adm * ((sg(A1) == sg(A2)) & present(sg(A1)) & present(sg(A2))).astype(adm.dtype))
        else:
            out += zero2
    return jnp.stack(out, 1), new, hits


def dp_layout(h, cfg):
    base = h - 2 * NPREG; P = DP_PROGRAMS[cfg['program']]; kinds = cfg['registers']
    wire = register_wiring(base, kinds); own = own_flags(base, kinds)
    return base, wire, own, np.concatenate([own > 0, np.repeat(np.array([pr[0] in ('cmp', 'own') for pr in P['pairs']]), 2)])


def dp_count_half(b, z, m, cfg, x, regs):
    t = norm(z); h = half(b); base, wire, own, _ = dp_layout(h, cfg)
    xa, xc = lin(t, b['a'][:, :h], b['ba'][:h]), lin(t, b['b'][:, :h], b['bb'][:h])
    y1 = lead_compare(xa[:, :base], xc[:, :base], m, x['reg'], wire, own=own)
    y2, new, hits = dp_reg_program(xa[:, base:], xc[:, base:], m, x, regs, DP_PROGRAMS[cfg['program']], fixed=bool(cfg.get('fixed')))
    return jnp.concatenate([y1, y2], 1), new, hits


def dp_start(p, cfg, x):
    """dp_program: -> e, and the loop's first (bits, registers, hits)."""
    sg = jax.lax.stop_gradient
    e = embed(p, x); b = p['blk']; h = half(b); y = jnp.zeros_like(e); regs = dp_start_regs(x, cfg)
    ownbit = jnp.array(dp_layout(h, cfg)[3], e.dtype)[None, :, None, None]
    c0, _, hits0 = dp_count_half(b, rebuild(b, e, y), x['m'], cfg, x, regs)
    return e, (jnp.concatenate([sg(c0) * ownbit, y[:, h:]], 1), regs, hits0)


def dp_step(p, cfg, x, e, s):
    """dp_program, one loop: kept bits, then the program's registers, then the count half on the registers the loop began with."""
    y, regs, _ = s; b = p['blk']; h = half(b); ownbit = jnp.array(dp_layout(h, cfg)[3], e.dtype)[None, :, None, None]
    L = jnp.maximum(y[:, h:], read_half(b, rebuild(b, e, y), x['m'], 'exists'))
    cb, new, hits = dp_count_half(b, rebuild(b, e, jnp.concatenate([y[:, :h] * ownbit, L], 1)), x['m'], cfg, x, regs)
    return (jnp.concatenate([cb, L], 1), new, hits)


# --------------------------------------------------------------------------------- item memory
# item_memory: the cold block with memory on ITEMS instead of pairs. The state is one bit vector per item [B, D, N]; one loop sums
# over item i's partners k (cost N^2, ordinary attention), read cold as the pair block reads: the test of partner k by item i in
# channel c is l_c(i, k) = (a_c . t_i + b_a,c)(b_c . t_k + b_b,c) + u_c . eps_ik + w_c (the pair block's two leg maps as a rank-1
# query and key, plus an edge bias of the pair's input facts eps_ik); admitted: [l > 0], k real. Count half: fact pairs compare
# the numbers of admitted partners; register pairs compare the lowest value among admitted partners and how many reach it; own
# channels have the item itself as the only witness. Exists half: [some real partner admitted]. item_w (weighted graphs): the
# weights enter only as lead values W(k, i) and W(i, k). item_store ('plus' or 'max'): one built number per item, d, grown over
# admitted partners as pair_numbers grows R. item_read 'hit': pointers and edge masks read from the partners that win item i's
# cold reading; 'bilinear': a bilinear read. Read when the item bits (and d) repeat, cap n loops.
def item_combos(kinds):
    """the register combinations (a, b), a <= b, within one kind (wiring's combos, in its order)."""
    return [(a, b) for g in kinds for i, a in enumerate(g) for b in g[i:]]


def item_layout(cfg, h, T):
    """-> wire [h] (the value type of each count channel: 0 none, 1..T the rows of reg, T + 1 and T + 2 the edge values W(k, i), W(i, k)
    with item_w), own [h] (own_flags; nonzero: the only witness is the item itself), the kinds made of rows of reg (eps's comparisons).
    item_store: the layout covers the first h - 2 NPREG channels only; the last NPREG pairs are the store's (ITEM_STORE_TABLE), as in pair_numbers."""
    kinds = cfg['registers']; hb = h - 2 * NPREG if cfg.get('item_store') else h; wire = register_wiring(hb, kinds); own = own_flags(hb, kinds)
    assert wire.max() == T + (2 if cfg.get('item_w') else 0), 'item_memory: registers names every register row (with item_w, then the two edge values)'
    return wire, own, [g for g in kinds if max(g) <= T]


def item_nfeat(cfg, feats):
    """the number of pair input facts in eps: 7 (graph: embed's pair features) or 2 (arrays: [i, k real], [i = k]), plus 4 per register
    combination within a kind of rows of reg."""
    T = 1 if feats is None else 1 + feats['S'] + feats['G']
    return (7 if feats is None else 2) + 4 * len(item_combos([g for g in cfg['registers'] if max(g) <= T]))


def lin1(t, w, b=None):
    """lin for item tensors [B, C, N]."""
    y = jnp.einsum('bci,cd->bdi', t, w)
    return y if b is None else y + b[None, :, None]


def item_context(p, cfg, x):
    """what the item block reads besides its bits, fixed for an input. e [B, D, N]: the item embedding (the pair embedding's diagonal).
    eps [B, i, k, F]: the pair input facts of the leg (i, k), 0 off real pairs. V [B, types, i, k]: the value of partner k for item i by
    register type (0: none; t: row t of reg at k; with item_w, W(k, i) and W(i, k)); P, the same shape: 1 where that value exists (k real;
    for the edge values, an edge in that direction)."""
    m = x['m']; N = m.shape[1]; mm = m[:, :, None] * m[:, None, :]; eye = jnp.broadcast_to(jnp.eye(N, dtype=m.dtype), mm.shape)
    e = jnp.diagonal(embed(p, x), axis1=2, axis2=3)
    if 'f' in x: cols = [mm, eye]
    else:
        A, adj = x['A'], x['adj']; has = (A != 0).astype(A.dtype); T_ = lambda q: q.swapaxes(1, 2); cols = [A, T_(A), has, T_(has), adj, T_(adj), eye]
    reg = x['reg'].astype(m.dtype); T = reg.shape[1]; rk = item_layout(cfg, half(p['blk']), T)[2]
    for a, b in item_combos(rk):
        ra, rb = reg[:, a - 1], reg[:, b - 1]
        cols += [ra[:, :, None] < rb[:, None, :], ra[:, :, None] == rb[:, None, :], ra[:, None, :] < rb[:, :, None], ra[:, None, :] == rb[:, :, None]]
    eps = jnp.stack([c.astype(m.dtype) for c in cols], -1) * mm[..., None]
    mk = jnp.broadcast_to(m[:, None, :], mm.shape); V = [jnp.zeros_like(mm)] + [jnp.broadcast_to(reg[:, t][:, None, :], mm.shape) for t in range(T)]; P = [mk] * (T + 1)
    ctx = {}
    if cfg.get('item_w'):
        W = x['W'].astype(m.dtype); Wt = W.swapaxes(1, 2); V += [Wt, W]; P += [present(Wt).astype(m.dtype) * mk, present(W).astype(m.dtype) * mk]; ctx = dict(Win=Wt, Wout=W)
    return dict(ctx, e=e, eps=eps, V=jnp.stack(V, 1), P=jnp.stack(P, 1))


def item_scores(b, t, eps, c):
    """the test logits of the channels c (a slice) on every leg (i, k): (a . t_i + b_a)(b . t_k + b_b) + u . eps_ik + w. t [B, D, N] -> [B, C, i, k]."""
    q, k = lin1(t, b['a'][:, c], b['ba'][c]), lin1(t, b['b'][:, c], b['bb'][c])
    return q[:, :, :, None] * k[:, :, None, :] + jnp.einsum('bikf,fc->bcik', eps, b['ie'][:, c]) + b['ib'][c][None, :, None, None]


def item_level_counts(adm1, adm2, V1, V2, rule='sens'):
    """(level_counts on items, witness axis last): adm [B, C, i, k] 0/1 forward, V [B, C, i, k] values -> n1, n2 [B, C, i] admitted partners at
    the joint leading level E*, and E* [B, C, i]."""
    sg = jax.lax.stop_gradient; o1, o2 = sg(adm1) > 0, sg(adm2) > 0
    Es = jnp.minimum(jnp.where(o1, V1, jnp.inf).min(-1), jnp.where(o2, V2, jnp.inf).min(-1))[..., None]
    i1, i2 = (V1 <= Es, V2 <= Es) if rule == 'sens' else (V1 == Es, V2 == Es) if rule == 'level' else ((V1 == Es) & o1, (V2 == Es) & o2)
    return (adm1 * sg(i1.astype(adm1.dtype))).sum(-1), (adm2 * sg(i2.astype(adm2.dtype))).sum(-1), Es[..., 0]


def item_compare_bits(adm1, adm2, V1, V2, rule='flip'):
    """(compare_bits on item_level_counts): more = [n1 > n2], same = [n1 = n2], backward rule `rule`."""
    sg, sig = jax.lax.stop_gradient, jax.nn.sigmoid
    n1, n2, Es = item_level_counts(adm1, adm2, V1, V2, 'win' if rule == 'win' else 'sens'); d = n1 - n2
    if rule in ('sens', 'win'):
        s = sig(d); return st_hard(d > 0, s), st_hard(d == 0, 4. * s * (1. - s))
    dv = sg(d); hm, hs = (dv > 0).astype(d.dtype), (dv == 0).astype(d.dtype); ds = d
    if rule in ('unit', 'flip'): dv = jnp.where(((sg(n1) > 0) & (sg(n2) > 0)) | ~jnp.isfinite(Es), dv, jnp.clip(dv, -1., 1.))
    if rule == 'flip': m1, m2, _ = item_level_counts(adm1, adm2, V1, V2, 'level'); ds = m1 - m2
    s = sig(dv); return st_hard(d > 0, only_if_wrong(s * (1. - s) * d, hm)), st_hard(d == 0, only_if_wrong(4. * s * (1. - s) * (1. - 2. * s) * ds, hs))


def item_count_half(l, m, ctx, wire, own, rule='flip'):
    """(lead_compare for items): l [B, h, i, k] test logits -> bits [B, h, i]. Fact pairs: the numbers of admitted partners compared (count_compare); register pairs: the lowest value among admitted partners and how many reach it, compared (the lead comparison); own channels: the
    only witness is the item itself, with no learned test (the pair block's own channels on its diagonal)."""
    sig = jax.nn.sigmoid; wire = np.asarray(wire); N = l.shape[-1]; mk = m[:, None, None, :]
    A = st_hard(l > 0, sig(l)) * mk
    if own is not None: A = jnp.where((np.asarray(own) > 0)[None, :, None, None], jnp.eye(N, dtype=l.dtype)[None, None] * mk, A)
    isnum = (wire[0::2] > 0) | (wire[1::2] > 0); ch = lambda ix: np.stack([2 * ix, 2 * ix + 1], 1).reshape(-1); fact, num = ch(np.nonzero(~isnum)[0]), ch(np.nonzero(isnum)[0]); parts = []
    if len(fact):
        Nc = A[:, fact].sum(-1); d = Nc[:, 0::2] - Nc[:, 1::2]; s = sig(d)
        parts.append(jnp.stack([st_hard(d > 0, s), st_hard(d == 0, 4. * s * (1. - s))], 2).reshape(Nc.shape))
    if len(num):
        c1, c2 = num[0::2], num[1::2]; w1, w2 = wire[c1], wire[c2]; V, P = ctx['V'], ctx['P']
        more, same = item_compare_bits(A[:, c1] * P[:, w1], A[:, c2] * P[:, w2], V[:, w1], V[:, w2], rule)
        parts.append(jnp.stack([more, same], 2).reshape(A.shape[0], len(num), N))
    return jnp.concatenate(parts, 1)[:, np.argsort(np.concatenate([fact, num]))]


def item_exists(l, m):
    """(exists_bits for items): l [B, C, i, k] -> [B, C, i]: 1 where some real partner is admitted. Backward the cold slope with one leg:
    the largest sigmoid(l) over real k (exists_bits with the single cost softplus(-l), exp(-cost) = sigmoid(l))."""
    sg = jax.lax.stop_gradient; mid = m[:, None, None, :] > 0
    return st_hard(((sg(l) > 0) & mid).any(-1), jnp.where(mid, jax.nn.sigmoid(l), 0.).max(-1))


ITEM_HIT = 8  # item_read 'hit': read channels per pointer or edge-mask output
ITEM_STORE_TABLE = ('grow', 'edge', 'fwd', 'bwd', 'fwd', 'bwd', 'fwd', 'bwd')  # item_store: pair_numbers' PREG_TABLE, on items
ITEM_STORE_TASKS = {'bellman_ford': 'plus', 'dijkstra': 'plus', 'dag_shortest_paths': 'plus', 'mst_prim': 'max'}  # the single-source weighted tasks


def item_comb(cfg):
    """item_store: the arithmetic of a route, d(u) (+) w(u, v): + for paths ('plus'), max for Prim ('max'), as pair_numbers' arithmetic."""
    return (lambda a, c: a + c) if cfg['item_store'] == 'plus' else jnp.maximum


def item_store_start(cfg, x):
    """item_store: d_0 = 0 ('plus') or -PREG_BIG ('max', the identity of max, as pair_numbers' diagonal) at the task's start node (x['s1']), absent
    (PREG_BIG) elsewhere. [B, N] float32."""
    return jnp.where((x['s1'] > 0) & (x['m'] > 0), 0. if cfg['item_store'] == 'plus' else -PREG_BIG, PREG_BIG).astype(jnp.float32)


def item_store_pairs(l, m, ctx, d, cfg):
    """item_store (pair_numbers' register pairs on items; partner axis last; each pair's first channel admits, as in pair_numbers). l [B, 2 NPREG, i, k] test logits;
    d [B, N] the built number. Values: route(i, k) = d(k) (+) W(k, i), out(i, k) = d(i) (+) W(i, k). Table ITEM_STORE_TABLE:
      grow  L(i) = min over admitted k of route(i, k) (pair_numbers' reg_min: slope to the best admitted value and to the admissions);
            d <- min(d, L); bits [L < d(i)], [L = d(i)] (reg_bits)
      edge  bits of min over admitted k of W(k, i) against d(i)
      fwd   [some admitted k: route(i, k) < d(i)], [... = d(i)] (reg_any)
      bwd   [some admitted k: out(i, k) < d(k)], [... = d(k)]
    -> bits [B, 2 NPREG, i], the new d. pair_numbers' functions are called with a witness axis of length 1."""
    sig = jax.nn.sigmoid; op = item_comb(cfg); E = lambda a: a[..., None]
    route, outr = op(d[:, None, :], ctx['Win']), op(d[:, :, None], ctx['Wout']); di, dk = jnp.broadcast_to(d[:, :, None], route.shape), jnp.broadcast_to(d[:, None, :], route.shape)
    out, dn = [], d
    for i, kind in enumerate(ITEM_STORE_TABLE):
        adm = st_hard(l[:, 2 * i] > 0, sig(l[:, 2 * i])) * m[:, None, :]
        if kind == 'grow': L1 = reg_min(E(adm), E(route))[..., 0]; dn = jnp.minimum(dn, L1); out += list(reg_bits(L1, d))
        elif kind == 'edge': out += list(reg_bits(reg_min(E(adm), E(ctx['Win']))[..., 0], d))
        elif kind == 'fwd': out += [q[..., 0] for q in reg_any(E(adm), E(route), E(di))]
        else: out += [q[..., 0] for q in reg_any(E(adm), E(outr), E(dk))]
    return jnp.stack(out, 1), dn


def item_channels(b, z, m, ctx, wire, own, rule='flip', part=None, cfg=None, d=None):
    """(the cold block's channels, for items): one loop's read of the item state z [B, D, N] -> (bits [B, D, N], or [B, D/2, N] for part 'count' or
    'exists'; the store's new number d, unchanged without item_store or for part 'exists'). With item_store the last NPREG count pairs are
    the store's (item_store_pairs), on the d the loop started with."""
    t = norm(z); h = half(b); D = b['a'].shape[1]; out = []; dn = d
    if part != 'exists':
        l = item_scores(b, t, ctx['eps'], slice(0, h))
        if cfg is not None and cfg.get('item_store'):
            hb = h - 2 * NPREG; sb, dn = item_store_pairs(l[:, hb:], m, ctx, d, cfg); out.append(jnp.concatenate([item_count_half(l[:, :hb], m, ctx, wire, own, rule), sb], 1))
        else: out.append(item_count_half(l, m, ctx, wire, own, rule))
    if part != 'count': out.append(item_exists(item_scores(b, t, ctx['eps'], slice(h, D)), m))
    return (jnp.concatenate(out, 1) if part is None else out[0]), dn


def item_rebuild(b, e, y):
    """(the cold rebuild, per item): s = e + W_o y + b_o, then z = s + MLP(norm(s)). e, y [B, D, N]."""
    s = e + lin1(y, b['o'], b['bo']); return s + lin1(jax.nn.gelu(lin1(norm(s), b['f1'], b['bf1'])), b['f2'], b['bf2'])


def item_ownmask(own, h, dtype):
    """[1, h, 1]: 1 on the own channels (bits that are input: kept from y_0 with kept_bits); 0 on the store's channels."""
    o = np.zeros(h, bool); o[:len(own)] = np.asarray(own) > 0; return jnp.array(o, dtype)[None, :, None]


def item_start(p, cfg, x):
    """(the cold start, for items): -> the item context (item_context), and the first state: the bits y_0 = 0 (with kept_bits already holding
    the own channels' bits, input, as the cold start does); with item_store the pair (y_0, d_0)."""
    sg = jax.lax.stop_gradient; b = p['blk']; h = half(b); st = cfg.get('item_store')
    ctx = item_context(p, cfg, x); wire, own, _ = item_layout(cfg, h, x['reg'].shape[1]); y = jnp.zeros_like(ctx['e']); d = item_store_start(cfg, x) if st else None
    if cfg.get('kept_bits', True):
        c0, _ = item_channels(b, item_rebuild(b, ctx['e'], y), x['m'], ctx, wire, own, 'flip', 'count', cfg, d)
        y = jnp.concatenate([sg(c0) * item_ownmask(own, h, y.dtype), y[:, h:]], 1)
    return ctx, ((y, d) if st else y)


def item_step(p, cfg, x, e, s, rule='flip'):
    """(the cold step, for items): one loop. e: item_start's context; s: the bits [B, D, N] (with item_store: (bits, d)). -> the new s, the new bits.
    With item_store, as pair_numbers' step: kept bits, then the count half on the d the loop started with, which also builds the new d."""
    b = p['blk']; h = half(b); m = x['m']; wire, own, _ = item_layout(cfg, h, x['reg'].shape[1]); st = cfg.get('item_store'); y, d = s if st else (s, None)
    if cfg.get('kept_bits', True):  # kept bits: the exists bits, once on, stay on; the count half reads the input, the own bits and the kept bits only
        L = jnp.maximum(y[:, h:], item_channels(b, item_rebuild(b, e['e'], y), m, e, wire, own, rule, 'exists', cfg, d)[0])
        cb, dn = item_channels(b, item_rebuild(b, e['e'], jnp.concatenate([y[:, :h] * item_ownmask(own, h, y.dtype), L], 1)), m, e, wire, own, rule, 'count', cfg, d)
        yn = jnp.concatenate([cb, L], 1)
    else: yn, dn = item_channels(b, item_rebuild(b, e['e'], y), m, e, wire, own, rule, None, cfg, d)
    return ((yn, dn) if st else yn), yn


def item_hit_types(ctx, x, cfg, d):
    """item_read 'hit': the read's value types, a list of (V, P) [B, i, k] (value of partner k for item i, and where it exists): none;
    every row of reg at k, ascending and descending; with item_w the edge's weight W(k, i), both ways; with item_store the route d(k) (+) W(k, i).
    A decreasing map is still a comparison: the read stays unchanged under every increasing recoding of a register."""
    T = x['reg'].shape[1]; V, P = ctx['V'], ctx['P']; ty = [(V[:, 0], P[:, 0])]
    for t in range(1, T + 1): ty += [(V[:, t], P[:, t]), (-V[:, t], P[:, t])]
    if cfg.get('item_w'): ty += [(V[:, T + 1], P[:, T + 1]), (-V[:, T + 1], P[:, T + 1])]
    if cfg.get('item_store'): r = item_comb(cfg)(d[:, None, :], ctx['Win']); ty += [(r, present(r).astype(r.dtype) * P[:, 0])]
    return ty


def item_hits(q, t, ctx, m, types):
    """item_read 'hit': [B, i, k] = sum over the ITEM_HIT read channels c of gain_c * hit_c(i, k), hit_c = [k admitted by channel c and its
    value is the lowest among the admitted] (the partner that wins item i's cold reading). Channel c's test is the block's form with its own
    maps (q: a, ba, b, bb, u, w); its value type is types[c mod len(types)]. Backward, the 'sens' level rule: the test's slope reaches every
    partner at or below the admitted lead (the ones that would win if admitted), none above it."""
    sg = jax.lax.stop_gradient; H = q['g'].shape[0]
    l = lin1(t, q['a'], q['ba'])[:, :, :, None] * lin1(t, q['b'], q['bb'])[:, :, None, :] + jnp.einsum('bikf,fc->bcik', ctx['eps'], q['u']) + q['w'][None, :, None, None]
    A = st_hard(l > 0, jax.nn.sigmoid(l)) * m[:, None, None, :]; out = 0.
    for c in range(H):
        V, P = types[c % len(types)]; a = A[:, c] * P; Es = jnp.where(sg(a) > 0, V, jnp.inf).min(-1, keepdims=True)
        out = out + q['g'][c] * a * sg((V <= Es).astype(a.dtype))
    return out


def item_read(p, cfg, x, e, s, outs):
    """logits per output from the item state s (the module docstring's read-outs, on t = norm of the rebuilt item state). item_read 'hit':
    pointer and edge-mask outputs from the winning partners (item_hits), the others as 'bilinear'."""
    st = cfg.get('item_store'); y, d = s if st else (s, None); hit = cfg['item_read'] == 'hit'
    b, m, eps = p['blk'], x['m'], e['eps']; t = norm(item_rebuild(b, e['e'], y)); D = t.shape[1]; r = p.get('iro', {}); out = {}
    bil = lambda: jnp.einsum('bdi,bdj->bij', lin1(t, r['p'], r['bp']), lin1(t, r['q'], r['bq'])) * D ** -.5
    hits = lambda name: item_hits(p['ihit'][name], t, e, m, item_hit_types(e, x, cfg, d))
    for name, kind in outs:
        if kind == 'node': lg = jnp.where(m[:, None, :] > 0, hits(name) if hit else bil() + eps @ r['e'], -1e9)
        elif kind == 'one': lg = jnp.where(m > 0, jnp.einsum('bcn,c->bn', t, p['one30'][name] if 'one30' in p else p['w_one']), -1e9)
        elif kind == 'nmask': lg = jnp.einsum('bcn,c->bn', t, p['w_mask']) + p['b_mask']
        elif kind == 'emask' and hit: l = hits(name); lg = (l + l.swapaxes(1, 2)) / 2 + p['ihit'][name]['b0']
        elif kind == 'emask': l = bil() + eps @ r['e'] + p['b_mask']; lg = (l + l.swapaxes(1, 2)) / 2
        elif kind == 'edge':
            qij = lin1(t, r['p'], r['bp'])[:, :, :, None] + lin1(t, r['p2'], r['bp2'])[:, :, None, :]
            lg = jnp.einsum('bdij,bdk->bijk', qij, lin1(t, r['q'], r['bq'])) * D ** -.5 + (eps @ r['ea'])[:, :, None, :] + (eps @ r['eb']).swapaxes(1, 2)[:, None, :, :]
            lg = jnp.where(m[:, None, None, :] > 0, lg, -1e9)
        elif kind == 'ecat': lg = jnp.einsum('bdi,bdj,dc->bijc', lin1(t, r['p'], r['bp']), lin1(t, r['q'], r['bq']), p['ec30']['w']) * D ** -.5 + p['ec30']['b'] + eps @ r['ec']
        elif kind == 'gmask': lg = ((t * m[:, None]).sum(2) / m.sum(1)[:, None]) @ p['gm30']['w'] + p['gm30']['b']
        else: raise ValueError(kind)
        out[name] = lg
    return out


# --------------------------------------------------------------------------------- two blocks (a census block beside a second block)
TWO_ROWS = 3   # two_blocks: the register rows of the second cold block's fixed layout


def two_layout(h):
    """two_blocks, second = 'cold': the cold block's count-half layout, fixed (it names no task). Fact pairs; then, for each
    register row 1..3, four channel pairs that compare that row's numbers only (the first two with fixed tests: the pair's
    own two items); then the NPREG register pairs ('max' arithmetic). -> base, wire, own, ownbit, as pn_layout."""
    P = h // 2
    nfact = P - NPREG - 4 * TWO_ROWS
    assert nfact >= 2, f'cold block too narrow: {P} channel pairs'
    wire, own = [0] * (2 * nfact), [0] * (2 * nfact)
    for r in range(1, TWO_ROWS + 1):
        wire += [r] * 8
        own += [1, 2, 2, 1, 0, 0, 0, 0]
    wire, own = np.array(wire, np.int32), np.array(own, np.int32)
    ownbit = np.concatenate([own > 0, np.repeat(np.array([t == 'edge' for t in PREG_TABLE]), 2)])
    return 2 * (P - NPREG), wire, own, ownbit


def tie_legs(b, T):
    """The last T channels of a block (the end of its best half) read with the left leg's test on both legs."""
    D = b['a'].shape[1]
    return dict(b, b=b['b'].at[:, D - T:].set(b['a'][:, D - T:]), bb=b['bb'].at[D - T:].set(b['ba'][D - T:]))


def two_count_half(b, z, m, x, R):
    t = norm(z)
    h = half(b)
    base, wire, own, _ = two_layout(h)
    xa, xc = lin(t, b['a'][:, :h], b['ba'][:h]), lin(t, b['b'][:, :h], b['bb'][:h])
    y1 = lead_compare(xa[:, :base], xc[:, :base], m, x['reg'], wire, own=own)
    y2, Rn = reg_pairs(xa[:, base:], xc[:, base:], m, x['W'], R, 'max')
    return jnp.concatenate([y1, y2], 1), Rn


def two_cold_start(b, e, x):
    """The cold block's first (bits, R): kept bits at 0, the own bits of the first pass already in."""
    sg = jax.lax.stop_gradient
    h = half(b)
    y = jnp.zeros_like(e)
    R = pn_start_reg(x, 'max')
    ownbit = jnp.array(two_layout(h)[3], e.dtype)[None, :, None, None]
    c0, _ = two_count_half(b, rebuild(b, e, y), x['m'], x, R)
    return (jnp.concatenate([sg(c0) * ownbit, y[:, h:]], 1), R)


def two_cold_step(b, ties, e, x, s):
    """The cold block, one loop (as pair_numbers): kept bits, then R, then the count half on the R the loop started with."""
    y, R = s
    h = half(b)
    ownbit = jnp.array(two_layout(h)[3], e.dtype)[None, :, None, None]
    L = jnp.maximum(y[:, h:], read_half(tie_legs(b, ties) if ties else b, rebuild(b, e, y), x['m'], 'exists'))
    cb, Rn = two_count_half(b, rebuild(b, e, jnp.concatenate([y[:, :h] * ownbit, L], 1)), x['m'], x, R)
    return (jnp.concatenate([cb, L], 1), Rn)


def two_start(p, cfg, x):
    """two_blocks: -> ((e, e2), (z, s2)): the census block's embedding and state, the second block's embedding and state."""
    e = embed(p, x)
    if cfg['second'] == 'warm':
        e2 = embed(p['second'], x)
        return (e, e2), (e, e2)
    e2 = embed(p['second'], x['cold'])
    return (e, e2), (e, two_cold_start(p['second']['blk'], e2, x['cold']))


def two_step(p, cfg, x, e, s):
    """two_blocks, one loop: the second block's loop; the census state receives the second block's new bits (cold) or its
    normalised state (warm) through a linear map that starts at zero; then the census block's loop. The second block never
    reads the census state."""
    z, s2 = s
    m = x['m']
    pair = (m[:, :, None] * m[:, None, :])[:, None]
    if cfg['second'] == 'warm':
        s2 = census_loop(p['second']['blk'], s2, m)
        z = z + lin(norm(s2), p['couple']['w'], p['couple']['b']) * pair
    else:
        s2 = two_cold_step(p['second']['blk'], int(cfg.get('tie_legs', 0)), e[1], x['cold'], s2)
        z = z + lin(s2[0], p['couple']['w'], p['couple']['b']) * pair
    return (census_loop(p['blk'], z, m), s2)


def two_read(p, cfg, x, e, s, outs):
    """two_blocks: each output's logits = the task's read-out of the census state + its second read-out of the second
    block's state (cold: rebuilt from the input embedding and the bits)."""
    z, s2 = s
    z2 = s2 if cfg['second'] == 'warm' else rebuild(p['second']['blk'], e[1], s2[0])
    return {name: readout(p, z, x['m'], kind, name) + readout(p['second'], z2, x['m'], kind, name) for name, kind in outs}

def read_all(p, cfg, x, e, s, outs):
    """Logits of every output from the loop state s. With via pairs, a pair's pointer to a third item ('edge') adds
    via_c * hit_c of the loop that made the state. The cold, compare_first and paths-table pair_numbers blocks rebuild
    the state once per output (the same values; the paper's code does so, and the order in which the outputs' slopes
    are summed in training follows it, so training is bit-for-bit the paper's on tasks with two outputs)."""
    if cfg['block'] == 'two_blocks':
        return two_read(p, cfg, x, e, s, outs)
    if cfg['block'] == 'item_memory':
        return item_read(p, cfg, x, e, s, outs)
    dp = cfg['block'] == 'dp_program'
    once = cfg['block'] == 'census' or pn_v2(cfg) or dp
    z = state(p, cfg, e, s) if once else None
    out = {}
    for name, kind in outs:
        lg = readout(p, z if once else state(p, cfg, e, s), x['m'], kind, name)
        if kind == 'edge' and (pn_v2(cfg) or dp) and s[-1]:
            lg = lg + sum(p['via'][c] * s[-1][c].swapaxes(2, 3) for c in range(len(s[-1])))   # hit [B, i, k, j] -> [B, i, j, k]
        out[name] = lg
    return out


# ------------------------------------------------------------------------------------------------------------ read-outs
def readout(p, z, m, kind, name):
    """Logits of one output from the state z."""
    t = norm(z)
    if kind == 'node':   # [B, i, j]: i points to j
        return jnp.where(m[:, None, :] > 0, jnp.einsum('bcij,c->bij', t, p['w_node']), -1e9)
    if kind == 'one':    # [B, i]
        w = p['one30'][name] if 'one30' in p else p['w_one']
        return jnp.where(m > 0, jnp.einsum('bcn,c->bn', jnp.diagonal(t, axis1=2, axis2=3), w), -1e9)
    if kind == 'ecat':   # [B, i, j, 4]
        return jnp.einsum('bcij,ck->bijk', t, p['ec30']['w']) + p['ec30']['b']
    if kind == 'gmask':  # [B]
        mm = m[:, :, None] * m[:, None, :]
        return ((t * mm[:, None]).sum((2, 3)) / mm.sum((1, 2))[:, None]) @ p['gm30']['w'] + p['gm30']['b']
    if kind == 'nmask':  # [B, i]
        return jnp.einsum('bcn,c->bn', jnp.diagonal(t, axis1=2, axis2=3), p['w_mask']) + p['b_mask']
    if kind == 'emask':  # [B, i, j], symmetric
        l = jnp.einsum('bcij,c->bij', t, p['w_mask']) + p['b_mask']
        return (l + l.swapaxes(1, 2)) / 2
    P, R = lin(t, p['P'], p['bP']), lin(t, p['Rr'], p['bR'])   # 'edge': [B, i, j, k] from (z_ik, z_kj)
    return jnp.where(m[:, None, None, :] > 0, jnp.einsum('bcik,bckj->bijk', P, R) / P.shape[1] ** .5, -1e9)


def decide(lg, kind):
    """Predicted index (pointers, one-of) or 0/1 (masks: logit > 0)."""
    return (lg > 0).astype(jnp.int32) if kind in MASKS or kind == 'gmask' else lg.argmax(-1)


# ------------------------------------------------------------------------------------------------------------ forward
def start(p, cfg, x):
    """-> (e, s): the input embedding and the loop's first state (census: z; compare_first: bits; pair_numbers: (bits, R))."""
    if cfg['block'] == 'census':
        e = embed(p, x)
        return e, e
    if cfg['block'] == 'two_blocks':
        return two_start(p, cfg, x)
    if cfg['block'] == 'dp_program':
        return dp_start(p, cfg, x)
    if cfg['block'] == 'item_memory':
        return item_start(p, cfg, x)
    if cfg['block'] == 'cold':
        return cold_start(p, cfg, x)
    if cfg['block'] == 'compare_first':
        return cf_start(p, cfg, x)
    return pn2_start(p, cfg, x) if pn_v2(cfg) else pn_start(p, cfg, x)


def step(p, cfg, x, e, s):
    if cfg['block'] == 'census':
        return census_loop(p['blk'], s, x['m'])
    if cfg['block'] == 'two_blocks':
        return two_step(p, cfg, x, e, s)
    if cfg['block'] == 'dp_program':
        return dp_step(p, cfg, x, e, s)
    if cfg['block'] == 'item_memory':
        return item_step(p, cfg, x, e, s)[0]
    if cfg['block'] == 'cold':
        return cold_step(p, cfg, x, e, s)
    if cfg['block'] == 'compare_first':
        return cf_step(p, cfg, x, e, s)
    return pn2_step(p, cfg, x, e, s) if pn_v2(cfg) else pn_step(p, cfg, x, e, s)


def state(p, cfg, e, s):
    """The pair state the read-outs read."""
    if cfg['block'] == 'census':
        return s
    return rebuild(p['blk'], e, bits(cfg, s))


def bits(cfg, s):
    if cfg['block'] == 'item_memory':
        return s[0] if cfg.get('item_store') else s
    return s[0] if cfg['block'] in ('pair_numbers', 'dp_program') else s


def built_changes(cfg, s2, s, pair0):
    """pair_numbers: per input, the real pairs at which a built number (R, and H) changed in one loop."""
    if cfg['block'] == 'dp_program':   # every built register of the program
        return sum(((s2[1][n] != s[1][n]) & pair0).sum((1, 2)) for n in DP_PROGRAMS[cfg['program']]['built'])
    c = ((s2[1] != s[1]) & pair0).sum((1, 2))
    return c + ((s2[2] != s[2]) & pair0).sum((1, 2)) if pn_v2(cfg) else c


def forward(p, cfg, x, K, outs, every=False):
    """Logits per output after K loops (training reads here). every=True: the predicted indices after each loop, a list."""
    e, s = start(p, cfg, x)
    seq = []
    for _ in range(K):
        s = step(p, cfg, x, e, s)
        if every:
            lg = read_all(p, cfg, x, e, s, outs)
            seq.append({name: decide(lg[name], kind) for name, kind in outs})
    if every:
        return seq
    return read_all(p, cfg, x, e, s, outs)


def forward_rule(p, cfg, x, kv, lo, hi, outs):
    """Logits per output with each input read after its own loop count kv [B] (lo <= kv <= hi, lo and hi static): hi loops
    run, and input i's logits are taken at loop kv[i]. Inputs never interact, so these equal forward's after kv[i] loops."""
    e, s = start(p, cfg, x)
    out = None
    for t in range(1, hi + 1):
        s = step(p, cfg, x, e, s)
        if t >= lo:
            lg = read_all(p, cfg, x, e, s, outs)
            out = lg if out is None else {k: jnp.where((kv == t).reshape((-1,) + (1,) * (v.ndim - 1)), v, out[k]) for k, v in lg.items()}
    return out


def make_rest_read(cfg, outs, ties=None):
    """The read-out rule of the resting blocks. Returns f(p, x, cap) -> dict(pred {name: [B, ...]}, loops [B], stable [B],
    per_loop {name: [L, B, ...]}, changed [L, B]). An input's loop count is the first loop whose bits (and, for
    pair_numbers, its built numbers) equal the previous loop's; it is capped at n (cold, compare_first) or 4 n
    (pair_numbers) loops for an input of n items, or at `cap`. pred is the prediction at that loop. The batch stops once
    every input has stopped."""
    pn = cfg['block'] in ('pair_numbers', 'dp_program')
    it = cfg['block'] == 'item_memory'
    ties = it if ties is None else ties   # item_memory: also '<name>#unique', 1 where an argmax output's largest logit is unique
    assert cfg['block'] in ('cold', 'compare_first', 'pair_numbers', 'dp_program', 'item_memory')

    @jax.jit
    def one(p, x, e, s):
        s2 = step(p, cfg, x, e, s)
        pair = (x['m'][:, None, :, None] * x['m'][:, None, None, :]) > 0
        lg = read_all(p, cfg, x, e, s2, outs)
        pred = {name: decide(lg[name], kind) for name, kind in outs}
        if ties:
            pred |= {name + '#unique': ((lg[name] == lg[name].max(-1, keepdims=True)).sum(-1) == 1).astype(jnp.int32)
                     for name, kind in outs if not (kind in MASKS or kind == 'gmask')}
        if it:   # item bits [B, D, N] on real items; with item_store also the built number d
            d = (bits(cfg, s2) != bits(cfg, s)) & (x['m'][:, None, :] > 0)
            c = d.sum((1, 2))
            if cfg.get('item_store'):
                c = c + ((s2[1] != s[1]) & (x['m'] > 0)).sum(1)
            return s2, pred, c
        d = (bits(cfg, s2) != bits(cfg, s)) & pair
        c = d.sum((1, 2, 3))
        if pn:
            c = c + built_changes(cfg, s2, s, pair[:, 0])
        return s2, pred, c

    def f(p, x, cap=None):
        e, s = start(p, cfg, x)
        n = np.asarray(x['m']).sum(1).astype(int)
        capv = (PAIR_NUMBER_CAP * n if pn else n) if cap is None else np.full(len(n), cap)
        L = int(capv.max())
        loops = np.zeros(len(n), int)
        per, chg, pred = {}, [], {}
        for l in range(1, L + 1):
            s, pr, c = one(p, x, e, s)
            c = np.asarray(c)
            chg.append(c)
            for name in pr:
                per.setdefault(name, []).append(np.asarray(pr[name]))
            stop = (loops == 0) & ((c == 0) | (l >= capv))
            for name in per:
                if name not in pred:
                    pred[name] = per[name][-1].copy()
                pred[name][stop] = per[name][-1][stop]
            loops[stop] = l
            if (loops > 0).all():
                break
        ch = np.stack(chg)
        stable = np.array([ch[loops[i] - 1, i] == 0 for i in range(len(n))])
        return dict(pred=pred, loops=loops, stable=stable, per_loop={k: np.stack(v) for k, v in per.items()}, changed=ch)
    return f


# ------------------------------------------------------------------------------------------------------------ loss
def outputs_loss(lg, m, y, outs):
    """Cross-entropy on the outputs (no hints), averaged over real items or pairs; binary cross-entropy for masks over the
    entries the library does not mark -1."""
    tot = 0.
    for name, kind in outs:
        tgt = y[name]
        if kind == 'gmask':
            w = (tgt >= 0).astype(lg[name].dtype)
            t, l = jnp.maximum(tgt, 0).astype(lg[name].dtype), lg[name]
            tot = tot + ((jax.nn.softplus(l) - l * t) * w).sum() / jnp.maximum(w.sum(), 1.)
            continue
        if kind == 'ecat':
            w = (tgt >= 0) * m[:, :, None] * m[:, None, :]
            nll = -jnp.take_along_axis(jax.nn.log_softmax(lg[name], -1), jnp.maximum(tgt, 0)[..., None], -1)[..., 0]
            tot = tot + (nll * w).sum() / jnp.maximum(w.sum(), 1.)
            continue
        if kind in MASKS:
            w = (tgt >= 0) * (m if kind == 'nmask' else m[:, :, None] * m[:, None, :])
            t, l = jnp.maximum(tgt, 0).astype(lg[name].dtype), lg[name]
            tot = tot + ((jax.nn.softplus(l) - l * t) * w).sum() / w.sum()
            continue
        lp = jax.nn.log_softmax(lg[name], -1)
        w = m if kind == 'node' else m[:, :, None] * m[:, None, :] if kind == 'edge' else jnp.ones(tgt.shape)
        nll = -jnp.take_along_axis(lp, jnp.maximum(tgt, 0)[..., None], -1)[..., 0]
        tot = tot + (nll * w).sum() / w.sum()
    return tot


def loss_fn(p, cfg, x, y, K, outs):
    return outputs_loss(forward(p, cfg, x, K, outs), x['m'], y, outs)
