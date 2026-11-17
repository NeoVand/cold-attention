"""Train the organs: the cold block beside a frozen language model (organ.py holds the generators, the read-ins and the
reads of the trained organs).

  python organ/train.py features kinship <out dir> --model Qwen/Qwen2.5-7B --closed <organ>/closed.json [--large]
  python organ/train.py features comparison <out dir> --names <organ>/names.json
  python organ/train.py kinship '<job json>'                                          # one organ
  python organ/train.py comparison '<job json>'

Features. The frozen model (at the pinned revision, input embeddings in float16) reads each split once, with the readers of
organ.py, and the reads are stored as float16:
  kinship     <split>/s_key.npy [statements, 3, d] (the first-named and the second-named mention's key and the wording key),
              q_key.npy [contexts, questions, 2, d], read.npy [statements] (the model's own reading of who the parent is),
              meta.npz; and stats.npz (centring and spread of the keys, from the first 20,000 training statements). closed.json
              (the closed-class words, found on the training sentences without labels) ships with every kinship organ. --large:
              the training split uses organ.TRAIN_WORDINGS_LARGE, as for the 7B and 1.5B organs (wordings 1 and 6 never seen).
  comparison  <split>/fs_L0.npy [facts, tokens, d] and qs_L0.npy [contexts, questions, tokens, d] (the input embeddings of the
              fact lines and the question lines), meta.npz. names.json ships with the comparison organs.
Nothing in a read uses a label.

Training. Each update draws a context size n and a batch of questions of that size from the training split, and a loop
count K = ceil(log2(items)) + loop_c plus a draw from 0..loop_extra; the loss is the cross-entropy of the yes/no answer read
after K loops and again `late` loops later. AdamW (learning rate 1e-3, warm-up over min(2,000, updates / 10), cosine decay to
1e-5, weight decay 0.01 except on the shape numbers and the bit mix), gradients clipped to norm 1. The released parameters
are the mean of the snapshots taken from half way, one every 250 updates.

The block (block 'mono', the released organs): `bits` kept bits per pair of items; a loop turns bit c of (u, v) on when some
real item k passes one test with both (u, k) and (k, v); bits never turn off; a test reads the pair's input planes (any
function of them) and a non-negative mix of the pair's kept bits, so it cannot ask for a bit to be off; the answer is a
non-negative read of the bits at the question's pair, so a bit can only push it towards yes. block 'warm' (the control):
the general block (pair state of width D that carries strengths) on the same input planes.

Job keys, comparison: seed, out, arm ('l0': the frozen model's comparator bits, from feats; 'names': the same bits computed
from the names; 'oracle': the true 'same name' relations, from feats), feats, block ('mono' or 'warm'), tie, pos_read, D,
mlp, steps, batch, lr, late, loop_c, loop_extra, swa_every, name_splits.
Job keys, kinship: seed, out, arm ('states': the stored reads, from feats; 'names': the people's ids), feats, feats_train,
planes ('pooled': two classes, the model's reading pooled over each wording class; 'direction': each sentence's own
reading; 'wording': one class per wording; 'none'), links ('within' or 'partner'), block ('mono' or 'warm'), bits, D, mlp,
steps, batch, lr, wd, late, loop_c, loop_extra, wordings, splits, restarts, restart_at, restart_loss (a start whose mean
training loss at update restart_at is above restart_loss is drawn again, up to `restarts` times; seed + 1000 a for attempt a).
The released organs: comparison {"arm": "l0", "block": "mono", "tie": true, "pos_read": true, "late": 10, "loop_c": 2,
"loop_extra": 1, "steps": 4000, "batch": 16, "D": 6}; its warm control {"arm": "l0", "block": "warm", "steps": 25000,
"batch": 32, "D": 128}; kinship {"arm": "states", "planes": "pooled" (7B, 1.5B) or "wording" (0.5B), "links": "within" or
"partner", "steps": 4000}; the kinship warm control {"arm": "states", "block": "warm", "planes": "wording", "steps": 25000}.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import organ as O  # noqa: E402

NAME_PLANES = ((1, 1), (1, 3), (1, 5), (1, 6), (3, 1), (3, 5), (5, 1), (5, 3), (5, 5), (5, 6), (6, 1), (6, 5))   # arm 'names': the comparators the read-in keeps at layer 0 (fact names at tokens 1 and 5, question names at 3 and 6)
SHRINK = .1


def _jax():
    import jax
    import jax.numpy as j
    import optax
    from witness_clrs import model as M     # lin, norm, soft_clip, exists_bits, census_loop, init_params; full float32 products
    return jax, j, optax, M


# ============================================================================================ data: contexts and features
def comparison_contexts(seed, sizes, per_size, n_names=1400, q_per=8):
    rng = np.random.default_rng(seed)
    return [O.comparison_context(rng, n, O.FAMILIES[i % len(O.FAMILIES)], n_names, q_per) for n in sizes for i in range(per_size)]


def kinship_contexts(seed, sizes, per_size, wordings=None, q_per=8):
    rng = np.random.default_rng(seed)
    return [O.kinship_context(rng, n, O.STRUCTS[i % len(O.STRUCTS)], q_per=q_per, wordings=wordings) for n in sizes for i in range(per_size)]


REVISIONS = {'Qwen/Qwen2.5-7B': 'd149729398750b98c0af14eb82c78cfe92750796', 'Qwen/Qwen2.5-1.5B': '8faed761d45a263340a0528343f099c05c9a4323',
             'Qwen/Qwen2.5-0.5B': '060db6499f32faf8b98477b0a26969ef7d8b9987'}


def slots(facts, cls):
    """-> s1, s2 [n]: the person named first and second in each statement."""
    fwd = np.array([O.WORDINGS[c][1] for c in cls])
    return np.where(fwd, facts[:, 0], facts[:, 1]), np.where(fwd, facts[:, 1], facts[:, 0])


def kinship_features(out, model, closed, train_wordings=None, splits=None, bs=64):
    """The frozen model reads the kinship splits (organ.KinshipReader) -> <out>/<split>/{s_key,q_key,read}.npy, meta.npz, and
    <out>/stats.npz. closed: the closed-class words ({'statements': [...], 'questions': [...]})."""
    out = Path(out); rd = O.KinshipReader(O.FrozenModel(model, REVISIONS[model]), closed)
    for name in splits or list(O.KINSHIP_SPLITS):
        a = O.KINSHIP_SPLITS[name]
        cs = kinship_contexts(a['seed'], a['sizes'], a['per_size'], train_wordings) if name == 'train' else O.kinship_split(name)
        r = rd.read([O.kinship_text(c) for c in cs], bs); f = out / name; f.mkdir(parents=True, exist_ok=True)
        np.save(f / 's_key.npy', r['s_key'].astype(np.float16)); np.save(f / 'q_key.npy', np.asarray(r['q_key']).astype(np.float16)); np.save(f / 'read.npy', r['read'])
        slot = np.concatenate([np.stack(slots(c['facts'], c['cls']), 1) for c in cs]).astype(np.int32)
        np.savez(f / 'meta.npz', n=np.array([c['n'] for c in cs]), off=r['off'], facts=np.concatenate([c['facts'] for c in cs]), cls=np.concatenate([c['cls'] for c in cs]), slot=slot,
                 struct=np.array([c['struct'] for c in cs]), q=np.stack([c['q'] for c in cs]), label=np.stack([c['label'] for c in cs]),
                 kind=np.stack([c['kind'] for c in cs]), depth=np.stack([c['depth'] for c in cs]))
        print(name, len(cs), 'contexts', 'every statement and question has two mentions' if r['found'] else 'SOME STATEMENT OR QUESTION DOES NOT HAVE TWO MENTIONS', flush=True)
    k = np.asarray(np.load(out / 'train' / 's_key.npy', mmap_mode='r')[:20000], np.float32); d = k.shape[-1]
    np.savez(out / 'stats.npz', mu_p=k[:, :2].reshape(-1, d).mean(0), sd_p=k[:, :2].reshape(-1, d).std(0) + 1e-3, mu_w=k[:, 2].mean(0), sd_w=k[:, 2].std(0) + 1e-3)


def comparison_features(out, names, model='Qwen/Qwen2.5-0.5B', splits=None):
    """The frozen model's input embeddings of the comparison splits' lines (organ.ComparisonReader) -> <out>/<split>/fs_L0.npy,
    qs_L0.npy, meta.npz. names: names.json (single-token names of the model's vocabulary)."""
    out = Path(out); rd = O.ComparisonReader(O.FrozenModel(model, REVISIONS[model]), names); nm = names['names']
    for name in splits or list(O.COMPARISON_SPLITS):
        cs = O.comparison_split(name); f = out / name; f.mkdir(parents=True, exist_ok=True); fs, qs = [], []
        for c in cs:
            X, _, _ = rd.read([(nm[a], nm[b]) for a, b in c['facts']], [(nm[x], nm[y]) for x, y in c['q']])
            fs.append(X[0, 1:, :rd.TF].astype(np.float16)); qs.append(X[:, 0, :rd.TQ].astype(np.float16))
        np.save(f / 'fs_L0.npy', np.concatenate(fs)); np.save(f / 'qs_L0.npy', np.stack(qs)); n = np.array([c['n'] for c in cs])
        np.savez(f / 'meta.npz', n=n, off=np.concatenate([[0], np.cumsum(n)]), facts=np.concatenate([c['facts'] for c in cs]), family=np.array([c['family'] for c in cs]),
                 q=np.stack([c['q'] for c in cs]), label=np.stack([c['label'] for c in cs]), kind=np.stack([c['kind'] for c in cs]), depth=np.stack([c['depth'] for c in cs]))
        print(name, len(cs), 'contexts', flush=True)


# ============================================================================================ the comparison organ
class ComparisonFeatures:
    """A split of stored states for one layer: meta.npz (n, off, facts, family, q, label, kind, depth), fs_L<layer>.npy and
    qs_L<layer>.npy."""

    def __init__(self, folder, name, layer=0):
        f = Path(folder) / name; m = np.load(f / 'meta.npz'); self.m = {k: m[k] for k in m.files}; self.name = name
        self.fs = np.load(f / f'fs_L{layer}.npy', mmap_mode='r'); self.qs = np.load(f / f'qs_L{layer}.npy', mmap_mode='r')   # [facts, T, d], [C, Q, Tq, d]
        self.T = max(self.fs.shape[1], self.qs.shape[2]); self.d = self.fs.shape[2]; self.C, self.Q = self.qs.shape[:2]

    def item(self, c, q, n_pad=None):
        """-> X [n_pad, T, d] (item 0 the question), tok [n_pad, T], real [n_pad], rel [4, n_pad, n_pad], label."""
        m = self.m; n = int(m['n'][c]); o = int(m['off'][c]); P = n_pad or n + 1
        X = np.zeros((P, self.T, self.d), np.float32); tok = np.zeros((P, self.T), bool); real = np.zeros(P, bool)
        X[0, :self.qs.shape[2]] = self.qs[c, q]; tok[0, :self.qs.shape[2]] = True
        X[1:n + 1, :self.fs.shape[1]] = self.fs[o:o + n]; tok[1:n + 1, :self.fs.shape[1]] = True; real[:n + 1] = True
        rel = np.zeros((4, P, P), np.float32); rel[:, :n + 1, :n + 1] = O.comparison_relations(m['facts'][o:o + n], m['q'][c, q])
        return X, tok, real, rel, float(m['label'][c, q])


class ComparisonNames:
    """Arm 'names': the comparator bits computed from the names (bit (i, j) at (u, v): the name at token i of item u is the
    name at token j of item v), for the comparators the layer-0 read-in keeps (NAME_PLANES)."""

    def __init__(self, name, seed, sizes, per_size, n_names=1400):
        cs = comparison_contexts(seed, sizes, per_size, n_names); n = np.array([c['n'] for c in cs]); self.name = name
        self.m = dict(n=n, off=np.concatenate([[0], np.cumsum(n)]), facts=np.concatenate([c['facts'] for c in cs]), family=np.array([c['family'] for c in cs]),
                      q=np.stack([c['q'] for c in cs]), label=np.stack([c['label'] for c in cs]), kind=np.stack([c['kind'] for c in cs]), depth=np.stack([c['depth'] for c in cs]))
        self.C, self.Q = len(cs), self.m['q'].shape[1]

    def item(self, c, q, n_pad=None):
        m = self.m; n = int(m['n'][c]); o = int(m['off'][c]); P = n_pad or n + 1; nm = -np.ones((P, 7), np.int64)
        nm[0, 3], nm[0, 6] = m['q'][c, q]; nm[1:n + 1, 1] = m['facts'][o:o + n, 0]; nm[1:n + 1, 5] = m['facts'][o:o + n, 1]
        bits = np.stack([(nm[:, i][:, None] == nm[:, jj][None, :]) & (nm[:, i][:, None] >= 0) for i, jj in NAME_PLANES]) & ~np.eye(P, dtype=bool)[None]
        real = np.zeros(P, bool); real[:n + 1] = True
        return bits.astype(np.float32), np.zeros((P, 1), bool), real, np.zeros((1, 1, 1), np.float32), float(m['label'][c, q])


def comparison_batch(split, picks, n_pad):
    parts = [split.item(c, q, n_pad) for c, q in picks]
    return tuple(np.stack([p[i] for p in parts]) for i in range(4)) + (np.array([p[4] for p in parts], np.float32),)


def make_stats(tr, shrink=SHRINK):
    """Label-free numbers from the training lines: the mean state per token position (facts and questions apart), one spread
    per dimension, a whitening matrix (pooled over positions), and a floor under which a centred state counts as 'nothing
    here' (a token that is the same in every line, such as the words of the sentence pattern at layer 0)."""
    _, j, _, _ = _jax()
    d, T = tr.d, tr.T; fa = np.asarray(tr.fs[:min(len(tr.fs), 20000)], np.float32); qa = np.asarray(tr.qs[:min(len(tr.qs), 2500)], np.float32).reshape(-1, tr.qs.shape[2], d)
    fmu, qmu = fa.mean(0), qa.mean(0); z = fa - fmu; sd = z.reshape(-1, d).std(0) + 1e-3; z = (z / sd).reshape(-1, d).astype(np.float64)
    w, V = np.linalg.eigh(z.T @ z / len(z)); Wh = ((V / np.sqrt(np.maximum(w, 0.) + shrink * w.mean())) @ V.T).astype(np.float32)
    nr = np.linalg.norm(z.astype(np.float32) @ Wh, axis=1); floor = .05 * float(np.quantile(nr, .9))
    pad = lambda v: np.concatenate([v, np.zeros((T - v.shape[0], d), np.float32)]) if v.shape[0] < T else v
    return dict(fmu=j.array(pad(fmu)), qmu=j.array(pad(qmu)), sd=j.array(sd), Wh=j.array(Wh), floor=j.array(floor, j.float32))


def unit_keys(X, tok, st):
    """-> one unit key per token [B, N, T, d]; zero where there is no token or the centred state is under the floor."""
    _, j, _, _ = _jax()
    isq = (j.arange(X.shape[1]) == 0)[None, :, None, None]   # item 0 is the question
    x = (j.where(isq, X - st['qmu'], X - st['fmu']) / st['sd']) @ st['Wh']
    nr = j.sqrt((x * x).sum(-1, keepdims=True)); return j.where((nr > st['floor']) & tok[..., None], x / j.maximum(nr, 1e-9), 0.)


def comparators(X, tok, real, st, planes, theta):
    """-> bits [B, M, N, N] (exact 0/1): comparator m = (i, j) compares token position i of item u with token position j of
    item v, cosine above theta[m]. No comparator on an item against itself."""
    jax, j, _, _ = _jax()
    k = unit_keys(X, tok, st); ki, kj = j.moveaxis(k[:, :, planes[:, 0]], 2, 1), j.moveaxis(k[:, :, planes[:, 1]], 2, 1)   # [B, M, N, d]
    cos = j.matmul(ki, j.swapaxes(kj, 2, 3)); N = X.shape[1]
    pair = (real[:, None, :, None] & real[:, None, None, :]) & ~j.eye(N, dtype=bool)[None, None]
    return jax.lax.stop_gradient(((cos > theta[None, :, None, None]) & pair).astype(X.dtype)), cos


def calibrate(tr, st, rng, picks=384):
    """Choose the comparators and their thresholds without labels. For each pair of token positions, over the item pairs of
    some training contexts: sort the cosines; a comparator is kept when the upper part of the distribution has a clear gap
    (0.05 or more) with at least 20 pairs above it, a separate cluster of matches; its threshold sits just under that
    cluster (0.03 under its lowest member), because the largest cosine among non-matches grows with the number of facts."""
    _, j, _, _ = _jax()
    n0 = max(int(n) for n in np.unique(tr.m['n'])); pool = np.flatnonzero(tr.m['n'] >= n0 // 2); allq = [(int(c), q) for c in pool for q in range(tr.Q)]   # contexts of the upper half of the training sizes
    X, tok, real, _, _ = comparison_batch(tr, [allq[i] for i in rng.permutation(len(allq))[:picks]], n0 + 1); k = np.asarray(unit_keys(j.array(X), j.array(tok), st)); T = k.shape[2]
    has = (k * k).sum(-1) > 0; off = ~np.eye(k.shape[1], dtype=bool)[None]; planes, theta, info = [], [], []   # padded items have no key
    for i in range(T):
        if not has[:, :, i].any(): continue
        for jj in range(T):
            ok = has[:, :, i][:, :, None] & has[:, :, jj][:, None, :] & off
            if ok.sum() < 200: continue
            c = np.sort(np.einsum('bur,bvr->buv', k[:, :, i], k[:, :, jj])[ok]); lo = int(.7 * len(c)); hi = len(c) - 20
            if hi <= lo: continue
            g = np.diff(c)[lo:hi]; a = int(g.argmax())
            if g[a] < .05: continue
            edge = float(c[lo + a + 1]); th = edge - min(.03, float(g[a]) / 2); planes.append((i, jj)); theta.append(th)
            info.append(dict(positions=[i, jj], threshold=round(th, 4), gap=round(float(g[a]), 4), lowest_match=round(edge, 4), highest_other=round(float(c[lo + a]), 4), share_on=round(float((c > th).mean()), 5)))
    return np.array(planes, np.int32).reshape(-1, 2), np.array(theta, np.float32), info


def warm_init(key, cfg, M):
    """The warm control: the input embedding of the M planes and four flags, a read-out at (question, question), and the
    general block."""
    jax, j, _, Mo = _jax()
    ks = iter(jax.random.split(key, 8)); g = lambda *s: jax.random.normal(next(ks), s) * s[0] ** -.5; D = cfg['D']; F = M + 4
    p = dict(E=g(F, D), eb=j.zeros((D,)), e1=g(F, 32), eb1=j.zeros((32,)), e2=g(32, D), w_out=jax.random.normal(next(ks), (D,)) * D ** -.5, b_out=j.zeros(()))
    p['blk'] = Mo.init_params(next(ks), dict(block='census', width=D, mlp=cfg['mlp'], edge_mlp=32), [])['blk']
    return p


def planes_in(bits, real):
    """-> the input planes of a pair [B, M + 4, N, N]: the comparator bits, 'the first item is the question', 'the second item
    is the question', 'the same item', and 'both items are real'."""
    _, j, _, _ = _jax()
    B, _, N, _ = bits.shape; r = real.astype(bits.dtype); isq = (j.arange(N) == 0).astype(bits.dtype)
    pair = r[:, :, None] * r[:, None, :]; one = lambda a: (a * pair)[:, None]
    return j.concatenate([bits, one(isq[None, :, None] * j.ones((B, N, N))), one(isq[None, None, :] * j.ones((B, N, N))), one(j.eye(N)[None] * j.ones((B, 1, 1))), one(j.ones((B, N, N)))], 1)


def warm_embed(p, bits, real):
    jax, _, _, Mo = _jax()
    F = planes_in(bits, real); return Mo.lin(F, p['E'], p['eb']) + Mo.lin(jax.nn.gelu(Mo.lin(F, p['e1'], p['eb1'])), p['e2'])


def mono_init(key, M, Cn, tie=False):
    """Cn kept bits per pair. Each leg of a join: any function of the pair's input planes (a linear map and a small MLP), and
    a non-negative mix of the pair's kept bits (softplus of P). tie: the two legs are one test."""
    jax, j, _, _ = _jax()
    ks = iter(jax.random.split(key, 12)); F = M + 4; g = lambda *s: jax.random.normal(next(ks), s) * s[0] ** -.5
    leg = lambda: dict(E=g(F, Cn), e1=g(F, 32), b1=j.zeros((32,)), e2=g(32, Cn), b=j.full((Cn,), -1.), P=jax.random.normal(next(ks), (Cn, Cn)) * .5 - .5)   # mix weights start near 0.5
    legs = dict(a=leg()) if tie else dict(a=leg(), c=leg())
    return dict(legs, th_t=j.linspace(0., 1., Cn), w_out=jax.random.normal(next(ks), (Cn,)) * Cn ** -.5, b_out=j.zeros(()))


def mono_step(p, m, F, L):
    """One loop: bit c of (u, v) turns on if some real k passes test a_c at (u, k) and test c_c at (k, v); bits stay on."""
    jax, j, _, Mo = _jax()
    f = lambda q: Mo.lin(F, q['E'], q['b']) + Mo.lin(jax.nn.gelu(Mo.lin(F, q['e1'], q['b1'])), q['e2']) + Mo.lin(L, jax.nn.softplus(q['P']))
    xa = f(p['a']); xc = f(p['c']) if 'c' in p else xa
    return j.maximum(L, Mo.exists_bits(xa, xc, m, Mo.soft_clip(p['th_t'], 0., 1.)))


def comparison_run(p, cfg, bits, real, K, at=None):
    """-> logit [B] after K loops; at: the logits after each of those loop counts [len(at), B]."""
    jax, j, _, Mo = _jax()
    out = []
    if cfg['block'] == 'mono':
        F = planes_in(bits, real); m = real.astype(F.dtype); s = j.zeros((F.shape[0], p['w_out'].shape[0]) + F.shape[2:], F.dtype)
        if cfg.get('pos_read'): read = lambda s: s[:, :, 0, 0] @ jax.nn.softplus(p['w_out']) - jax.nn.softplus(p['b_out'])   # a kept bit can only speak for yes
        else: read = lambda s: s[:, :, 0, 0] @ p['w_out'] + p['b_out']
        step = lambda s: mono_step(p, m, F, s)
    else:
        e = warm_embed(p, bits, real); m = real.astype(e.dtype); s = e
        read = lambda s: Mo.norm(s)[:, :, 0, 0] @ p['w_out'] + p['b_out']
        step = lambda s: Mo.census_loop(p['blk'], s, m)
    for k in range(K):
        s = step(s)
        if at is not None and (k + 1) in at: out.append(read(s))
    return j.stack(out) if at is not None else read(s)


def train_comparison(job):
    """-> (final params, weight-averaged params, log). Writes the organ to job['out']."""
    jax, j, optax, _ = _jax()
    T0 = time.time(); say = lambda *a: print(*a, f'[{time.time() - T0:.0f}s]', flush=True)
    arm = job['arm']; oracle, names = arm == 'oracle', arm == 'names'
    cfg = dict(block=job.get('block', 'mono'), D=job.get('D', 128), mlp=job.get('mlp', 312), pos_read=bool(job.get('pos_read')))
    assert cfg['block'] in ('mono', 'warm'), cfg['block']
    out = Path(job['out']); out.mkdir(parents=True, exist_ok=True)
    name_splits = {k: dict(O.COMPARISON_SPLITS[k], **job.get('name_splits', {}).get(k, {})) for k in O.COMPARISON_SPLITS}
    tr = ComparisonNames('train', **name_splits['train']) if names else ComparisonFeatures(job['feats'], 'train', 0)
    rng = np.random.default_rng(job['seed']); steps = job['steps']; B = job.get('batch', 32); st = planes = theta = None; M = len(NAME_PLANES) if names else 4
    if not oracle and not names:   # the read-in: label-free statistics and comparators of the frozen model's layer-0 states
        st = make_stats(tr); planes, theta, info = calibrate(tr, st, np.random.default_rng(1000 + job['seed'])); M = len(theta)
        say('comparators kept:', M); (out / 'calib.json').write_text(json.dumps(dict(layer=0, comparators=info), indent=1))
        if M == 0: raise SystemExit('no comparator has a separate cluster of matches')
        np.savez(out / 'readin.npz', planes=planes, theta=theta, **{k: np.asarray(v) for k, v in st.items()}); planes, theta = j.array(planes), j.array(theta)
    p = mono_init(jax.random.PRNGKey(job['seed']), M, cfg['D'] // 2, bool(job.get('tie'))) if cfg['block'] == 'mono' else warm_init(jax.random.PRNGKey(job['seed']), cfg, M)
    bits_of = (lambda X, tok, real, rel: rel) if oracle else (lambda X, tok, real, rel: X) if names else (lambda X, tok, real, rel: comparators(X, tok, real, st, planes, theta)[0])
    sched = optax.warmup_cosine_decay_schedule(0., job.get('lr', 1e-3), min(2000, steps // 10), steps, 1e-5)
    decay = jax.tree_util.tree_map_with_path(lambda path, v: 'th_' not in jax.tree_util.keystr(path) and "['P']" not in jax.tree_util.keystr(path), p)   # no decay on the shape numbers or on the bit mix
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01, mask=decay)); os_ = opt.init(p)
    late = int(job.get('late', 0))

    def loss_fn(p, X, tok, real, rel, lab, K):
        if late:
            lg = comparison_run(p, cfg, bits_of(X, tok, real, rel), real, K + late, at=(K, K + late))
            return optax.sigmoid_binary_cross_entropy(lg, j.broadcast_to(lab, lg.shape)).mean(), ((lg[0] > 0) == (lab > .5)).mean()
        lg = comparison_run(p, cfg, bits_of(X, tok, real, rel), real, K)
        return optax.sigmoid_binary_cross_entropy(lg, lab).mean(), ((lg > 0) == (lab > .5)).mean()
    fns = {}

    def step_fn(K):
        if K not in fns:
            @jax.jit
            def f(p, os_, X, tok, real, rel, lab):
                (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(p, X, tok, real, rel, lab, K); up, os2 = opt.update(g, os_, p)
                return optax.apply_updates(p, up), os2, l, a
            fns[K] = f
        return fns[K]
    by_n = {int(n): np.flatnonzero(tr.m['n'] == n) for n in np.unique(tr.m['n'])}; sizes = sorted(by_n); swa, n_swa, log = None, 0, []
    for it in range(steps):
        n = int(rng.choice(sizes)); picks = [(int(rng.choice(by_n[n])), int(rng.integers(tr.Q))) for _ in range(B)]
        K = O.loops_for(n + 1, job.get('loop_c', 3)) + int(rng.integers(0, job.get('loop_extra', 3) + 1)); X, tok, real, rel, lab = comparison_batch(tr, picks, 17)
        p, os_, l, a = step_fn(K)(p, os_, X, tok, real, rel, lab)
        if it >= steps // 2 and (it + 1) % max(1, job.get('swa_every', 250)) == 0:
            n_swa += 1; swa = p if swa is None else jax.tree_util.tree_map(lambda s, q: s + (q - s) / n_swa, swa, p)
        if it % max(1, steps // 50) == 0 or it == steps - 1:
            log.append(dict(step=it, loss=float(l), acc=float(a), n=n, K=K, t=round(time.time() - T0, 1))); say(json.dumps(log[-1]))
        if not np.isfinite(float(l)): raise SystemExit('loss is not finite')
    swa = p if swa is None else swa
    save(out, swa, p, dict(task='comparison', block=cfg['block'], arm=arm, kept_bits=cfg['D'] // 2 if cfg['block'] == 'mono' else None, train=job, snapshots=n_swa), log)
    return p, swa, log


# ============================================================================================ the kinship organ
def design_class(cls, planes):
    return cls.astype(np.int64) if planes == 'wording' else np.array([0 if O.WORDINGS[c][1] else 1 for c in cls], np.int64) if planes in ('direction', 'pooled') else np.zeros(len(cls), np.int64)


class KinshipNames:
    """Arm 'names': the 'same person' planes from the people's ids."""

    def __init__(self, name, seed, sizes, per_size, wordings, planes, lay):
        self.cs = kinship_contexts(seed, sizes, per_size, wordings); self.name = name; self.planes = planes; self.lay = lay
        self.n = np.array([c['n'] for c in self.cs]); self.Q = len(self.cs[0]['label'])

    def item(self, ci, q, n_pad=None):
        c = self.cs[ci]; n = c['n']; N = n_pad or 2 * n + 2; sent, slot, real = O.frame(n, N); ids = -np.ones(N, np.int64); cls = np.zeros(N, np.int64)
        ids[:2] = c['q'][q]; s1, s2 = slots(c['facts'], c['cls']); ids[2:2 * n + 2:2], ids[3:2 * n + 2:2] = s1, s2; cls[2:2 * n + 2] = np.repeat(design_class(c['cls'], self.planes), 2)
        return self.lay.pid((ids[:, None] == ids[None]) & (ids[:, None] >= 0), sent, slot, cls, real), real, float(c['label'][q])


class KinshipFeatures:
    """Arm 'states': a split of stored reads (kinship_features). Nothing here uses a label."""

    def __init__(self, folder, name, planes, cal):
        f = Path(folder) / name; m = np.load(f / 'meta.npz'); self.m = {k: m[k] for k in m.files}; self.name = name; self.planes = planes; self.cal = cal; self.lay = cal['lay']
        self.sk = np.load(f / 's_key.npy', mmap_mode='r'); self.qk = np.load(f / 'q_key.npy', mmap_mode='r'); self.read = np.load(f / 'read.npy'); self.n = self.m['n']; self.Q = self.m['q'].shape[1]
        if planes == 'pooled':   # one direction per wording class, the model's reading pooled over the class's sentences in this split
            self.pool = O.pooled_directions(np.asarray(self.sk[:, 2]), self.read, cal)[0]

    def item(self, ci, q, n_pad=None):
        m = self.m; cal = self.cal; n = int(m['n'][ci]); o = int(m['off'][ci]); N = n_pad or 2 * n + 2; sent, slot, real = O.frame(n, N); k = np.asarray(self.sk[o:o + n], np.float32)
        P = np.zeros((N, k.shape[-1]), np.float32); P[:2] = np.asarray(self.qk[ci, q], np.float32); P[2:2 * n + 2] = k[:, :2].reshape(2 * n, -1); has = np.abs(P).sum(-1) > 0
        U = (P - cal['mu_p']) / cal['sd_p']; U = U / np.maximum(np.linalg.norm(U, axis=-1, keepdims=True), 1e-9) * has[:, None]; sp = ((U @ U.T) > cal['theta_p']) & has[:, None] & has[None]; cl = np.zeros(n, np.int64)
        if self.planes == 'wording':
            W = (k[:, 2] - cal['mu_w']) / cal['sd_w']; W = W / np.maximum(np.linalg.norm(W, axis=-1, keepdims=True), 1e-9); cl = (W @ cal['centres'].T).argmax(1)
        elif self.planes == 'direction': cl = np.where(self.read[o:o + n] == 0, 1, 0)   # class 0: the model reads the first-named as the parent (undecided counts as 0)
        elif self.planes == 'pooled': cl = self.pool[o:o + n]
        cls = np.zeros(N, np.int64); cls[2:2 * n + 2] = np.repeat(cl, 2)
        return self.lay.pid(sp, sent, slot, cls, real), real, float(m['label'][ci, q])


def kinship_calibrate(folder, planes, links='within'):
    """Label-free numbers from the training statements: centring and spread of the keys; the 'same person' threshold (just
    under the cluster of matches in the sorted cosines); for planes 'wording' and 'pooled', the wording classes (the
    distinct closed-class keys)."""
    st = np.load(Path(folder) / 'stats.npz'); cal = {k: st[k].astype(np.float32) for k in st.files}; m = np.load(Path(folder) / 'train' / 'meta.npz'); sk = np.load(Path(folder) / 'train' / 's_key.npy', mmap_mode='r')
    big = np.flatnonzero(m['n'] >= m['n'].max() // 2)[:400]; cos = []
    for ci in big:
        o = int(m['off'][ci]); n = int(m['n'][ci]); U = (np.asarray(sk[o:o + n, :2], np.float32).reshape(-1, sk.shape[-1]) - cal['mu_p']) / cal['sd_p']; U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9); S = U @ U.T
        cos.append(S[np.triu_indices(len(U), 1)])
    c = np.sort(np.concatenate(cos)); lo = int(.5 * len(c)); g = np.diff(c)[lo:]; a = int(g.argmax()); edge = float(c[lo + a + 1]); cal['theta_p'] = edge - min(.03, float(g[a]) / 2)
    info = dict(person=dict(threshold=round(cal['theta_p'], 4), gap=round(float(g[a]), 4), lowest_match=round(edge, 4), highest_other=round(float(c[lo + a]), 4), share_on=round(float((c > cal['theta_p']).mean()), 5)))
    K = 2 if planes in ('direction', 'pooled') else 1
    if planes in ('wording', 'pooled'):
        W = (np.asarray(sk[:20000, 2], np.float32) - cal['mu_w']) / cal['sd_w']; W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9); centres = []
        for w in W:   # a new class whenever a key matches no class yet
            if not centres or float((np.stack(centres) @ w).max()) < .98: centres.append(w)
            if len(centres) > 64: break
        cal['centres'] = np.stack(centres); Kw = len(centres); K = Kw if planes == 'wording' else K
        S = cal['centres'] @ cal['centres'].T; np.fill_diagonal(S, -1); info['wording'] = dict(classes=Kw, highest_cosine_between_classes=round(float(S.max()), 4))
    cal['lay'] = O.Layout(K, links); return cal, info


def f_in(q, X):
    """The input part of a test for each pattern: any function of the pattern's planes. X [n_pat, P] -> [n_pat, Cn]."""
    jax, _, _, _ = _jax()
    return X @ q['E'] + q['b'] + jax.nn.gelu(X @ q['e1'] + q['b1']) @ q['e2']


def kinship_run(p, pat, pid, real, K, at):
    """The block; the answer at the pair (X, Y) = (0, 1). -> the logits after the loop counts `at` [len(at), B]."""
    jax, j, _, Mo = _jax()
    I = j.moveaxis(f_in(p['a'], pat)[pid], -1, 1); m = real.astype(I.dtype); mix = jax.nn.softplus(p['a']['P']); L = j.zeros_like(I); out = []
    read = lambda L: L[:, :, 0, 1] @ jax.nn.softplus(p['w_out']) - jax.nn.softplus(p['b_out'])
    for k in range(K):
        x = I + Mo.lin(L, mix)
        L2 = j.maximum(L, Mo.exists_bits(x, x, m, Mo.soft_clip(p['th_t'], 0., 1.)))
        if (k + 1) in at: out.append(read(L2))
        L = L2
    return j.stack(out)


def kinship_warm_run(p, pat, pid, real, K):
    """The control: the general block on the same planes, read at (X, Y). -> logit [B] after K loops."""
    jax, j, _, Mo = _jax()
    F = j.moveaxis(pat[pid], -1, 1); e = Mo.lin(F, p['E'], p['eb']) + Mo.lin(jax.nn.gelu(Mo.lin(F, p['e1'], p['eb1'])), p['e2']); m = real.astype(e.dtype); s = e
    for _ in range(K):
        s = Mo.census_loop(p['blk'], s, m)
    return Mo.norm(s)[:, :, 0, 1] @ p['w_out'] + p['b_out']


def train_kinship(job):
    """-> (final params, weight-averaged params, log). Writes the organ to job['out']."""
    jax, j, optax, _ = _jax()
    T0 = time.time(); say = lambda *a: print(*a, f'[{time.time() - T0:.0f}s]', flush=True)
    out = Path(job['out']); out.mkdir(parents=True, exist_ok=True); planes = job.get('planes', 'wording'); wordings = job.get('wordings', list(range(O.NW)))
    states = job.get('arm', 'names') == 'states'; warm = job.get('block') == 'warm'
    Cn = job.get('bits', 3); steps = job.get('steps', 4000); B = job.get('batch', 16); late = int(job.get('late', 10))
    splits = {k: dict(O.KINSHIP_SPLITS[k], **job.get('splits', {}).get(k, {})) for k in O.KINSHIP_SPLITS}
    if states:
        ftr = job.get('feats_train', job['feats']); cal, info = kinship_calibrate(ftr, planes, job.get('links', 'within')); lay = cal['lay']
        say('read-in', json.dumps(info)); (out / 'calib.json').write_text(json.dumps(info, indent=1)); tr = KinshipFeatures(ftr, 'train', planes, cal)
        np.savez(out / 'readin.npz', **{k: v for k, v in cal.items() if k != 'lay'})
    else:
        lay = O.Layout(O.NW if planes == 'wording' else 2 if planes in ('direction', 'pooled') else 1, job.get('links', 'within'))
        tr = KinshipNames('train', wordings=splits['train'].get('wordings', wordings), planes=planes, lay=lay, **{k: v for k, v in splits['train'].items() if k != 'wordings'})
    pat = j.array(lay.pat); rng = np.random.default_rng(job['seed'])
    wcfg = dict(D=job.get('D', 128), mlp=job.get('mlp', 312)) if warm else None
    fresh = lambda a: warm_init(jax.random.PRNGKey(job['seed'] + 1000 * a), wcfg, lay.P - 4) if warm else mono_init(jax.random.PRNGKey(job['seed'] + 1000 * a), lay.P - 4, Cn, True)
    p = fresh(0)
    sched = optax.warmup_cosine_decay_schedule(0., job.get('lr', 1e-3), min(2000, steps // 10), steps, 1e-5)
    decay = jax.tree_util.tree_map_with_path(lambda path, v: 'th_' not in jax.tree_util.keystr(path) and "['P']" not in jax.tree_util.keystr(path), p)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=job.get('wd', 0.01), mask=decay)); os_ = opt.init(p)

    def loss_fn(q, pid, real, lab, K):
        lg = kinship_warm_run(q, pat, pid, real, K)[None] if warm else kinship_run(q, pat, pid, real, K + late, at=(K, K + late) if late else (K,))
        return optax.sigmoid_binary_cross_entropy(lg, j.broadcast_to(lab, lg.shape)).mean(), ((lg[0] > 0) == (lab > .5)).mean()
    fns = {}

    def step_fn(K):
        if K not in fns:
            @jax.jit
            def f(q, os_, pid, real, lab):
                (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(q, pid, real, lab, K); up, os2 = opt.update(g, os_, q)
                return optax.apply_updates(q, up), os2, l, a
            fns[K] = f
        return fns[K]
    stack = lambda parts: (np.stack([x[0] for x in parts]), np.stack([x[1] for x in parts]), np.array([x[2] for x in parts], np.float32))
    by_n = {int(n): np.flatnonzero(tr.n == n) for n in np.unique(tr.n)}; sizes = sorted(by_n); swa, n_swa, log = None, 0, []
    restarts, r_at, r_loss = int(job.get('restarts', 0)), int(job.get('restart_at', 2000)), float(job.get('restart_loss', .5)); attempt, it, ema, checks = 0, -1, None, []
    while it + 1 < steps:
        it += 1
        n = int(rng.choice(sizes)); picks = [(int(rng.choice(by_n[n])), int(rng.integers(tr.Q))) for _ in range(B)]
        K = O.loops_for(2 * n + 2, job.get('loop_c', 3 if warm else 2)) + int(rng.integers(0, job.get('loop_extra', 3 if warm else 1) + 1))
        pid, real, lab = stack([tr.item(c, q, 34) for c, q in picks])   # padded to 34 mentions (16 statements)
        p, os_, l, a = step_fn(K)(p, os_, pid, real, lab)
        if it >= steps // 2 and (it + 1) % max(1, job.get('swa_every', 250)) == 0:
            n_swa += 1; swa = p if swa is None else jax.tree_util.tree_map(lambda s_, q: s_ + (q - s_) / n_swa, swa, p)
        if it % max(1, steps // 50) == 0 or it == steps - 1:
            log.append(dict(step=it, attempt=attempt, loss=float(l), acc=float(a), n=n, K=K, t=round(time.time() - T0, 1))); say(json.dumps(log[-1]))
        if not np.isfinite(float(l)): raise SystemExit('loss is not finite')
        ema = float(l) if ema is None else .99 * ema + .01 * float(l)
        if it + 1 == r_at and restarts:   # the restart rule, on the training loss alone
            checks.append(dict(attempt=attempt, update=it + 1, training_loss=round(ema, 4), kept=bool(ema <= r_loss or attempt >= restarts))); say('restart rule', json.dumps(checks[-1]))
            if ema > r_loss and attempt < restarts:
                attempt += 1; p = fresh(attempt); os_ = opt.init(p); swa, n_swa, it, ema = None, 0, -1, None
    swa = p if swa is None else swa
    save(out, swa, p, dict(task='kinship', block='warm' if warm else 'mono', arm=job.get('arm', 'names'), planes=planes, links=job.get('links', 'within'),
                           kept_bits=None if warm else Cn, classes=lay.K, input_planes=lay.P, attempts=attempt + 1, restart_checks=checks, train=job, snapshots=n_swa), log)
    return p, swa, log


# ============================================================================================ save, command line
def flat(tree):
    """{'a/b': array} of a parameter tree (the released files' names)."""
    out = {}

    def walk(t, pre):
        for k, v in t.items():
            if isinstance(v, dict): walk(v, f'{pre}{k}/')
            else: out[f'{pre}{k}'] = np.asarray(v, np.float32)
    walk(tree, ''); return out


def save(out, swa, final, cfg, log):
    """The weight-averaged parameters (the released ones) in out/model.safetensors, the final ones in out/final/, the config
    and the training log."""
    from safetensors.numpy import save_file
    out = Path(out); (out / 'final').mkdir(parents=True, exist_ok=True)
    save_file(flat(swa), str(out / 'model.safetensors')); save_file(flat(final), str(out / 'final' / 'model.safetensors'))
    (out / 'config.json').write_text(json.dumps(cfg, indent=1)); (out / 'progress.jsonl').write_text('\n'.join(json.dumps(r) for r in log) + '\n')


if __name__ == '__main__':
    cmd, arg = sys.argv[1], lambda k: sys.argv[sys.argv.index(k) + 1]
    if cmd == 'features' and sys.argv[2] == 'kinship':
        kinship_features(sys.argv[3], arg('--model'), json.loads(Path(arg('--closed')).read_text()), O.TRAIN_WORDINGS_LARGE if '--large' in sys.argv else None)
    elif cmd == 'features' and sys.argv[2] == 'comparison':
        comparison_features(sys.argv[3], json.loads(Path(arg('--names')).read_text()))
    elif cmd == 'kinship':
        train_kinship(json.loads(sys.argv[2]))
    elif cmd == 'comparison':
        train_comparison(json.loads(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
