"""The constraint-problem generators: shared machinery (one module per class with SPEC, `reference` (the standard
label routine), an independent `brute`, HAND cases and its own `sample`).

An instance is a dict (unpadded, one per sample):
  n      number of items (real nodes for the trainer), items 0..n-1 in a shuffled order
  A, adj [n, n] uint8: the class's two pair inputs (adj's diagonal may carry an item mark; see each module)
  s      index of the global item (the trainer's s1 mark); the answer is asked there only
  unsat  [n] int16: -1 everywhere except at s, where it is 1 (unsatisfiable) or 0 (satisfiable)
  meta   dict: kind (0 random, 1 twin), pair (twin pair id within the split, -1), depth (-1 if none), nv (variables),
         and class-specific numbers; plus 'cons', the constraint list in the module's own form (for brute force and checks).
pack() pads to M items: n int16, A uint8, adj uint8, pos float32 (i / n on real items, 0 on padding: the library's
positions; the trainer randomises them in training and zeroes them under no_pos), s int32, unsat int16, and the meta
columns kind int8, pair int32, depth int32, nv int16 (extra keys the trainer ignores).
Seeds (SEED_BASE = 730,000,000): SEED_BASE + 1_000_000 * code + 1_000 * size + chunk (training) and
SEED_BASE + 1_000_000 * code + 999_000 + tag (evaluation), so classes and splits never share a stream."""
import hashlib, json, time
from pathlib import Path
import numpy as np

SEED_BASE = 730_000_000


def seed_train(code, size, chunk):
    assert 0 < code < 1000 and size < 999 and chunk < 1000
    return SEED_BASE + 1_000_000 * code + 1000 * size + chunk


def seed_eval(code, tag):
    assert 0 < code < 1000 and 0 <= tag < 1000
    return SEED_BASE + 1_000_000 * code + 999_000 + tag


def K_rule(n, c=3):
    """The training loop rule K(n) = ceil(log2 n) + c on n items."""
    return int(n - 1).bit_length() + c


def digest(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode()); h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def finish(n, A, adj, g, label, rng, meta, cons):
    """Shuffle the items of an instance built with the global item last (index n - 1) -> the instance dict."""
    perm = rng.permutation(n)  # new index of old item i is inv[i]; A_new[inv[i], inv[j]] = A[i, j]
    inv = np.empty(n, np.int64); inv[perm] = np.arange(n)
    A2, adj2 = A[np.ix_(perm, perm)], adj[np.ix_(perm, perm)]
    s = int(inv[g])
    un = np.full(n, -1, np.int16); un[s] = int(label)
    return dict(n=n, A=A2.astype(np.uint8), adj=adj2.astype(np.uint8), s=s, unsat=un, meta=dict(meta, perm=perm), cons=cons)


def pack(rows, M):
    num = len(rows)
    d = dict(n=np.array([r['n'] for r in rows], np.int16), A=np.zeros((num, M, M), np.uint8), adj=np.zeros((num, M, M), np.uint8),
             pos=np.zeros((num, M), np.float32), s=np.array([r['s'] for r in rows], np.int32), unsat=np.full((num, M), -1, np.int16),
             kind=np.array([r['meta']['kind'] for r in rows], np.int8), pair=np.array([r['meta']['pair'] for r in rows], np.int32),
             depth=np.array([r['meta']['depth'] for r in rows], np.int32), nv=np.array([r['meta']['nv'] for r in rows], np.int16))
    for i, r in enumerate(rows):
        n = r['n']; assert n <= M
        d['A'][i, :n, :n], d['adj'][i, :n, :n], d['unsat'][i, :n] = r['A'], r['adj'], r['unsat']
        d['pos'][i, :n] = (np.arange(n) / n).astype(np.float32)
    return d


def label_stats(rows):
    lab = np.array([int(r['unsat'][r['s']]) for r in rows]); kind = np.array([r['meta']['kind'] for r in rows])
    dep = np.array([r['meta']['depth'] for r in rows])
    out = dict(num=len(rows), unsat_share=float(lab.mean()), twin_share=float(kind.mean()),
               items=dict(min=int(min(r['n'] for r in rows)), max=int(max(r['n'] for r in rows))))
    if (kind == 1).any():
        out['twin_unsat_share'] = float(lab[kind == 1].mean())
        dt = dep[(kind == 1) & (dep >= 0)]
        if len(dt): out['twin_depth'] = dict(min=int(dt.min()), median=float(np.median(dt)), max=int(dt.max()))
    if (kind == 0).any():
        out['random_unsat_share'] = float(lab[kind == 0].mean())
        dr = dep[(kind == 0) & (dep >= 0)]
        if len(dr): out['random_unsat_depth'] = dict(min=int(dr.min()), median=float(np.median(dr)), max=int(dr.max()))
    return out


def write(dest, name, rows, M, extra_meta=None):
    """rows -> dest/<name>.npz (padded to M items) and its manifest dest/<name>.json; returns the manifest."""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    d = pack(rows, M)
    np.savez_compressed(dest / f'{name}.npz', **d)
    meta = dict(split=name, M=M, sha256=digest(d), shapes={k: list(v.shape) for k, v in d.items()}, numpy=np.__version__,
                label_stats=label_stats(rows), **(extra_meta or {}))
    (dest / f'{name}.json').write_text(json.dumps(meta))
    return meta
