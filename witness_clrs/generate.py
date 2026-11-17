"""CLRS-30 data from DeepMind's own library (dm-clrs 2.0.3), saved as padded numpy arrays without hints.

  python -m witness_clrs.generate <out dir> <task> <split>        split: test | val | train:<size> | fresh:<seed>:<count>

Splits, as the library builds them:
- test: build_sampler(task, seed=CLRS30['test']['seed'] = 3, num_samples=32 * multiplier, length=64), default sampler
  arguments, in generation order (clrs/_src/dataset.py). val: the same with the 'val' row (seed 2, 32 samples, 16 items).
- train: the library's training protocol (clrs/examples/run.py, create_samplers): sizes 4, 7, 11, 13 and 16, edge
  probability p drawn per graph from 0.1, ..., 0.9 (halved for articulation_points, bridges and mst_kruskal, as run.py
  does), needles of 1 to 8 characters for the string matchers (which train at the single length 20). run.py samples on the
  fly; we pre-generate a pool of 80,000 inputs per size, in chunks of 8,000 with seeds 10^7 (task index + 1) + 1000 size
  + chunk (task index in tasks.TASKS).
- fresh:<seed>:<count>: more inputs drawn as the test split is drawn, at another seed.
The string matchers use 80 items at test and 20 at val (5/4 of 64 and 16); optimal_bst has length + 1 items;
segments_intersect always 4. A json manifest with the arrays' sha256 is written beside each split file.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .tasks import GRAPH_TASKS, TASKS

MATCHERS = ('naive_string_matcher', 'kmp_matcher')
LENGTH_NEEDLE = -8
SIZES = [4, 7, 11, 13, 16]
P_TRAIN = tuple(0.1 + 0.1 * i for i in range(9))
HALF_P = ('articulation_points', 'bridges', 'mst_kruskal')
CHUNK = 8000


def split_length(task, split):
    import clrs
    L = clrs.CLRS30['test' if split == 'fresh' else split]['length']
    return L * 5 // 4 if task in MATCHERS else L


def train_sizes(task):
    return [max(SIZES) * 5 // 4] if task in MATCHERS else list(SIZES)


def nodes(task, L):
    return 4 if task == 'segments_intersect' else L + 1 if task == 'optimal_bst' else L


def pack_items(fb, M=None):
    """Inputs and outputs of the non-graph tasks, padded to M items (default: no padding)."""
    import clrs
    T = clrs.Type
    n = np.asarray(next(d for d in fb.features.inputs if d.name == 'pos').data).shape[1]
    M = n if M is None else M
    padn = lambda x, v, k: np.pad(x, [(0, 0)] + [(0, M - n)] * k + [(0, 0)] * (x.ndim - 1 - k), constant_values=v)
    N = np.asarray(fb.outputs[0].data).shape[0]
    out = dict(n=np.full(N, n, np.int16))
    for d in fb.features.inputs:
        a = np.asarray(d.data)
        if d.location == 'graph':
            assert d.type_ == T.SCALAR, (d.name, d.type_)
            out['x_' + d.name] = a.astype(np.float64)
        elif d.type_ == T.SCALAR:
            out['x_' + d.name] = padn(a.astype(np.float64), 0, 1)
        elif d.type_ == T.MASK:
            assert np.isin(a, (0, 1)).all(), d.name
            out['x_' + d.name] = padn(a.astype(np.uint8), 0, 1)
        elif d.type_ == T.CATEGORICAL:
            assert np.isin(a, (0, 1)).all() and (a.sum(-1) == 1).all(), d.name
            out['x_' + d.name] = padn(a.astype(np.uint8), 0, 1)
        else:
            raise ValueError((d.name, d.location, d.type_))
    for d in fb.outputs:
        a = np.asarray(d.data)
        if d.type_ in (T.POINTER, T.SHOULD_BE_PERMUTATION):
            out[d.name] = padn(np.rint(a).astype(np.int16), -1, a.ndim - 1)
        elif d.type_ == T.MASK_ONE:
            assert np.isin(a, (0, 1)).all() and (a.sum(-1) == 1).all(), d.name
            out[d.name] = a.argmax(-1).astype(np.int16)
        elif d.type_ == T.MASK:
            assert np.isin(a, (-1, 0, 1)).all(), d.name
            out[d.name] = np.rint(a).astype(np.int16) if d.location == 'graph' else padn(np.rint(a).astype(np.int16), -1, a.ndim - 1)
        elif d.type_ == T.CATEGORICAL:
            masked = (a == -1).any(-1)
            assert (masked | ((a.sum(-1) == 1) & np.isin(a, (0, 1)).all(-1))).all(), d.name
            out[d.name] = padn(np.where(masked, -1, a.argmax(-1)).astype(np.int16), -1, a.ndim - 2)
        else:
            raise ValueError((d.name, d.location, d.type_))
    return out


def pack(fb, M):
    if not any(d.name == 'adj' for d in fb.features.inputs):
        return pack_items(fb)
    ins = {d.name: np.asarray(d.data) for d in fb.features.inputs}
    outs = {d.name: np.asarray(d.data) for d in fb.outputs}
    N, n = ins['pos'].shape
    padn = lambda x, v: np.pad(x, [(0, 0)] + [(0, M - n)] * (x.ndim - 1), constant_values=v)
    out = dict(n=np.full(N, n, np.int16), pos=padn(ins['pos'].astype(np.float32), 0), A=padn(ins['A'].astype(np.float32), 0),
               adj=padn(ins['adj'].astype(np.uint8), 0))
    if 's' in ins:
        out['s'] = ins['s'].argmax(-1).astype(np.int16)
    for name, d in outs.items():
        if name == 'topo_head':
            out[name] = d.argmax(-1).astype(np.int16)
        else:
            out[name] = padn(np.rint(d).astype(np.int16), -1)
    return out


def _is_collinear_2d(point_1, point_2, point_3, eps):
    """The library's collinearity test with np.cross on 2-vectors written out (numpy 2.5 no longer takes 2-vectors)."""
    for point in (point_1, point_2, point_3):
        if point.shape != (2,):
            raise ValueError(f'Point {point} is not a 2D vector.')
    v_1, v_2 = point_2 - point_1, point_3 - point_1
    return bool(abs(v_1[0] * v_2[1] - v_1[1] * v_2[0]) < eps)


def sample(task, num, length, seed, **kw):
    import clrs
    if task in ('graham_scan', 'jarvis_march'):
        from clrs._src import samplers as S
        S._is_collinear = _is_collinear_2d
    sampler, _ = clrs.build_sampler(task, num_samples=num, length=length, seed=seed, **kw)
    return sampler.next(None)  # the whole dataset in generation order, as dataset.py does


def digest(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode()); h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def make(task, what, per_size=80000):
    import clrs
    graph = task in GRAPH_TASKS
    if what.startswith('fresh:'):
        _, seed, num = what.split(':')
        L = split_length(task, 'fresh')
        fb = sample(task, int(num), L, int(seed))
        return (pack(fb, L) if graph else pack_items(fb)), dict(seed=int(seed), num=int(num), length=L)
    if what in ('test', 'val'):
        c = clrs.CLRS30[what]
        num = c['num_samples'] * clrs.CLRS_30_ALGS_SETTINGS[task]['num_samples_multiplier']
        L = split_length(task, what)
        fb = sample(task, num, L, c['seed'])
        return (pack(fb, L) if graph else pack_items(fb)), dict(seed=c['seed'], num=num, length=L)
    L = int(what.split(':')[1])
    assert L in train_sizes(task), (task, L)
    parts, seeds = [], []
    if graph:
        total, M = per_size, 16
        kw = dict(p=tuple(np.array(P_TRAIN) / 2) if task in HALF_P else P_TRAIN)
    else:
        total = per_size * len(SIZES) // len(train_sizes(task))  # the matchers: one size, the same pool
        M = nodes(task, max(train_sizes(task)))
        kw = dict(p=P_TRAIN, length_needle=LENGTH_NEEDLE)  # the library drops what a sampler does not take
    for ci in range((total + CHUNK - 1) // CHUNK):
        seed = 10 ** 7 * (TASKS.index(task) + 1) + 1000 * L + ci
        fb = sample(task, min(CHUNK, total - ci * CHUNK), L, seed, **kw)
        parts.append(pack(fb, M) if graph else pack_items(fb, M))
        seeds.append(seed)
    meta = dict(seeds=seeds, num=total, length=L, p=[float(q) for q in kw['p']])
    if not graph:
        meta['length_needle'] = LENGTH_NEEDLE
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, meta


def main(out_dir, task, what, per_size=80000):
    t0 = time.time()
    d, meta = make(task, what, int(per_size))
    name = f"fresh{what.split(':')[1]}" if what.startswith('fresh:') else what.replace(':', '_L')
    dest = Path(out_dir) / task
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f'{name}.npz', **d)
    import clrs
    meta |= dict(task=task, split=what, sha256=digest(d), shapes={k: list(v.shape) for k, v in d.items()},
                 seconds=round(time.time() - t0, 1), clrs_version=getattr(clrs, '__version__', '?'), numpy=np.__version__)
    (dest / f'{name}.json').write_text(json.dumps(meta))
    return meta


if __name__ == '__main__':
    print(json.dumps(main(*sys.argv[1:])))
