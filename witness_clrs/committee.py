"""Committee-chosen training graphs for strongly connected components.

A committee is the three seeds of the cold block with positions compared (the released models
cold-attention-clrs30-variants/positions_compared_only/compare_first/strongly_connected_components/seed0..2). A graph is
disputed when two members give a different answer (pointer argmax) on some node after n loops, n the graph's nodes.

The walk, per training size L: start from a standard graph, change one edge at a time (add it if absent, remove it if
present; the edges from np.random.default_rng([1188, L, start]), 30 of them); after each change read every member; keep
the graph the first time two members differ; give up after 30 changes. The starts are the second half of the standard
pool (rows 40,000 to 79,999), then more standard graphs from the same sampler (chunks 10 to 14 of the pool's seed
formula), at most 80,000; the walk stops after the block of starts in which 40,000 graphs are kept, and keeps the first
40,000 in start order. The new pool of size L: the standard pool's rows 0 to 39,999, then the kept graphs (labelled by the
library's routine), then the standard pool's own rows of the same index if fewer than 40,000 were kept. Training then uses
witness_clrs.train with the cold block's configuration on the new pools.

  python -m witness_clrs.committee starts <data dir> <out dir> <L>          # the extra standard starts (needs dm-clrs)
  python -m witness_clrs.committee walk <models dir> <data dir> <out dir> <L>   # the committee's walk (a GPU)
  python -m witness_clrs.committee pack <data dir> <out dir> <L>            # labels and the new pool (needs dm-clrs)

The paper's pools were walked on an NVIDIA L4: reads on other hardware can differ on near ties, and so keep other graphs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

TASK = 'strongly_connected_components'
NAME = 'scc_id'
FLIPS, SEED_POOL, HALF, PER, M = 30, 1188, 40000, 80000, 16
EXTRA_CHUNKS = range(10, 15)


def eb_for(n):
    """Graphs per read batch, fixed per size (padded with copies of the batch's first graph, so a graph's answers do not
    depend on which other graphs share its batch)."""
    return {4: 16384, 7: 8192, 11: 4096, 13: 2048, 16: 2048}.get(n, 2048)


def flips_for(seed, L, s):
    rng = np.random.default_rng([seed, L, s])
    return np.array([rng.choice(L, 2, replace=False) for _ in range(FLIPS)], np.int8)


def disputed(P):
    """P [members, G, n] -> [G] bool: some two members differ on some node."""
    return (P != P[:1]).any(0).any(-1)


class Reader:
    """answers(A, pos, K): every member's pointer after K loops on graphs A [G, n, n] (0/1, own diagonal), positions pos [n]
    -> int16 [members, G, n]."""

    def __init__(self, members):
        import jax
        import jax.numpy as jnp
        from .data import model_inputs
        from .model import forward
        from .tasks import OUTPUTS
        self.members, self.fns, self.graph_evals, self.seconds = members, {}, 0, 0.
        self.cfg = members[0][1]['model']
        assert all(c['model'] == self.cfg for _, c in members)
        self._jax, self._jnp, self._inputs, self._forward, self._outs = jax, jnp, model_inputs, forward, OUTPUTS[TASK]

    def fn(self, K):
        if K not in self.fns:
            cfg, outs, fw, jnp = self.cfg, self._outs, self._forward, self._jnp
            self.fns[K] = self._jax.jit(lambda p, x: fw(p, cfg, x, K, outs)[NAME].argmax(-1).astype(jnp.int16))
        return self.fns[K]

    def answers(self, A, pos, K, eb=None):
        t0 = time.time()
        G, n = A.shape[0], A.shape[-1]
        eb, f = eb or eb_for(n), self.fn(K)
        out = np.zeros((len(self.members), G, n), np.int16)
        for i in range(0, G, eb):
            blk = A[i:i + eb]
            g = len(blk)
            if g < eb:
                blk = np.concatenate([blk, np.repeat(blk[:1], eb - g, 0)])
            Ap = blk.astype(np.float32)
            adj = Ap.copy()
            adj[:, np.arange(n), np.arange(n)] = 1.
            d = dict(n=np.full(eb, n, np.int16), pos=np.repeat(pos[None, :n].astype(np.float32), eb, 0), A=Ap,
                     adj=(adj != 0).astype(np.uint8), scc_id=np.zeros((eb, n), np.int16))
            x = {k: self._jnp.array(v) for k, v in self._inputs(self.cfg, d, np.arange(eb)).items()}
            for m, (p, _) in enumerate(self.members):
                out[m, i:i + g] = np.asarray(f(p, x))[:g, :n]
        self.graph_evals += G
        self.seconds += time.time() - t0
        return out


def walk(reader, A0, pos, L, flips, target=None, block=None, log=None):
    """The walk on starts A0 [S, L, L] with flips [S, 30, 2]; blocks of starts in order, stopping after the block in which
    `target` graphs are kept. -> dict(first [S] (changes at the first dispute; 0 gave up; -1 not run), A [S, L, L] (the kept
    graph where first > 0), P [S, members, L] (the answers there), run (starts run))."""
    S = len(A0)
    block = block or 4 * eb_for(L)
    first = np.full(S, -1, np.int16)
    keptA = np.zeros((S, L, L), np.uint8)
    keptP = np.zeros((S, len(reader.members), L), np.int16)
    run = 0
    for b0 in range(0, S, block):
        ix = np.arange(b0, min(b0 + block, S))
        cur = A0[ix].astype(np.float32).copy()
        live = np.arange(len(ix))
        first[ix] = 0
        for r in range(FLIPS):
            if not len(live):
                break
            fi = flips[ix[live], r]
            cur[live, fi[:, 0], fi[:, 1]] = 1. - cur[live, fi[:, 0], fi[:, 1]]
            P = reader.answers(cur[live], pos, L)
            dis = disputed(P)
            hit = live[dis]
            first[ix[hit]] = r + 1
            keptA[ix[hit]] = cur[hit].astype(np.uint8)
            keptP[ix[hit]] = P[:, dis].transpose(1, 0, 2)
            live = live[~dis]
        run = int(ix[-1]) + 1
        k = int((first > 0).sum())
        if log:
            log(dict(L=L, starts_run=run, kept=k, graph_evals=reader.graph_evals, seconds=round(reader.seconds, 1)))
        if target is not None and k >= target:
            break
    return dict(first=first, A=keptA, P=keptP, run=run)


def extra_starts(L):
    """More standard graphs of L nodes from the pool's sampler and seed formula, chunks 10 to 14 (the pool used 0 to 9)."""
    from . import generate as G
    parts, seeds = [], []
    for ci in EXTRA_CHUNKS:
        seed = 10 ** 7 * (G.TASKS.index(TASK) + 1) + 1000 * L + ci
        parts.append(G.pack(G.sample(TASK, G.CHUNK, L, seed, p=G.P_TRAIN), M))
        seeds.append(seed)
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}, seeds


def run_walk(models, pool, extra, L, cap=PER, target=HALF):
    """The committee's walk at size L. models: [(params, config)] of the three members; pool: the standard pool of size L;
    extra: the extra starts. -> dict(A kept graphs [K, L, L], start, changes, answers, disputed_nodes, starts_run, starts_tried)."""
    A0 = np.concatenate([(pool['A'][HALF:, :L, :L] != 0), (extra['A'][:, :L, :L] != 0)]).astype(np.uint8)[:cap]
    pos = pool['pos'][HALF, :L]
    assert (pool['pos'][HALF:, :L] == pos).all() and (extra['pos'][:, :L] == pos).all()
    fl = np.stack([flips_for(SEED_POOL, L, s) for s in range(len(A0))])
    rd = Reader(models)
    w = walk(rd, A0, pos, L, fl, target=target)
    hit = np.flatnonzero(w['first'] > 0)[:target]
    tried = int(hit[-1]) + 1 if len(hit) >= target else w['run']
    P = w['P'][hit]
    return dict(A=w['A'][hit], start=hit.astype(np.int32), changes=w['first'][hit].astype(np.int8), answers=P,
                disputed_nodes=(P != P[:, :1]).any(1).sum(-1).astype(np.int16), starts_run=w['run'], starts_tried=tried)


def library_row(A):
    """A kept graph [L, L] (0/1, own diagonal) -> its packed row, labelled by the library's routine (padded to 16 nodes)."""
    from clrs._src import probing
    from clrs._src.algorithms import graphs
    L = len(A)
    A = A.astype(np.float32)
    sid, probes = graphs.strongly_connected_components(A)
    pos = np.asarray(probes['input']['node']['pos']['data'])
    adj = probing.graph(np.copy(A))
    pad = lambda x, v: np.pad(x, [(0, M - L)] * x.ndim, constant_values=v)
    return dict(n=np.int16(L), pos=pad(pos.astype(np.float32), 0), A=pad(A, 0), adj=pad(adj.astype(np.uint8), 0),
                scc_id=pad(np.rint(sid).astype(np.int16), -1))


def new_pool(pool, kept):
    """The standard pool's first half, the kept graphs (library labels), then the standard rows of the same index."""
    K = len(kept['A'])
    assert K <= HALF and len(pool['n']) == PER
    rows = [library_row(a) for a in kept['A']]
    new = {k: np.stack([r[k] for r in rows]) if K else pool[k][:0] for k in pool}
    return {k: np.concatenate([pool[k][:HALF], new[k].astype(pool[k].dtype), pool[k][HALF + K:]]) for k in pool}


def main(cmd, *args):
    from .data import load_split
    from .generate import digest
    if cmd == 'starts':
        data, out, L = Path(args[0]), Path(args[1]), int(args[2])
        d, seeds = extra_starts(L)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f'extra_L{L}.npz', **d)
        return dict(L=L, seeds=seeds, sha256=digest(d))
    if cmd == 'walk':
        from .weights import load
        models_dir, data, out, L = Path(args[0]), Path(args[1]), Path(args[2]), int(args[3])
        members = [load(models_dir / f'seed{s}') for s in range(3)]
        r = run_walk(members, load_split(data / TASK / f'train_L{L}.npz'), load_split(out / f'extra_L{L}.npz'), L)
        np.savez_compressed(out / f'kept_L{L}.npz', **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        return dict(L=L, kept=len(r['A']), starts_run=r['starts_run'], starts_tried=r['starts_tried'])
    if cmd == 'pack':
        data, out, L = Path(args[0]), Path(args[1]), int(args[2])
        kept = load_split(out / f'kept_L{L}.npz')
        d = new_pool(load_split(data / TASK / f'train_L{L}.npz'), kept)
        (out / TASK).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / TASK / f'train_L{L}.npz', **d)
        return dict(L=L, kept=len(kept['A']), sha256=digest(d))
    raise SystemExit(__doc__)


if __name__ == '__main__':
    print(json.dumps(main(*sys.argv[1:])))
