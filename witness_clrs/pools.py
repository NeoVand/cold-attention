"""The two further training pools of the depth study's shared blocks (configs/shared/eleven_*.json and
general_and_cold.json read them as <data dir>/deep and <data dir>/multi_entry), and that study's deeper test graphs.

  python -m witness_clrs.pools deep <out dir> <task> train:<size> | big:<size>
  python -m witness_clrs.pools multi_entry <out dir> strongly_connected_components train:<size>
  python -m witness_clrs.pools tests <out dir> <task> deep128 | deep256 | lib128 | chain64 | dfs96 | dfs128 | dfs256
  python -m witness_clrs.pools depth <out dir> <task> <split>     depth labels of <out dir>/<task>/<split>.npz
  python -m witness_clrs.pools copy <data dir> <out dir> <task>   the standard val, test and fresh files, unchanged

Each command writes <out dir>/<task>/<name>.npz in generate.py's layout and a json manifest with the arrays' sha256
(generate.digest). The library tasks' answers come from the library itself: the task's sampler (dm-clrs 2.0.3,
num_samples -1, track_max_steps False) with only the function that draws the input replaced, drawing on the sampler's own
random state, so every answer keeps the library's conventions and tie rules.

Deep pool (topological_sort, strongly_connected_components, bellman_ford, mst_prim).
- train:<size>, sizes 4, 7, 11, 13 and 16, padded to 16: 80,000 graphs, the standard pool's first 40,000 (generate.py's
  chunks 0 to 4), then 40,000 graphs drawn with a set answer depth.
- big:<size>, sizes 24, 28 and 32, padded to 32: 40,000 graphs drawn with a set answer depth.
- Chunks of 8,000, seeds 840,000,000 + 1,000,000 code + 1,000 size + chunk (code: topological_sort 1,
  strongly_connected_components 2, bellman_ford 3, mst_prim 4).
- A draw of n nodes picks a target depth D, uniform in 1..n-1 (in the test graphs deep128 and deep256: ceil(n/2)..n-1),
  builds the graph on nodes 0..n-1 and shuffles the labels (node i gets label lab[i], lab a random permutation).
  - bellman_ford, mst_prim (undirected, weighted, source s = the spine's start): a spine 0 - 1 - ... - D; the other nodes
    attach as legs of the spine ('caterpillar', to spine nodes 0..D-1) or as branches (to any node of depth below D); or a
    branch tree plus 1 to 3 random extra edges ('shortcut', so the answer tree may be shallower than D). Self-loops with
    chance p drawn from 0.1..0.9; weights sqrt(u_ij u_ji + 1e-3), u uniform on [0, 1), as the library draws them.
  - topological_sort (acyclic): D + 1 layers, one spine node per layer (the spine is a path), the other nodes in random
    layers, each outside layer 0 with a parent in the layer before (so its longest path from a source is its layer);
    extra edges between adjacent layers with chance q (0, 0.1, 0.3 or 0.5); 1 to 3 long edges that skip two or more layers.
  - strongly_connected_components: a directed ring of D + 1 nodes; the other nodes in rings of random size (1 is a single
    node, 2 a two-cycle); 0 to 2 chords inside each ring of 4 or more; the rings in a random order, with edges between them
    only forward (chance 0.1, 0.3 or 0.6 per ordered pair, 1 or 2 edges), so no two rings merge; self-loops with chance p.

Multi-entry pool (strongly_connected_components): train:<size>, sizes 4 to 16, 80,000 graphs padded to 16. Chunks 0 to 4 are
the standard pool's; chunks 5 to 9 come from the library's own sampler with 2 groups instead of 4, edge chance in a group
0.5 to 0.9, and chance 0.2 (odd chunks) or 0.1 (even chunks) instead of 0.01 for an edge between the groups, so components
are entered by several edges; seeds the standard pool's formula + 500.

Deeper test graphs.
- deep128, deep256 (the deep pool's four tasks): 32 and 16 graphs drawn as above with depth ceil(n/2)..n-1, unpadded;
  seed 840,000,000 + 1,000,000 code + 999,000 + n.
- lib128 (the same tasks): 32 graphs of 128 nodes from the library's sampler with its default arguments; seed
  840,000,000 + 1,000,000 code + 999,228.
- chain64 (the nine tasks of LONG): 32 graphs of 64 nodes, seed 20,261,115 + the task's place in LONG. Two chains of L
  (36..44) and 64 - L nodes, directed along the chain (both ways for undirected tasks); strong components and Kruskal: rings
  of the sizes in RINGS, one row per graph in turn; labels shuffled; self-loops with chance sqrt(2.2 / 63) (undirected) or
  1.6 / 63 (directed), none for acyclic tasks; weights as the library draws them; a uniform source where the task has one.
- dfs96, dfs128, dfs256 (dfs): 8, 8 and 4 graphs, every edge with chance 0.5, every node with a self-loop, positions k/n,
  numpy default_rng(1000 n + 50); the answer is the textbook's depth-first search (as the library's: roots tried from node
  0 upward, neighbours in index order).

Depth labels (<split>_depth.npz, [graphs, nodes] int16, -1 on padding):
- bellman_ford, mst_prim, dag_shortest_paths: depth = hops from the source along the answer pointers; reach = 1 for the
  nodes whose pointer chain ends at the source (an unreached node points to itself, depth 0).
- dfs: hops to the root along the answer pointers (depth in the search forest).
- topological_sort: depth = longest path from a source; height = longest path to a sink; reach_in = the largest
  shortest-path distance from a node that reaches it.
- strongly_connected_components: depth = the node's eccentricity in its component (max over u of max(d(v, u), d(u, v)));
  comp_size = the component's size.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import generate as gen

SCC = 'strongly_connected_components'
DEEP_TASKS = ('topological_sort', SCC, 'bellman_ford', 'mst_prim')
CODE = {'topological_sort': 1, SCC: 2, 'bellman_ford': 3, 'mst_prim': 4}
SEED_BASE = 840_000_000
BIG_SIZES = (24, 28, 32)
TEST_NUM = {'deep128': 32, 'deep256': 16}
P_TRAIN = gen.P_TRAIN

# chain64: (kind, weighted, has a source) per task, in the order that sets the seeds
LONG = {'bfs': ('und', False, True), 'bellman_ford': ('und', True, True), 'dijkstra': ('und', True, True),
        'mst_prim': ('und', True, True), 'dag_shortest_paths': ('dag', True, True), 'topological_sort': ('dag', False, False),
        'dfs': ('dir', False, False), SCC: ('dir', False, False), 'mst_kruskal': ('und', True, False)}
LONG_N, LONG_NUM, LONG_SEED = 64, 32, 20261115
LONG_SELF = {'und': float(np.sqrt(2.2 / 63)), 'dir': 1.6 / 63}
RINGS = [(60, 4), (56, 8), (48, 16), (44, 20), (40, 24), (36, 28), (32, 32), (32, 24, 8), (32, 16, 16), (28, 20, 16),
         (24, 24, 16), (24, 16, 12, 8, 4), (16, 16, 16, 16), (48, 8, 8), (40, 12, 12), (20, 20, 12, 12)]
DFS_TESTS = {'dfs96': (8, 96), 'dfs128': (8, 128), 'dfs256': (4, 256)}


def seed_pool(task, L, chunk):
    assert 0 <= chunk < 1000 and L < 999
    return SEED_BASE + 1_000_000 * CODE[task] + 1000 * L + chunk


def seed_test(task, tag):
    assert 0 <= tag < 1000
    return SEED_BASE + 1_000_000 * CODE[task] + 999_000 + tag


# ------------------------------------------------------------------------------------------------------- deep draws
def target_depth(rng, n, mode):
    lo = 1 if mode == 'pool' else (n + 1) // 2
    return int(rng.randint(lo, n))


def spine_tree(rng, n, D, family):
    """Parents [n] (the root points to itself) and depths: the spine 0..D, then each further node attaches to a spine node
    0..D-1 ('caterpillar') or to any earlier node of depth below D ('branch')."""
    par, dep = np.arange(n), np.zeros(n, np.int64)
    for i in range(1, D + 1):
        par[i], dep[i] = i - 1, i
    for v in range(D + 1, n):
        if family == 'caterpillar':
            u = int(rng.randint(D))
        else:
            cand = np.nonzero(dep[:v] < D)[0]
            u = int(cand[rng.randint(len(cand))])
        par[v], dep[v] = u, dep[u] + 1
    return par, dep


def relabel(M, lab):
    out = np.zeros_like(M)
    out[np.ix_(lab, lab)] = M
    return out


def draw_undirected(rng, n, mode):
    D = target_depth(rng, n, mode)
    family = ('caterpillar', 'branch', 'shortcut')[rng.randint(3)]
    par, _ = spine_tree(rng, n, D, 'caterpillar' if family == 'caterpillar' else 'branch')
    M = np.zeros((n, n))
    for v in range(1, n):
        M[v, par[v]] = M[par[v], v] = 1.
    if family == 'shortcut':
        for _ in range(rng.randint(1, 4)):
            u, v = rng.choice(n, 2, replace=False)
            M[u, v] = M[v, u] = 1.
    M[np.arange(n), np.arange(n)] = rng.binomial(1, P_TRAIN[rng.randint(len(P_TRAIN))], n)
    w = rng.uniform(0., 1., size=(n, n))
    w = np.sqrt(w * w.T + 1e-3)
    lab = rng.permutation(n)
    return [relabel(M * w, lab), int(lab[0])]


def draw_layered(rng, n, mode):
    D = target_depth(rng, n, mode)
    layer = np.concatenate([np.arange(D + 1), rng.randint(0, D + 1, size=n - D - 1)]).astype(np.int64)
    M = np.zeros((n, n))
    for v in range(1, D + 1):
        M[v - 1, v] = 1.
    for v in range(D + 1, n):
        if layer[v] > 0:
            cand = np.nonzero(layer == layer[v] - 1)[0]
            M[int(cand[rng.randint(len(cand))]), v] = 1.
    q = (0., .1, .3, .5)[rng.randint(4)]
    adjacent = (layer[None, :] == layer[:, None] + 1)
    M = np.maximum(M, (adjacent & (rng.random_sample((n, n)) < q)).astype(float))
    far = np.argwhere(layer[None, :] >= layer[:, None] + 2)
    for _ in range(int(rng.randint(1, 4)) if len(far) else 0):
        u, v = far[rng.randint(len(far))]
        M[u, v] = 1.
    return [relabel(M, rng.permutation(n))]


def draw_rings(rng, n, mode):
    D = target_depth(rng, n, mode)
    sizes, rest = [D + 1], n - D - 1
    while rest > 0:
        k = int(rng.randint(1, rest + 1))
        sizes.append(k)
        rest -= k
    order = rng.permutation(len(sizes))
    sizes = [sizes[i] for i in order]
    M, comps, at = np.zeros((n, n)), [], 0
    for k in sizes:
        c = np.arange(at, at + k)
        at += k
        comps.append(c)
        if k >= 2:
            for i in range(k):
                M[c[i], c[(i + 1) % k]] = 1.
        if k >= 4:
            for _ in range(rng.randint(3)):
                u, v = rng.choice(c, 2, replace=False)
                M[u, v] = 1.
    pd = (.1, .3, .6)[rng.randint(3)]
    for a in range(len(comps)):
        for b in range(a + 1, len(comps)):
            if rng.random_sample() < pd:
                for _ in range(1 + rng.randint(2)):
                    M[comps[a][rng.randint(len(comps[a]))], comps[b][rng.randint(len(comps[b]))]] = 1.
    diag = np.arange(n)
    M[diag, diag] = np.maximum(M[diag, diag], rng.binomial(1, P_TRAIN[rng.randint(len(P_TRAIN))], n))
    return [relabel(M, rng.permutation(n))]


DRAW = {'bellman_ford': draw_undirected, 'mst_prim': draw_undirected, 'topological_sort': draw_layered, SCC: draw_rings}


def long_graph(rs, kind, weighted, ring, i):
    N = LONG_N
    lab, A = rs.permutation(N), np.zeros((N, N))
    parts = RINGS[i % len(RINGS)] if ring else (lambda L: (L, N - L))(int(rs.randint(36, 45)))
    at = 0
    for L in parts:
        c = lab[at:at + L]
        at += L
        for u, v in list(zip(c[:-1], c[1:])) + ([(c[-1], c[0])] if ring else []):
            A[u, v] = 1.
            if kind == 'und':
                A[v, u] = 1.
    if kind in LONG_SELF:
        A[np.arange(N), np.arange(N)] = rs.binomial(1, LONG_SELF[kind], N)
    if weighted:
        w = rs.uniform(0., 1., size=(N, N))
        if kind == 'und':
            w = np.sqrt(w * w.T + 1e-3)
        A = A * w
    return A


def library_draws(task, num, L, seed, draw, M):
    """num inputs of L nodes from `draw(rng, length, i)` (input i), answers by the library's sampler for `task`, packed to M nodes."""
    import clrs
    sampler, _ = clrs.build_sampler(task, num_samples=-1, length=L, seed=seed, track_max_steps=False)
    rs, count = sampler._rng, [0]

    def sample_data(length, **kw):
        out = draw(rs, length, count[0])
        count[0] += 1
        return out
    sampler._sample_data = sample_data
    return gen.pack(sampler.next(num), M)


def deep_draws(task, num, L, seed, mode, M):
    return library_draws(task, num, L, seed, lambda rs, length, i: DRAW[task](rs, length, mode), M)


def chunks(total):
    return [(ci, min(gen.CHUNK, total - ci * gen.CHUNK)) for ci in range((total + gen.CHUNK - 1) // gen.CHUNK)]


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ------------------------------------------------------------------------------------------------------- the files
def deep_pool(task, what, per_size=80000):
    assert task in DEEP_TASKS, task
    big, L = what.startswith('big:'), int(what.split(':')[1])
    assert L in (BIG_SIZES if big else gen.SIZES), what
    M, half = (32 if big else 16), per_size // 2
    parts, seeds = [], dict(standard=[], deep=[])
    if not big:
        d, m = gen.make(task, f'train:{L}', half)
        parts.append(d)
        seeds['standard'] = m['seeds']
    for ci, num in chunks(half):
        sd = seed_pool(task, L, ci)
        parts.append(deep_draws(task, num, L, sd, 'pool', M))
        seeds['deep'].append(sd)
    d = cat(parts)
    return d, dict(length=L, padded=M, num=len(d['n']), standard=0 if big else half, deep=half, seeds=seeds)


def multi_entry_pool(what, per_size=80000):
    L = int(what.split(':')[1])
    assert what.startswith('train:') and L in gen.SIZES, what
    n_chunks = (per_size + gen.CHUNK - 1) // gen.CHUNK
    parts, seeds, kws = [], [], []
    for ci in range(n_chunks):
        seed, kw = 10 ** 7 * (gen.TASKS.index(SCC) + 1) + 1000 * L + ci, dict(p=P_TRAIN)
        if ci >= n_chunks // 2:
            seed, kw = seed + 500, dict(p=P_TRAIN[4:], k=2, eps=(0.1, 0.2)[ci % 2])
        parts.append(gen.pack(gen.sample(SCC, min(gen.CHUNK, per_size - ci * gen.CHUNK), L, seed, **kw), 16))
        seeds.append(seed)
        kws.append({a: (list(map(float, b)) if isinstance(b, tuple) else b) for a, b in kw.items()})
    return cat(parts), dict(length=L, num=per_size, seeds=seeds, chunk_kwargs=kws)


def depth_first(adj, n):
    """The textbook's depth-first search (roots from node 0 upward, neighbours in index order): parents and depths."""
    seen, par, dep = np.zeros(n, bool), np.arange(n), np.zeros(n, int)
    for s in range(n):
        if seen[s]:
            continue
        seen[s], stack = True, [[s, 0]]
        while stack:
            top = stack[-1]
            u = top[0]
            v = top[1]
            while v < n and not (adj[u, v] and not seen[v]):
                v += 1
            if v == n:
                stack.pop()
                continue
            top[1] = v + 1
            seen[v], par[v], dep[v] = True, u, dep[u] + 1
            stack.append([v, 0])
    return par, dep


def test_graphs(task, what):
    if what in TEST_NUM:
        assert task in DEEP_TASKS, task
        L = int(what[4:])
        sd = seed_test(task, L)
        return deep_draws(task, TEST_NUM[what], L, sd, 'test', L), dict(length=L, num=TEST_NUM[what], seed=sd)
    if what == 'lib128':
        assert task in DEEP_TASKS, task
        sd = seed_test(task, 228)
        return gen.pack(gen.sample(task, 32, 128, sd), 128), dict(length=128, num=32, seed=sd)
    if what == 'chain64':
        kind, weighted, source = LONG[task]
        ring, sd = task in (SCC, 'mst_kruskal'), LONG_SEED + list(LONG).index(task)

        def draw(rs, length, i):
            g = long_graph(rs, kind, weighted, ring, i)
            return [g, int(rs.choice(LONG_N))] if source else [g]
        return library_draws(task, LONG_NUM, LONG_N, sd, draw, LONG_N), dict(length=LONG_N, num=LONG_NUM, seed=sd)
    if what in DFS_TESTS:
        assert task == 'dfs', task
        G, n = DFS_TESTS[what]
        rng = np.random.default_rng(1000 * n + 50)
        a = (rng.random((G, n, n)) < .5).astype(np.float32)
        a[:, np.arange(n), np.arange(n)] = 1.
        pi = np.stack([depth_first(a[g] > 0, n)[0] for g in range(G)]).astype(np.int16)
        d = dict(n=np.full(G, n, np.int16), pos=np.tile(np.arange(n) / n, (G, 1)).astype(np.float32), A=a,
                 adj=a.astype(np.uint8), pi=pi)
        return d, dict(length=n, num=G, seed=1000 * n + 50)
    raise ValueError(what)


# ------------------------------------------------------------------------------------------------------- depth labels
def tree_hops(par):
    """Hops from each node to the root of its pointer chain (par[root] = root), and that root."""
    n = len(par)
    dep, cur, idx = np.zeros(n, np.int64), np.asarray(par, np.int64).copy(), np.arange(n)
    for _ in range(n):
        moved = cur[idx] != idx
        if not moved.any():
            break
        dep += moved
        idx = np.where(moved, cur[idx], idx)
    return dep, idx


def hops(pi, n):
    dep, root = np.zeros(n, int), np.zeros(n, int)
    for v in range(n):
        u, k = v, 0
        while pi[u] != u and k <= n:
            u, k = pi[u], k + 1
        dep[v], root[v] = k, u
    return dep, root


def hop_dist(E):
    """All-pairs hop distances of a directed 0/1 adjacency E [n, n] without the diagonal (inf where unreachable)."""
    n = len(E)
    D = np.full((n, n), np.inf)
    D[np.arange(n), np.arange(n)] = 0
    R = np.eye(n, dtype=bool)
    front, Ef = R.copy(), E.astype(np.float32)
    for k in range(1, n):
        nxt = ((front.astype(np.float32) @ Ef) > 0) & ~R
        if not nxt.any():
            break
        D[nxt] = k
        R |= nxt
        front = nxt
    return D


def longest_into(E):
    """Longest path (hops) from a source to each node of a DAG E [n, n] (0/1, no diagonal)."""
    n = len(E)
    indeg, best = E.sum(0).astype(np.int64), np.zeros(n, np.int64)
    stack, seen = list(np.nonzero(indeg == 0)[0]), 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in np.nonzero(E[u])[0]:
            best[v] = max(best[v], best[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    assert seen == n, 'not a DAG'
    return best


def depth_labels(task, d):
    """Per-node depth labels of a split d (module docstring): {name: [G, M] int16}, -1 on padding."""
    if task in ('dfs', 'dag_shortest_paths'):
        G, M = d['pi'].shape
        out = dict(depth=np.full((G, M), -1, np.int16))
        if task == 'dag_shortest_paths':
            out['reach'] = np.full((G, M), -1, np.int16)
        for g in range(G):
            n = int(d['n'][g])
            dep, root = hops(d['pi'][g, :n].astype(int), n)
            if task == 'dag_shortest_paths':
                reach = root == int(d['s'][g])
                out['depth'][g, :n], out['reach'][g, :n] = np.where(reach, dep, 0), reach
            else:
                out['depth'][g, :n] = dep
        return out
    G, M = d['A'].shape[:2]
    out = {}

    def put(name, g, vals):
        out.setdefault(name, np.full((G, M), -1, np.int16))[g, :len(vals)] = vals
    for g in range(G):
        n = int(d['n'][g])
        E = (d['A'][g, :n, :n] != 0) & ~np.eye(n, dtype=bool)
        if task in ('bellman_ford', 'mst_prim'):
            dep, root = tree_hops(d['pi'][g, :n])
            reach = root == int(d['s'][g])
            put('depth', g, np.where(reach, dep, 0))
            put('reach', g, reach.astype(np.int16))
        elif task == 'topological_sort':
            Dm = hop_dist(E)
            put('depth', g, longest_into(E))
            put('height', g, longest_into(E.T))
            put('reach_in', g, np.where(np.isfinite(Dm), Dm, 0).max(0).astype(np.int64))
        elif task == SCC:
            Dm = hop_dist(E)
            same = np.isfinite(Dm) & np.isfinite(Dm.T)
            put('depth', g, np.where(same, np.maximum(Dm, Dm.T), 0).max(1).astype(np.int64))
            put('comp_size', g, same.sum(1))
        else:
            raise ValueError(task)
    return out


def make(pool, task, what, per_size=80000):
    """-> (arrays, manifest) of one file: pool 'deep', 'multi_entry' or 'tests'."""
    if pool == 'deep':
        return deep_pool(task, what, per_size)
    if pool == 'multi_entry':
        assert task == SCC, task
        return multi_entry_pool(what, per_size)
    if pool == 'tests':
        return test_graphs(task, what)
    raise ValueError(pool)


def file_name(what):
    return what.replace('train:', 'train_L').replace('big:', 'big_L')


def save(dest, name, d, meta, t0):
    import clrs
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f'{name}.npz', **d)
    meta = dict(meta, file=f'{name}.npz', sha256=gen.digest(d), shapes={k: list(v.shape) for k, v in d.items()},
                seconds=round(time.time() - t0, 1), clrs_version=getattr(clrs, '__version__', '?'), numpy=np.__version__)
    (dest / f'{name}.json').write_text(json.dumps(meta))
    return meta


def main(cmd, *args):
    t0 = time.time()
    if cmd == 'copy':
        src, dst = Path(args[0]) / args[2], Path(args[1]) / args[2]
        dst.mkdir(parents=True, exist_ok=True)
        done = []
        for f in sorted(src.iterdir()):
            if f.suffix in ('.npz', '.json') and not f.name.startswith('train_'):
                shutil.copyfile(f, dst / f.name)
                done.append(f.name)
        return dict(task=args[2], copied=done)
    if cmd == 'depth':
        dest, task, split = Path(args[0]) / args[1], args[1], args[2]
        with np.load(dest / f'{split}.npz') as z:
            d = {k: z[k] for k in z.files}
        return save(dest, f'{split}_depth', depth_labels(task, d), dict(task=task, split=split, of_sha256=gen.digest(d)), t0)
    out, task, what = Path(args[0]), args[1], args[2]
    d, meta = make(cmd, task, what, *(int(a) for a in args[3:4]))
    return save(out / task, file_name(what), d, dict(meta, task=task, split=what), t0)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    print(json.dumps(main(*sys.argv[1:])))
