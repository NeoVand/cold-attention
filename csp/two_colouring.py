"""Two-colouring as constraints "same" and "different" on pairs (binary parity). Items: the n variables
and one global item. Pair inputs, both symmetric: A[u, v] = 1 for "same", adj[u, v] = 1 for "different"; at most one
constraint per pair; the global item has none (the trainer marks it, s1). Unsatisfiable exactly when some cycle carries an
odd number of "different". Answer: 1 when unsatisfiable, read at the global item.

reference: the standard routine, union-find with parities. brute: every assignment (n <= 12). decode: the constraints
read back from A and adj, then a BFS two-colouring (independent of the union-find).

Instances (sample):
- random: a random forest (each variable after the first joins a uniformly random earlier one, or starts a new tree with
  chance 1/8; labels shuffled) plus r extra constraints on random new pairs, r uniform in 1..3 (cycle rank r: a dense
  random graph at large n is unsatisfiable almost surely, so a fixed edge ratio cannot be balanced), random relations;
  balanced to half unsatisfiable by rejection within each size.
- twins (kind 1): one long cycle v0 .. v(c-1) among trees (every other variable joins a random earlier variable, random
  relation). The unsatisfiable member closes the cycle with (v(c-1), v0) so that it carries an odd number of "different",
  with (v0, v1) "different". The satisfiable member moves that closing constraint, relation unchanged, to (v(c-1), v1):
  its cycle v1 .. v(c-1) has length c - 1 and an even count, and v0 hangs from v1. One constraint differs; the counts of
  "same" and "different" and of the item marks are equal. (Flipping one relation instead would change those counts by
  one.)
- depth of an unsatisfiable instance: ceil(c* / 2), c* its shortest cycle with an odd count (BFS on (variable, parity));
  twin pairs carry the unsatisfiable member's; also stored: cycle (c* for random, c for twins)."""
from collections import deque
import numpy as np
from .common import finish, K_rule

SPEC = dict(name='two_colouring', code=32, outs=[['unsat', 'nmask']], items=lambda n: n + 1)


def unsat_of(nv, cons):
    par, rel = list(range(nv)), [0] * nv  # rel[v]: parity of v against par[v]

    def find(v):
        path = []
        while par[v] != v: path.append(v); v = par[v]
        root, acc = v, 0
        for u in reversed(path): acc ^= rel[u]; rel[u] = acc; par[u] = root
        return root
    for u, v, d in cons:
        ru, rv = find(u), find(v); pu, pv = rel[u] if u != ru else 0, rel[v] if v != rv else 0
        if ru == rv:
            if pu ^ pv != d: return 1
        else:
            par[ru] = rv; rel[ru] = pu ^ pv ^ d
    return 0


def reference(x):
    return dict(unsat=unsat_of(x['meta']['nv'], x['cons']))


def brute(x):
    nv, cons = x['meta']['nv'], x['cons']; assert nv <= 12
    bits = (np.arange(2 ** nv)[:, None] >> np.arange(nv)[None]) & 1; ok = np.ones(2 ** nv, bool)
    for u, v, d in cons: ok &= (bits[:, u] ^ bits[:, v]) == d
    return dict(unsat=int(not ok.any()))


def odd_cycle(nv, cons):
    """Shortest closed walk with an odd count of "different" (= shortest such cycle), or -1."""
    nb = [[] for _ in range(nv)]
    for u, v, d in cons: nb[u].append((v, d)); nb[v].append((u, d))
    best = -1
    for s in range(nv):
        dist = {(s, 0): 0}; q = deque([(s, 0)])
        while q:
            v, p = q.popleft()
            for w, d in nb[v]:
                if (w, p ^ d) not in dist: dist[(w, p ^ d)] = dist[(v, p)] + 1; q.append((w, p ^ d))
        if (s, 1) in dist: best = dist[(s, 1)] if best < 0 else min(best, dist[(s, 1)])
    return best


def decode(x):
    n, s = x['n'], x['s']; A, D = x['A'].astype(bool), x['adj'].astype(bool)
    assert (A == A.T).all() and (D == D.T).all() and not (A & D).any() and not A.diagonal().any() and not D.diagonal().any()
    assert not A[s].any() and not D[s].any()
    items = [i for i in range(n) if i != s]; col = {}
    for r in items:
        if r in col: continue
        col[r] = 0; q = deque([r])
        while q:
            v = q.popleft()
            for w in range(n):
                if A[v, w] or D[v, w]:
                    c = col[v] ^ int(D[v, w])
                    if w not in col: col[w] = c; q.append(w)
                    elif col[w] != c: return dict(unsat=1)
    return dict(unsat=0)


def encode(nv, cons, label, rng, meta):
    n = nv + 1; A = np.zeros((n, n), np.uint8); adj = np.zeros((n, n), np.uint8)
    for u, v, d in cons:
        assert u != v and not A[u, v] and not adj[u, v], 'one constraint per pair'
        (adj if d else A)[u, v] = 1; (adj if d else A)[v, u] = 1
    return finish(n, A, adj, nv, label, rng, dict(meta, nv=nv, ncons=len(cons), same=int(sum(1 - c[2] for c in cons)),
                                                 different=int(sum(c[2] for c in cons))), [tuple(map(int, c)) for c in cons])


def random_instance(rng, nv, r):
    lab = rng.permutation(nv); seen, cons = set(), []
    for i in range(1, nv):
        if rng.rand() < 1 / 8: continue
        u, v = sorted((int(lab[i]), int(lab[rng.randint(i)])))
        seen.add((u, v)); cons.append((u, v, int(rng.randint(2))))
    r = min(r, nv * (nv - 1) // 2 - len(cons))
    while r > 0:
        u, v = sorted(int(q) for q in rng.choice(nv, 2, replace=False))
        if (u, v) not in seen: seen.add((u, v)); cons.append((u, v, int(rng.randint(2)))); r -= 1
    return cons


def c_min(nv):
    """Smallest unsatisfiable cycle c with both members' cycles (c and c - 1) at least twice the loop budget K(items)."""
    return 2 * K_rule(nv + 1) + 1


def twins_possible(nv):
    return c_min(nv) <= nv - 2  # at least two tree variables


def twin_pair(rng, nv, c):
    var = [int(v) for v in rng.permutation(nv)]; cyc, rest = var[:c], var[c:]
    rel = [int(rng.randint(2)) for _ in range(c - 1)]; rel[0] = 1  # (v0, v1) different
    close = (sum(rel) + 1) % 2  # odd total on the unsatisfiable member's cycle
    base = [(cyc[i], cyc[i + 1], rel[i]) for i in range(c - 1)]
    placed = list(cyc)
    for v in rest:
        u = placed[rng.randint(len(placed))]; base.append((u, v, int(rng.randint(2)))); placed.append(v)
    cu = base + [(cyc[-1], cyc[0], close)]; cs = base + [(cyc[-1], cyc[1], close)]
    assert unsat_of(nv, cu) == 1 and unsat_of(nv, cs) == 0
    return [tuple(sorted(q[:2])) + (q[2],) for q in cu], [tuple(sorted(q[:2])) + (q[2],) for q in cs], dict(cycle=c, depth=(c + 1) // 2)


def sample(num, nv, seed, twin_share=.5, deep=False):
    assert nv >= 3, 'two variables with one constraint per pair are always satisfiable: no balance possible'
    rng = np.random.RandomState(seed); rows = []
    npairs = int(round(twin_share * num / 2)) if twins_possible(nv) else 0
    lo = max(c_min(nv), int(np.ceil(nv / 2))) if deep else c_min(nv)
    for p_ in range(npairs):
        cu, cs, info = twin_pair(rng, nv, int(rng.randint(lo, nv - 1)))
        for lab, cl in ((1, cu), (0, cs)): rows.append(encode(nv, cl, lab, rng, dict(info, kind=1, pair=p_)))
    nr = num - 2 * npairs; want = {0: nr - nr // 2, 1: nr // 2}
    while want[0] or want[1]:
        cons = random_instance(rng, nv, int(rng.randint(1, 4))); lab = unsat_of(nv, cons)
        if want[lab]:
            want[lab] -= 1; oc = odd_cycle(nv, cons) if lab else -1
            rows.append(encode(nv, cons, lab, rng, dict(kind=0, pair=-1, cycle=oc, depth=(oc + 1) // 2 if lab else -1)))
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def _hand(nv, cons):
    return encode(nv, cons, unsat_of(nv, cons), np.random.RandomState(0), dict(kind=0, pair=-1, depth=-1))


HAND = [('one same', _hand(2, [(0, 1, 0)]), dict(unsat=0)), ('triangle, one different', _hand(3, [(0, 1, 1), (1, 2, 0), (0, 2, 0)]), dict(unsat=1)),
        ('triangle, two different', _hand(3, [(0, 1, 1), (1, 2, 1), (0, 2, 0)]), dict(unsat=0)),
        ('square, all different', _hand(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 1)]), dict(unsat=0)),
        ('pentagon, all different', _hand(5, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (0, 4, 1)]), dict(unsat=1)),
        ('a tree', _hand(5, [(0, 1, 1), (0, 2, 0), (2, 3, 1), (2, 4, 1)]), dict(unsat=0))]
