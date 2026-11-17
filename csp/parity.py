"""Parity equations with a hidden twist (Tseitin type). Base: a connected simple graph G in which
every vertex has three edges. Two variables per edge e, x(e, 0) and x(e, 1), with the equation x(e, 0) + x(e, 1) = 1
("they differ"); the 0 / 1 label is hidden (the items are shuffled; nothing marks it). For each vertex v with edges e1, e2,
e3, eight equations, one per choice (a, b, c) in {0, 1}^3: x(e1, a) + x(e2, b) + x(e3, c) = a + b + c + t_v (mod 2), t = 0
everywhere in the plain instance, t = 1 at one chosen vertex in the twisted one. With y_e = x(e, 0), every equation at v says
sum_{e at v} y_e = t_v, so the system is solvable exactly when sum_v t_v is even: plain solvable, twisted not. Both have
four of eight right-hand sides 1 at every vertex and 1 on every edge equation, so no count separates them.
Items (before the shuffle): the 2|E| variables, the |E| edge equations, the 8|V| vertex equations, one global item:
3|E| + 8|V| + 1 = 12.5 |V| + 1 items (cube 101, Petersen 126, 16 vertices 201). Pair input A[var, eq] = 1 when the
variable occurs in the equation (directed: variable to equation, so the embedding sees which end is which); the equation
item carries its right-hand side on the diagonal of adj (adj[eq, eq] = rhs; adj is 0 elsewhere); the global item has
neither (the trainer marks it, s1). Answer: 1 when unsolvable, read at the global item.
reference: Gaussian elimination over GF(2) on the equations (the standard routine). brute: every assignment (12
variables: K4 only). decode: the system read back from A and adj, then the same elimination."""
import itertools
import numpy as np
from .common import finish

SPEC = dict(name='parity', code=33, outs=[['unsat', 'nmask']], items=lambda nvert: int(12.5 * nvert) + 1)


def cube():
    return [(u, u ^ (1 << b)) for u in range(8) for b in range(3) if u < u ^ (1 << b)]


def petersen():
    return [(i, (i + 1) % 5) for i in range(5)] + [(5 + i, 5 + (i + 2) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]


def k4():
    return [(a, b) for a in range(4) for b in range(a + 1, 4)]


def connected(n, E):
    adj = [[] for _ in range(n)]
    for u, v in E: adj[u].append(v); adj[v].append(u)
    seen, st = {0}, [0]
    while st:
        for w in adj[st.pop()]:
            if w not in seen: seen.add(w); st.append(w)
    return len(seen) == n


def random_cubic(rng, n, tries=100000):
    """A uniformly random labelled connected simple cubic graph on n (even) vertices: the pairing model with rejection."""
    assert n % 2 == 0 and n >= 4
    for _ in range(tries):
        pts = rng.permutation(3 * n) // 3
        E = [tuple(sorted((int(pts[2 * i]), int(pts[2 * i + 1])))) for i in range(3 * n // 2)]
        if any(u == v for u, v in E) or len(set(E)) < len(E) or not connected(n, E): continue
        return E
    raise RuntimeError('no cubic graph')


def system(nvert, E, twist):
    """(equations [(vars tuple, rhs)], number of variables): variable 2e + a is x(e, a); twist: the vertex with t = 1, or None."""
    inc = [[] for _ in range(nvert)]
    for e, (u, v) in enumerate(E): inc[u].append(e); inc[v].append(e)
    assert all(len(q) == 3 for q in inc), 'every vertex has three edges'
    eqs = [((2 * e, 2 * e + 1), 1) for e in range(len(E))]
    for v in range(nvert):
        e1, e2, e3 = inc[v]
        for a, b, c in itertools.product((0, 1), repeat=3):
            eqs.append(((2 * e1 + a, 2 * e2 + b, 2 * e3 + c), (a + b + c + (1 if v == twist else 0)) % 2))
    return eqs, 2 * len(E)


def solvable(eqs, nvar):
    """Gaussian elimination over GF(2): rows as Python ints (bit i = variable i, bit nvar = rhs)."""
    piv = {}
    for vs, r in eqs:
        row = sum(1 << v for v in vs) | (r << nvar)
        while True:
            low = row & ((1 << nvar) - 1)
            if not low:
                if row: return False  # 0 = 1
                break
            h = low.bit_length() - 1
            if h in piv: row ^= piv[h]
            else: piv[h] = row; break
    return True


def reference(x):
    return dict(unsat=int(not solvable(x['cons'], x['meta']['nv'])))


def brute(x):
    nvar, eqs = x['meta']['nv'], x['cons']
    assert nvar <= 12
    bits = ((np.arange(2 ** nvar)[:, None] >> np.arange(nvar)[None]) & 1)
    ok = np.ones(2 ** nvar, bool)
    for vs, r in eqs: ok &= (bits[:, list(vs)].sum(1) % 2) == r
    return dict(unsat=int(not ok.any()))


def decode(x):
    """The system from the arrays: equations = items with an incoming occurrence, their variables = the items pointing at
    them, rhs = adj's diagonal; then the same elimination. Also checks the encoding's shape."""
    n, s = x['n'], x['s']; A, adj = x['A'].astype(bool), x['adj'].astype(np.int64)
    assert not A[s].any() and not A[:, s].any() and adj[s, s] == 0
    assert (adj - np.diag(np.diag(adj)) == 0).all(), 'adj carries only the diagonal mark'
    var = [i for i in range(n) if A[i].any()]; eq = [i for i in range(n) if A[:, i].any()]
    assert not set(var) & set(eq) and len(var) + len(eq) == n - 1
    vid = {v: i for i, v in enumerate(var)}
    eqs = [(tuple(vid[int(v)] for v in np.nonzero(A[:, e])[0]), int(adj[e, e])) for e in eq]
    return dict(unsat=int(not solvable(eqs, len(var))))


def encode(nvert, E, twist, rng, meta):
    eqs, nvar = system(nvert, E, twist)
    n = nvar + len(eqs) + 1; g = n - 1
    A = np.zeros((n, n), np.uint8); adj = np.zeros((n, n), np.uint8)
    for i, (vs, r) in enumerate(eqs):
        A[list(vs), nvar + i] = 1; adj[nvar + i, nvar + i] = r
    label = int(twist is not None)
    return finish(n, A, adj, g, label, rng, dict(meta, nv=nvar, nvert=nvert, twist=-1 if twist is None else int(twist),
                                                 rhs_ones=int(sum(r for _, r in eqs)), edges=[list(e) for e in E]), eqs)


def sample(num, sizes, seed, graph=None):
    """num instances, twin pairs (plain, twisted on the same G): G drawn per pair, its size uniform in sizes (random cubic), or
    the fixed graph 'cube', 'petersen', 'k4'. Both members get their own item shuffle; the twisted vertex is uniform."""
    rng = np.random.RandomState(seed); rows = []
    assert num % 2 == 0
    for p_ in range(num // 2):
        if graph:
            E = dict(cube=cube, petersen=petersen, k4=k4)[graph](); nvert = 1 + max(max(e) for e in E)
        else:
            nvert = int(sizes[rng.randint(len(sizes))]); E = random_cubic(rng, nvert)
        tw = int(rng.randint(nvert))
        for twist in (None, tw):
            rows.append(encode(nvert, E, twist, rng, dict(kind=1, pair=p_, depth=-1)))
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def _h(E, twist):
    return encode(1 + max(max(e) for e in E), E, twist, np.random.RandomState(0), dict(kind=1, pair=0, depth=-1))


HAND = [('K4 plain', _h(k4(), None), dict(unsat=0)), ('K4 twisted at 0', _h(k4(), 0), dict(unsat=1)),
        ('cube plain', _h(cube(), None), dict(unsat=0)), ('cube twisted at 5', _h(cube(), 5), dict(unsat=1)),
        ('Petersen plain', _h(petersen(), None), dict(unsat=0)), ('Petersen twisted at 7', _h(petersen(), 7), dict(unsat=1))]
