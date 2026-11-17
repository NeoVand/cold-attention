"""2-SAT. Items: the 2n literals (literal 2v + 0 is variable v, 2v + 1 its negation, before the
shuffle) and one global item. Pair inputs: A[a, b] = 1 when "a implies b" (two per clause: (p or q) gives not-p -> q and
not-q -> p); adj[a, b] = adj[b, a] = 1 when b is the negation of a. The global item has neither; the trainer marks it (s1).
Clauses always join two different variables (no unit clauses, no tautologies) and never repeat, so each clause adds exactly
two implications and the counts of the twins are equal. Answer: 1 when unsatisfiable, read at the global item.

reference: the standard routine (Aspvall-Plass-Tarjan): unsatisfiable exactly when a literal and its negation share a
strongly connected component of the implication graph (Kosaraju, iterative). brute: every assignment (n <= 12).
decode: the clauses' implication graph read back from A and adj (checks that the encoding is the instance).

Instances (sample):
- random: m clauses, m uniform in n .. max(n + 2, ceil(1.25 n)) (clause-to-variable ratio near 1), distinct random clauses;
  balanced to half unsatisfiable by rejection within each size.
- twins (kind 1): a planted contradiction on x. Forward chain x -> a1 -> ... -> ak -> b -> not-ak -> ... -> not-a1 ->
  not-x (clauses (not-x or a1), (not-ai or ai+1), (not-ak or b), (not-b or not-ak); 2k + 2 implications, no shortcut) and
  back chain not-x -> c1 -> ... -> cj -> d -> not-cj -> ... -> not-c1 -> x (2j + 2), on n_chain = k + j + 3 variables, k
  and j as equal as possible; signs of every variable random. Its closed chain x -> not-x -> x has 2 n_chain - 2
  implications. Filler clauses on the other variables (ratio 0.5-1) plus up to 3 clauses joining a chain literal and a
  filler literal, accepted only if the planted distances, the shallowest certificate (depth, below) and both answers
  are unchanged. The satisfiable twin flips the sign of one literal of one chain clause (uniform among the flips that
  make the instance satisfiable and repeat no clause): the two differ in one constraint and have the same counts of
  every pair input and item mark.
- depth of an unsatisfiable instance: the shallowest certificate, min over variables v with v and not-v in one component
  of max(d(v, not-v), d(not-v, v)) in implications (BFS); a twin pair carries its unsatisfiable member's depth; -1 for
  a satisfiable random instance."""
import itertools
from collections import deque
import numpy as np
from .common import finish, K_rule

SPEC = dict(name='two_sat', code=31, outs=[['unsat', 'nmask']], items=lambda n: 2 * n + 1)


# ------------------------------------------------------------------------------------------------ the standard routine
def implication_graph(nv, clauses):
    out = [[] for _ in range(2 * nv)]
    for p, q in clauses:
        out[p ^ 1].append(q); out[q ^ 1].append(p)
    return out


def scc(out):
    """Kosaraju, iterative: component id per node."""
    N = len(out); order, seen = [], [False] * N
    for s in range(N):
        if seen[s]: continue
        seen[s] = True; st = [(s, 0)]
        while st:
            v, i = st.pop()
            if i < len(out[v]):
                st.append((v, i + 1)); w = out[v][i]
                if not seen[w]: seen[w] = True; st.append((w, 0))
            else:
                order.append(v)
    rev = [[] for _ in range(N)]
    for v in range(N):
        for w in out[v]: rev[w].append(v)
    comp, c = [-1] * N, 0
    for s in reversed(order):
        if comp[s] >= 0: continue
        comp[s] = c; st = [s]
        while st:
            v = st.pop()
            for w in rev[v]:
                if comp[w] < 0: comp[w] = c; st.append(w)
        c += 1
    return comp


def unsat_of(nv, clauses):
    comp = scc(implication_graph(nv, clauses))
    return int(any(comp[2 * v] == comp[2 * v + 1] for v in range(nv)))


def reference(x):
    return dict(unsat=unsat_of(x['meta']['nv'], x['cons']))


def bfs(out, s):
    d = [-1] * len(out); d[s] = 0; q = deque([s])
    while q:
        v = q.popleft()
        for w in out[v]:
            if d[w] < 0: d[w] = d[v] + 1; q.append(w)
    return d


def depth_of(nv, clauses):
    """Shallowest certificate of an unsatisfiable instance (module docstring); -1 when satisfiable."""
    out = implication_graph(nv, clauses); comp = scc(out); best = -1
    for v in range(nv):
        if comp[2 * v] == comp[2 * v + 1]:
            dv = max(bfs(out, 2 * v)[2 * v + 1], bfs(out, 2 * v + 1)[2 * v])
            best = dv if best < 0 else min(best, dv)
    return best


def brute(x):
    """Independent: every assignment of the nv <= 12 variables against every clause."""
    nv, cl = x['meta']['nv'], np.array(x['cons'], np.int64).reshape(-1, 2)
    assert nv <= 12
    bits = ((np.arange(2 ** nv)[:, None] >> np.arange(nv)[None]) & 1).astype(bool)  # [assignments, variables]
    val = lambda lit: bits[:, lit // 2] ^ (lit % 2 == 1)  # literal 2v true when v true, 2v + 1 when v false
    ok = np.ones(2 ** nv, bool)
    for p, q in cl: ok &= val(p) | val(q)
    return dict(unsat=int(not ok.any()))


def decode(x):
    """From the instance's arrays: the literal items (global removed), the implication graph A, the negation pairing adj;
    -> unsat by the same SCC test on the encoded graph (independent of the clause list)."""
    n, s = x['n'], x['s']; lits = [i for i in range(n) if i != s]
    A, adj = x['A'].astype(bool), x['adj'].astype(bool)
    assert not A[s].any() and not A[:, s].any() and not adj[s].any() and not adj[:, s].any(), 'the global item has no pair input'
    assert (adj == adj.T).all() and (adj.sum(1)[lits] == 1).all() and not adj.diagonal().any(), 'negation is a perfect matching'
    idx = {v: i for i, v in enumerate(lits)}
    out = [[idx[w] for w in np.nonzero(A[v])[0]] for v in lits]
    comp = scc(out); neg = [idx[int(np.nonzero(adj[v])[0][0])] for v in lits]
    return dict(unsat=int(any(comp[i] == comp[neg[i]] for i in range(len(lits)))))


# ------------------------------------------------------------------------------------------------ encoding and sampling
def encode(nv, clauses, label, rng, meta):
    n = 2 * nv + 1; g = 2 * nv
    A = np.zeros((n, n), np.uint8); adj = np.zeros((n, n), np.uint8)
    for p, q in clauses:
        assert p // 2 != q // 2
        A[p ^ 1, q] = 1; A[q ^ 1, p] = 1
    assert int(A.sum()) == 2 * len(clauses), 'a repeated implication'
    for v in range(nv):
        adj[2 * v, 2 * v + 1] = adj[2 * v + 1, 2 * v] = 1
    return finish(n, A, adj, g, label, rng, dict(meta, nv=nv, nclauses=len(clauses)), [tuple(map(int, c)) for c in clauses])


def key(c):
    return tuple(sorted(c))


def random_clause(rng, nv):
    u, v = rng.choice(nv, 2, replace=False)
    return (int(2 * u + rng.randint(2)), int(2 * v + rng.randint(2)))


def random_formula(rng, nv, m):
    m = min(m, 4 * nv * (nv - 1) // 2); seen, cl = set(), []
    while len(cl) < m:
        c = random_clause(rng, nv)
        if key(c) not in seen: seen.add(key(c)); cl.append(c)
    return cl


def m_range(nv):
    return nv, max(nv + 2, int(np.ceil(1.25 * nv)))


def chain_clauses(roles, k, j):
    """roles: role literal per chain variable (x, a1..ak, b, c1..cj, d) -> the chain's clauses (module docstring)."""
    X, As, B, Cs, Dd = roles[0], roles[1:1 + k], roles[1 + k], roles[2 + k:2 + k + j], roles[2 + k + j]
    n_ = lambda l: l ^ 1
    cl = []
    if k: cl += [(n_(X), As[0])] + [(n_(As[i]), As[i + 1]) for i in range(k - 1)] + [(n_(As[-1]), B), (n_(B), n_(As[-1]))]
    else: cl += [(n_(X), B), (n_(X), n_(B))]
    if j: cl += [(X, Cs[0])] + [(n_(Cs[i]), Cs[i + 1]) for i in range(j - 1)] + [(n_(Cs[-1]), Dd), (n_(Dd), n_(Cs[-1]))]
    else: cl += [(X, Dd), (X, n_(Dd))]
    return cl


def twin_pair(rng, nv, n_chain, tries=200):
    """(unsat clauses, sat clauses, info) of one twin pair on nv variables with a chain on n_chain of them."""
    assert 3 <= n_chain <= nv
    k = (n_chain - 3 + 1) // 2; j = n_chain - 3 - k
    for _ in range(tries):
        var = rng.permutation(nv); sign = rng.randint(2, size=nv)
        lit = lambda v: int(2 * v + sign[v])
        chain = chain_clauses([lit(v) for v in var[:n_chain]], k, j)
        x = int(var[0]); X = lit(x)
        out0 = implication_graph(nv, chain)
        d_fwd, d_back, dep0 = bfs(out0, X)[X ^ 1], bfs(out0, X ^ 1)[X], depth_of(nv, chain)
        assert (d_fwd, d_back) == (2 * k + 2, 2 * j + 2), (d_fwd, d_back, k, j)
        fill = [int(v) for v in var[n_chain:]]; cl = list(chain); seen = {key(c) for c in cl}
        if len(fill) >= 2:
            for _ in range(int(round(rng.uniform(.5, 1.) * len(fill)))):
                u, v = rng.choice(fill, 2, replace=False); c = (lit(u) ^ rng.randint(2), lit(v) ^ rng.randint(2))
                if key(c) not in seen: seen.add(key(c)); cl.append(c)
        for _ in range(rng.randint(0, min(3, len(fill)) + 1) if fill else 0):
            u = int(var[rng.randint(n_chain)]); v = fill[rng.randint(len(fill))]; c = (2 * u + rng.randint(2), 2 * v + rng.randint(2))
            if key(c) not in seen: seen.add(key(c)); cl.append(c)
        out = implication_graph(nv, cl)
        if (bfs(out, X)[X ^ 1], bfs(out, X ^ 1)[X]) != (d_fwd, d_back) or depth_of(nv, cl) != dep0:
            continue
        found = None  # the first valid flip in a random order of all chain flips = uniform among the valid ones
        for f_ in rng.permutation(2 * len(chain)):
            ci, pos = int(f_) // 2, int(f_) % 2
            c2 = list(cl[ci]); c2[pos] ^= 1; c2 = tuple(c2)
            if key(c2) in seen: continue
            cl2 = cl[:ci] + [c2] + cl[ci + 1:]
            if not unsat_of(nv, cl2): found = (ci, pos, cl2); break
        if found is None: continue
        ci, pos, cl2 = found
        return cl, cl2, dict(k=k, j=j, n_chain=n_chain, closed=d_fwd + d_back, d_fwd=d_fwd, d_back=d_back, depth=dep0, flip=[ci, pos],
                             fillers=len(cl) - len(chain), chain_clauses=len(chain))
    raise RuntimeError(f'no twin pair at nv={nv}, n_chain={n_chain}')


def nc_min(nv):
    """Smallest chain whose closed chain (2 n_chain - 2 implications) is at least twice the loop budget K(items) at this size."""
    return K_rule(2 * nv + 1) + 1


def twins_possible(nv):
    return nc_min(nv) <= nv


def sample(num, nv, seed, twin_share=.5, deep=False):
    """num instances at nv variables: twins (pairs, both members) where twins_possible(nv), else none; the rest random,
    half unsatisfiable. deep (evaluation): chains of n_chain uniform in max(nc_min, ceil(nv / 2)) .. nv; else nc_min .. nv."""
    rng = np.random.RandomState(seed)
    npairs = int(round(twin_share * num / 2)) if twins_possible(nv) else 0
    rows = []
    lo = max(nc_min(nv), int(np.ceil(nv / 2))) if deep else nc_min(nv)
    for p_ in range(npairs):
        nchain = int(rng.randint(lo, nv + 1))
        cu, cs, info = twin_pair(rng, nv, nchain)
        for lab, cl in ((1, cu), (0, cs)):
            rows.append(encode(nv, cl, lab, rng, dict(info, kind=1, pair=p_)))
    nr = num - 2 * npairs; want = {0: nr - nr // 2, 1: nr // 2}; a, b = m_range(nv)
    while want[0] or want[1]:
        cl = random_formula(rng, nv, int(rng.randint(a, b + 1))); lab = unsat_of(nv, cl)
        if want[lab]:
            want[lab] -= 1
            rows.append(encode(nv, cl, lab, rng, dict(kind=0, pair=-1, depth=depth_of(nv, cl) if lab else -1)))
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


# ------------------------------------------------------------------------------------------------ hand cases
def _hand(nv, clauses):
    rng = np.random.RandomState(0)
    return encode(nv, clauses, unsat_of(nv, clauses), rng, dict(kind=0, pair=-1, depth=-1))


P = lambda v: 2 * v; N = lambda v: 2 * v + 1
HAND = [
    ('one clause', _hand(2, [(P(0), P(1))]), dict(unsat=0)),
    ('all four clauses on two variables', _hand(2, [(P(0), P(1)), (P(0), N(1)), (N(0), P(1)), (N(0), N(1))]), dict(unsat=1)),
    ('three of the four', _hand(2, [(P(0), P(1)), (P(0), N(1)), (N(0), P(1))]), dict(unsat=0)),
    ('y forced, then z and not z', _hand(3, [(P(0), P(1)), (N(0), P(1)), (N(1), P(2)), (N(1), N(2))]), dict(unsat=1)),
    ('implication cycle without a contradiction', _hand(3, [(N(0), P(1)), (N(1), P(2)), (N(2), P(0))]), dict(unsat=0)),
    ('the bare twin chain, k = j = 0', _hand(3, chain_clauses([P(0), P(1), P(2)], 0, 0)), dict(unsat=1)),
    ('the bare twin chain, k = 2, j = 1', _hand(6, chain_clauses([P(0), N(1), P(2), P(3), N(4), P(5)], 2, 1)), dict(unsat=1)),
]
