"""Model inputs and targets from the split files that generate.py writes (padded numpy arrays, no hints).

Graph split: n [S], pos [S, M], A [S, M, M] (0 = no edge), adj [S, M, M], s [S] (source, if the task has one), outputs.
Other split: n [S], x_<name> (item numbers float64 [S, M], item masks uint8 [S, M], categories one-hot [S, M, C], per-input
numbers float64 [S]), outputs. Outputs are integer indices padded with -1.
"""
import numpy as np

from .model import PREG_BIG, TWO_ROWS
from .tasks import SAME_UNITS


def load_split(path):
    return {k: v for k, v in np.load(path).items()}


def graph_inputs(d, idx, rng=None):
    """x for a graph task. rng (training): positions randomised as the library's process_random_pos does (sorted uniforms)."""
    n = d['n'][idx].astype(np.int32)
    N = d['A'].shape[-1]
    m = (np.arange(N)[None] < n[:, None]).astype(np.float32)
    pos = d['pos'][idx].astype(np.float32)
    if rng is not None:
        u = np.where(m > 0, rng.random(m.shape), np.inf)
        pos = np.where(m > 0, np.sort(u, 1), 0.).astype(np.float32)
    s1 = np.zeros_like(m)
    if 's' in d:
        s1[np.arange(len(idx)), d['s'][idx]] = 1.
    return dict(A=d['A'][idx].astype(np.float32), adj=d['adj'][idx].astype(np.float32), pos=pos, s1=s1, m=m)


def pair_number_inputs(x, W=None, self_loops=False):
    """pair_numbers: every number out of the embedding. The edge weights go to the pair register W [B, N, N] (BIG = no edge),
    the positions to x['reg'] [B, 1, N]; the embedding's weight channel A receives the fact 'there is an edge' and pos is
    zeroed. W: replacement weights (for recoding tests). self_loops: a loop that carries a weight in the stored file is an
    edge (W(i, i) = its weight, and the edge fact is on)."""
    N = x['m'].shape[1]
    m = x['m'] > 0
    eye = np.eye(N, dtype=bool)[None]
    edge = (x['adj'] > 0) & ~eye & m[:, :, None] & m[:, None, :]
    A = np.asarray(x['A'] if W is None else W, np.float32)
    out = dict(x, A=edge.astype(np.float32), pos=np.zeros_like(x['pos']), reg=np.asarray(x['pos'], np.float32)[:, None, :],
               W=np.where(edge, A, np.float32(PREG_BIG)).astype(np.float32))
    if self_loops:
        ar = np.arange(N)
        A0 = np.asarray(x['A'], np.float32)
        loop = (np.asarray(x['adj'])[:, ar, ar] > 0) & (A0[:, ar, ar] != 0) & m
        Wn, An = out['W'].copy(), out['A'].copy()
        Wn[:, ar, ar] = np.where(loop, A[:, ar, ar], np.float32(PREG_BIG))
        An[:, ar, ar] = loop.astype(np.float32)
        out = dict(out, W=Wn, A=An)
    return out


def item_inputs(d, idx, rng=None, per_n=()):
    """x for a non-graph task: f [B, M, F] = (pos, item numbers, item masks, one-hot categories), inputs sorted by name within
    each group; g [B, G] per-input numbers; sc [B, M, S] item numbers other than pos; m [B, M]. rng (training): pos replaced,
    as process_random_pos does, by sorted uniforms within each run of items that starts where pos = 0 (a string's two parts).
    per_n: item numbers divided by the input's own n (task scheduling's deadlines, in the units of pos)."""
    n = d['n'][idx].astype(np.int32)
    N = d['x_pos'].shape[1]
    m = (np.arange(N)[None] < n[:, None]).astype(np.float32)
    pos = d['x_pos'][idx].astype(np.float64)
    if rng is not None:
        u = rng.random(m.shape)
        seg = np.cumsum((pos == 0) & (m > 0), 1)
        srt = np.sort(np.where(m > 0, seg + u, np.inf), 1)
        pos = np.where(m > 0, srt - seg, 0.)
    keys = sorted(k for k in d if k.startswith('x_') and k != 'x_pos')
    scal = [k for k in keys if d[k].ndim == 2 and d[k].dtype.kind == 'f']
    msk = [k for k in keys if d[k].ndim == 2 and d[k].dtype.kind != 'f']
    cat = [k for k in keys if d[k].ndim == 3]
    graph = [k for k in keys if d[k].ndim == 1]
    nd = np.maximum(d['n'][idx], 1).astype(np.float64)[:, None]
    num = lambda k: d[k][idx] / nd if k[2:] in per_n else d[k][idx]
    f = [pos[..., None]] + [num(k)[..., None].astype(np.float64) for k in scal + msk] + [d[k][idx].astype(np.float64) for k in cat]
    f = np.concatenate(f, -1) * m[..., None]
    sc = np.stack([num(k).astype(np.float64) for k in scal], -1) * m[..., None] if scal else np.zeros(m.shape + (0,))
    g = np.stack([d[k][idx].astype(np.float64) for k in graph], -1) if graph else np.zeros((len(idx), 0))
    return dict(f=f.astype(np.float32), g=g.astype(np.float32), sc=sc.astype(np.float32), m=m)


def register_inputs(x):
    """compare_first: every input number moves from the embedding's inputs into x['reg'] [B, T, N], T = 1 + S + G (pos, the S
    other item numbers, the G per-input numbers); f's number columns, g and sc are zeroed."""
    f, g = x['f'].copy(), x['g']
    S = x['sc'].shape[-1]
    N = f.shape[1]
    reg = np.concatenate([f[:, :, :1 + S].transpose(0, 2, 1), np.repeat(g[:, :, None], N, 2)], 1).astype(np.float32)
    f[:, :, :1 + S] = 0.
    return dict(x, f=f, g=np.zeros_like(g), sc=np.zeros_like(x['sc']), reg=reg)


def register_kinds(task, d):
    """compare_first's 'registers' for a task: groups of register rows (from 1; row 1 is pos) measured in the same units."""
    keys = sorted(k for k in d if k.startswith('x_') and k != 'x_pos')
    names = ['pos'] + [k[2:] for k in keys if d[k].ndim == 2 and d[k].dtype.kind == 'f'] + [k[2:] for k in keys if d[k].ndim == 1]
    groups = [g for g in SAME_UNITS.get(task, []) if all(q in names for q in g)]
    rest = [[q] for q in names if not any(q in g for g in groups)]
    return sorted(sorted(names.index(q) + 1 for q in g) for g in rest + groups)


def two_inputs(x):
    """The second cold block's inputs from a task's census inputs x (numpy): every number out of the embedding, as
    pair_numbers (graph tasks: weights to the pair register W, positions to register row 1) or compare_first (other tasks:
    the item and input numbers to register rows 1.., no W), the register padded to TWO_ROWS rows."""
    if 'f' in x:
        xr = register_inputs(x)
        B, _, N = xr['reg'].shape
        out, W = dict(f=xr['f'], g=xr['g'], sc=xr['sc'], m=x['m']), np.full((B, N, N), PREG_BIG, np.float32)
    else:
        xr = pair_number_inputs(x)
        out, W = dict(A=xr['A'], adj=xr['adj'], pos=xr['pos'], s1=xr['s1'], m=xr['m']), xr['W']
    reg = np.asarray(xr['reg'], np.float32)
    B, T, N = reg.shape
    assert T <= TWO_ROWS, f'{T} register rows; the fixed layout has {TWO_ROWS}'
    reg = np.concatenate([reg, np.zeros((B, TWO_ROWS - T, N), np.float32)], 1)
    return dict(out, reg=reg, W=np.asarray(W, np.float32))


def dp_line(pos, m, group):
    """dp_program: nxt[b, k, j] = j is the item right after k (same group); first[b, i] = i is the first item of its group."""
    real = (m > 0); same = (group[:, :, None] == group[:, None, :]) & real[:, :, None] & real[:, None, :]
    later = same & (pos[:, None, :] > pos[:, :, None])
    between = np.einsum('bkm,bmj->bkj', later.astype(np.int32), later.astype(np.int32)) > 0
    nxt = later & ~between
    return nxt, real & ~later.any(1)


def dp_inputs(d, idx, x, task):
    """dp_program: the line from the stored positions and the program's input pair registers (x: register_inputs's output).
    optimal_bst W: p_k + q_j along the line, q on the diagonal; find_maximum_subarray_kadane W: -key; lcs_length: N (0 along
    the line), M (-1 where two characters of the two strings are equal), cell."""
    m = x['m']; B, N = m.shape; ar = np.arange(N); f32 = lambda a: np.asarray(a[idx], np.float32)
    grp = np.asarray(d['x_string'][idx]) if 'x_string' in d else np.zeros((B, N), int)
    nxt, first = dp_line(np.asarray(d['x_pos'][idx], np.float64), m, grp); big = np.full((B, N, N), np.float32(PREG_BIG)); real = m > 0
    out = dict(x, nxt=nxt.astype(np.float32), first=first.astype(np.float32))
    if task == 'optimal_bst':
        p_, q_ = f32(d['x_p']), f32(d['x_q']); W = np.where(nxt, p_[:, :, None] + q_[:, None, :], big); W[:, ar, ar] = np.where(real, q_, np.float32(PREG_BIG)); out['W'] = W.astype(np.float32)
    elif task == 'find_maximum_subarray_kadane':
        k_ = -f32(d['x_key']); W = np.where(nxt, k_[:, None, :] + np.zeros((B, N, 1), np.float32), big); W[:, ar, ar] = np.where(real, k_, np.float32(PREG_BIG)); out['W'] = W.astype(np.float32)
    elif task in ('naive_string_matcher', 'kmp_matcher'):   # M(k, j) = 0 where text item k and pattern item j hold the same letter
        key = np.asarray(d['x_key'][idx]).argmax(-1); cell = (grp[:, :, None] == 0) & (grp[:, None, :] == 1) & real[:, :, None] & real[:, None, :]
        out['M'] = np.where(cell & (key[:, :, None] == key[:, None, :]), np.float32(0), big).astype(np.float32)
        st = (grp == 1) & real; r = np.array(out['reg'], np.float64); p0 = r[:, 0]
        top = np.where(real & ~st, p0, -np.inf).max(1, keepdims=True); low = np.where(st, p0, np.inf).min(1, keepdims=True)
        r[:, 0] = np.where(st, p0 - low + top + 1., p0); out['reg'] = r.astype(np.float32)   # the pattern's positions move above the text's
    else:
        assert task == 'lcs_length', task
        key = np.asarray(d['x_key'][idx]).argmax(-1); cell = (grp[:, :, None] == 0) & (grp[:, None, :] == 1) & real[:, :, None] & real[:, None, :]
        out['N'] = np.where(nxt, np.float32(0), big).astype(np.float32); out['M'] = np.where(cell & (key[:, :, None] == key[:, None, :]), np.float32(-1), big).astype(np.float32)
        out['cell'] = cell.astype(np.float32)
    return out


def model_inputs(cfg, d, idx, rng=None):
    """The model's inputs for samples idx of a split, for a block config (numpy arrays)."""
    if 'x_pos' in d:
        x = item_inputs(d, idx, rng, tuple(cfg.get('scale_by_n', ())))
        if cfg['block'] == 'two_blocks' and cfg.get('second') == 'cold':
            return dict(x, cold=two_inputs(x))
        if cfg['block'] == 'dp_program':
            return dp_inputs(d, idx, register_inputs(x), cfg['program'])
        if cfg['block'] == 'item_memory':
            return register_inputs(x)
        return register_inputs(x) if cfg['block'] == 'compare_first' else x
    x = graph_inputs(d, idx, rng)
    if cfg.get('positions') is False:  # positions withheld (the constraint problems): every position the block gets is 0
        x = dict(x, pos=np.zeros_like(x['pos']))
    if cfg['block'] == 'item_memory':   # weighted graphs: pair_numbers' data path; others: positions as the one register row
        if cfg.get('item_w'):
            return pair_number_inputs(x, self_loops=cfg['item_w'] == 'self')
        return dict(x, pos=np.zeros_like(x['pos']), reg=x['pos'][:, None, :].astype(np.float32))
    if cfg['block'] == 'compare_first':  # a graph task in compare-first: pos, its only number, rides as the one register row
        assert np.isin(x['A'], (0., 1.)).all(), 'compare_first on a graph task needs an unweighted graph'
        return dict(x, pos=np.zeros_like(x['pos']), reg=x['pos'][:, None, :].astype(np.float32))
    if cfg['block'] == 'two_blocks' and cfg.get('second') == 'cold':
        return dict(x, cold=two_inputs(x))
    return pair_number_inputs(x, self_loops=bool(cfg.get('self_loops'))) if cfg['block'] == 'pair_numbers' else x


def feature_counts(d):
    """Feature counts for init_params: None for a graph task, else F (per item), G (per input) and S (item numbers)."""
    if 'x_pos' not in d:
        return None
    x = item_inputs(d, np.arange(1))
    return dict(F=int(x['f'].shape[-1]), G=int(x['g'].shape[-1]), S=int(x['sc'].shape[-1]))


def targets(d, idx, outs):
    return {name: d[name][idx].astype(np.int32) for name, _ in outs}
