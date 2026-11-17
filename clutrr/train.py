"""Training the CLUTRR organs (clutrr.py reads them). Three trainers and the files between them:

  block        the bit block: 20 kept bits per ordered pair of people, one per relation; a learned rule table (a, b -> c)
               chains them; a read-out at the question's pair. Trained from the stories' answers. Its input is either the
               benchmark's own graph of each story (arm 'graph': the graph organs) or facts read by the frozen model (arm
               'facts': the zero-shot organs, and the trained organs' fallback block).
  facts        a reading (the frozen model's zero-shot word and its probability for every candidate pair, or the read-in's
               probabilities written as a reading by reading_from_probs) -> the facts the block is given: every pair whose
               relation is not 'unknown' and whose score is at least a threshold.
  readin       the read-in: one linear map from the frozen model's last-layer state (clutrr.Reader.states) to 21 scores
               (20 relations, nothing stated). First distilled: fitted to the model's own zero-shot reading of the training
               stories' pairs. Then trained from answers through the cost read with a fixed rule table (the zero-shot block's
               table of the same seed): the read-in's probabilities are the facts, each relation at the question's pair gets
               the probability of its most probable derivation, and the loss is the cross-entropy with the story's answer plus
               0.1 x the cross-entropy to the zero-shot reading. The read-in kept is the best on validation.

  python clutrr/train.py block '<job json>'        keys: out, data (folder <split>/<part>.csv), split, arm (graph | facts),
                                                   facts (folder of <split>_<part>.json, arm facts), seed, steps (2000), batch
                                                   (64), lr (0.02), late (4), loop_c (3), phase1_steps (steps / 2), phase1_k (2),
                                                   eval_parts (validation, test)
  python clutrr/train.py facts <reading folder> <split> <threshold> <out folder> [p | share]
  python clutrr/train.py reading <learn folder> <name> <data folder> <split> <out folder>
  python clutrr/train.py readin '<job json>'       keys: out, data, split, states_train / states_vt (folders of
                                                   <split>_<part>_H.npy and _index.json), reading_train / reading_vt (folders of
                                                   <split>_<part>_reading.json, the zero-shot reading), tables {seed: [[a, b, c]]},
                                                   steps (12,000), every (500), pull (0.1), batch (64), lr (0.0002), distil_epochs (60)

Every choice was made on validation stories (choose_reading, choose_costs); the test stories were read once per stage.
The released organs were trained so: the graph and zero-shot blocks and the fallback blocks with `block` (2,000 updates; the
zero-shot facts at the chosen threshold of the model's reading, the fallback's at the chosen threshold of its read-in's),
the read-ins with `readin` on an NVIDIA L40S (12,000 updates, checked every 500). The blocks were trained on a laptop CPU;
on another CPU the same runs (seed 0) give the released rule tables, accuracies and (where they were kept) every answer,
with weights within 2e-6.
"""
import ast
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

csv.field_size_limit(10 ** 9)
RELS = ['aunt', 'brother', 'daughter', 'daughter-in-law', 'father', 'father-in-law', 'granddaughter', 'grandfather', 'grandmother',
        'grandson', 'husband', 'mother', 'mother-in-law', 'nephew', 'niece', 'sister', 'son', 'son-in-law', 'uncle', 'wife']


def loops_for(n, c=3):
    return int(math.ceil(math.log2(max(n, 2)))) + c


# ============================================================================================== stories, facts, readings
def load_stories(folder, split, part):
    """-> stories: dict(k, n, edges [(i, j, relation name)], q (a, b), y relation name, proof, id). An edge (i, j, r): j is i's r."""
    out = []
    for r in csv.DictReader(open(Path(folder) / split / f'{part}.csv', newline='')):
        se, et = ast.literal_eval(r['story_edges']), ast.literal_eval(r['edge_types']); q = tuple(ast.literal_eval(r['query_edge']))
        out.append(dict(k=int(r['task_name'].split('.')[1]), n=max(max(a, b) for a, b in se + [q]) + 1, edges=[(a, b, t) for (a, b), t in zip(se, et)],
                        q=q, y=r['target_text'], proof=r['proof_state'], id=r['id']))
    return out


def load_facts(folder, split, part):
    """Arm 'facts': stories as a frozen model read them (facts below): people in order of first appearance in the text, the
    question's pair, and the relations read. -> the same dicts as load_stories, with edges = the facts read and no proof."""
    return [dict(k=s['k'], n=s['n'], edges=[(a, c, t) for a, c, t in s['facts']], q=tuple(s['q']), y=s['y'], proof=None, id=s['id'])
            for s in json.loads((Path(folder) / f'{split}_{part}.json').read_text())]


def facts(src, split, p_min, out, score='p'):
    """A reading -> the facts the block is given. A candidate (A, B) becomes the fact "B is A's r" when the reading's answer is a
    relation (not 'unknown') and its score is at least p_min. The score is the probability of the relation's word ('p'), or that
    probability as a share of the probability on all relation words ('share', for a base model whose probability is spread over
    words that are not relations). Nothing here looks at an answer or at the graph."""
    src, out = Path(src), Path(out); out.mkdir(parents=True, exist_ok=True); p_min = float(p_min)
    for part in ('train', 'validation', 'test'):
        f = src / f'{split}_{part}_reading.json'
        if not f.exists():
            continue
        d = json.loads(f.read_text()); R = d['relations']; res = []
        for s in d['stories']:
            fs = [[a, b, R[rel]] for a, b, rel, p, mass, second in s['cands'] if R[rel] != 'unknown' and (p if score == 'p' else p / max(mass, 1e-9)) >= p_min]
            res.append(dict(id=s['id'], k=s['k'], n=len(s['names']), q=s['q'], y=s['y'], facts=fs))
        (out / f'{split}_{part}.json').write_text(json.dumps(res))


def reading_from_probs(src, name, data, split, out):
    """The read-in's probabilities (readin below: probs_<name>_<part>.npy, index_<part>.json) -> reading files in the reader's
    format, so that facts, the block and the cost read are used unchanged. A pair's entries: every relation with probability
    0.02 or more, with its probability; 'unknown' if that is the top answer. The story's answer is copied from the csv."""
    src, out = Path(src), Path(out); out.mkdir(parents=True, exist_ok=True)
    for part in ('train', 'validation', 'test'):
        if not (src / f'probs_{name}_{part}.npy').exists():
            continue
        P = np.load(src / f'probs_{name}_{part}.npy').astype(np.float32); ix = json.loads((src / f'index_{part}.json').read_text())
        rows = {r['id']: r for r in csv.DictReader(open(Path(data) / split / f'{part}.csv', newline=''))}; stories = []
        for i, sid in enumerate(ix['ids']):
            cands = []
            for c in range(ix['first'][i], ix['first'][i + 1]):
                p = P[c]; top = int(p.argmax()); a, b = ix['a'][c], ix['b'][c]
                if top == 20:
                    cands.append([a, b, 20, round(float(p[20]), 4), 1.0, 0.0])
                for r in np.flatnonzero(p[:20] >= .02):
                    cands.append([a, b, int(r), round(float(p[r]), 4), 1.0, 0.0])
            stories.append(dict(id=sid, k=int(rows[sid]['task_name'].split('.')[1]), names=[f'p{j}' for j in range(ix['n'][i])], q=ix['q'][i],
                                y=rows[sid]['target_text'], cands=cands))
        (out / f'{split}_{part}_reading.json').write_text(json.dumps(dict(relations=RELS + ['unknown'], model='read-in on frozen Qwen2.5-7B-Instruct states',
                                                                          chat=True, readin=name, stories=stories)))


def choose_reading(val):
    """{(reading, threshold): validation accuracy of the block} -> the choice: the highest; ties: the higher threshold, then a
    base model's reading."""
    return max(val, key=lambda k: (val[k], k[1], 'base' in k[0]))


def choose_costs(val):
    """{(bit threshold, cost threshold, table): validation accuracy of the cost read} -> the choice: the highest; ties: the lower
    cost threshold, then the lower bit threshold, then the table's name."""
    return max(val, key=lambda k: (val[k], -k[1], -k[0], k[2]))


# ============================================================================================================ the block
def true_table(stories):
    """The generator's rules as the training proofs name them: {(a, b): c}. Reported beside the learned table; never trained on."""
    T = {}
    for s in stories:
        for step_ in ast.literal_eval(s['proof']):
            for (_, rel, _), body in step_.items():
                T[tuple(x[1] for x in body)] = rel
    return T


def batch_of(stories, idx, rid, N):
    """-> I [B, R, N, N] stated planes, m [B, N] real people, q [B, 2], y [B]."""
    B, R = len(idx), len(rid); I = np.zeros((B, R, N, N), np.float32); m = np.zeros((B, N), np.float32); q = np.zeros((B, 2), np.int32); y = np.zeros(B, np.int32)
    for b, i in enumerate(idx):
        s = stories[i]; m[b, :s['n']] = 1; q[b] = s['q']; y[b] = rid[s['y']]
        for a, c, t in s['edges']:
            if t in rid and a != c and a < N and c < N:
                I[b, rid[t], a, c] = 1
    return I, m, q, y


def block_step(p, I, L, m):
    """One loop. Forward values are hard bits; slopes come from a bounded soft twin: 'some middle exists' as 1 - exp(-count);
    'some learned rule fires' as 1 - exp(-sum of the weights of the rules whose join is on). A rule's weight is sigmoid(W); the
    rule is on when W > 0. A bit never turns off."""
    import jax
    import jax.numpy as j
    sg = jax.lax.stop_gradient
    T = j.maximum(I, L); cnt = j.einsum('baij,bj,bcjk->bacik', T, m, T)
    Js = 1. - j.exp(-cnt); J = Js + sg((cnt > 0).astype(cnt.dtype) - Js)
    zs = 1. - j.exp(-j.einsum('bacik,acr->brik', J, jax.nn.sigmoid(p['W'])))
    zh = (j.einsum('bacik,acr->brik', sg(J), (p['W'] > 0).astype(J.dtype)) > 0).astype(J.dtype)
    return j.maximum(L, zs + sg(zh - zs))


def block_run(p, I, m, q, K, at):
    """-> logits [len(at), B, R] after the loop counts in `at`, and the kept bits after K loops."""
    import jax
    import jax.numpy as j
    L = j.zeros_like(I); out = []; b = j.arange(I.shape[0])
    for k in range(K):
        L = block_step(p, I, L, m)
        if k + 1 in at:
            out.append(L[b, :, q[:, 0], q[:, 1]] * jax.nn.softplus(p['g']) + p['bias'])
    return j.stack(out), L


def train_block(job):
    """Two phases: the table is learned on the stories of phase1_k relations with ONE loop (so the only join at the question's
    pair is between two stated relations); then it is frozen and the read-out's scales and offsets are trained on all training
    stories, at ceil(log2 n) + loop_c loops less 0 or 1, the loss read there and `late` loops later. Released parameters: the mean
    of the snapshots every 50 updates from half way."""
    import jax
    import jax.numpy as j
    import optax
    T0 = time.time(); say = lambda *a: print(*a, f'[{time.time() - T0:.0f}s]', flush=True)
    out = Path(job['out']); out.mkdir(parents=True, exist_ok=True); split = job.get('split', 'gen_train234_test2to10'); arm = job.get('arm', 'graph')
    steps = job.get('steps', 2000); p1 = int(job.get('phase1_steps', steps // 2)); p1k = int(job.get('phase1_k', 2)); B = job.get('batch', 64); late = int(job.get('late', 4))
    graph = {k: load_stories(job['data'], split, k) for k in ['train'] + job.get('eval_parts', ['validation', 'test'])}
    parts = graph if arm == 'graph' else {k: load_facts(job['facts'], split, k) for k in graph}
    assert all(len(parts[k]) == len(graph[k]) for k in graph)
    rels = sorted({s['y'] for s in graph['train']} | {t for s in graph['train'] for _, _, t in s['edges']}); rid = {r: i for i, r in enumerate(rels)}; R = len(rels)
    (out / 'readin.json').write_text(json.dumps(dict(relations=rels, split=split, arm=arm, facts=job.get('facts'), bit='bit r at (i, j): j is i\'s r',
                                                     items='the people of a story, in the order of the benchmark\'s graph'), indent=1))
    ks = iter(jax.random.split(jax.random.PRNGKey(job['seed']), 4))
    p = dict(W=jax.random.normal(next(ks), (R, R, R)) * .5 - 3., g=j.full((R,), 2.95), bias=j.zeros((R,)))
    sched = optax.join_schedules([optax.warmup_cosine_decay_schedule(0., job.get('lr', 2e-2), min(100, p1 // 10), p1, 1e-4),
                                  optax.warmup_cosine_decay_schedule(0., job.get('lr', 2e-2), min(100, max(1, (steps - p1) // 10)), max(1, steps - p1), 1e-4)], [p1])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)); os_ = opt.init(p)
    tr = parts['train']; by_n = {}
    for i, s in enumerate(tr):
        by_n.setdefault(s['n'], []).append(i)
    rng = np.random.default_rng(job['seed']); sizes = sorted(by_n); wts = np.array([len(by_n[n]) for n in sizes], float); wts /= wts.sum(); fns = {}

    def loss_fn(q_, I, m, qq, y, K):
        lg, _ = block_run(q_, I, m, qq, K + (late if K > 1 else 0), (K, K + late) if K > 1 else (1,))
        return optax.softmax_cross_entropy_with_integer_labels(lg, j.broadcast_to(y, lg.shape[:2])).mean(), (lg[0].argmax(-1) == y).mean()

    def step_fn(K, frozen):
        if (K, frozen) not in fns:
            @jax.jit
            def f(q_, os_, I, m, qq, y):
                (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(q_, I, m, qq, y, K); up, os2 = opt.update(g, os_, q_)
                if frozen:
                    up = dict(up, W=j.zeros_like(up['W']))
                return optax.apply_updates(q_, up), os2, l, a
            fns[(K, frozen)] = f
        return fns[(K, frozen)]
    swa, n_swa, log = None, 0, []
    for it in range(steps):
        n = (p1k + 1) if it < p1 else int(rng.choice(sizes, p=wts)); idx = rng.choice(by_n[n], B)
        K = 1 if it < p1 else loops_for(n, job.get('loop_c', 3)) - int(rng.integers(0, 2)); I, m, qq, y = batch_of(tr, idx, rid, n)
        p, os_, l, a = step_fn(K, it >= p1)(p, os_, I, m, qq, y)
        if it >= steps // 2 and (it + 1) % 50 == 0:
            n_swa += 1; swa = p if swa is None else jax.tree_util.tree_map(lambda s_, q_: s_ + (q_ - s_) / n_swa, swa, p)
        if it % max(1, steps // 40) == 0 or it == steps - 1:
            log.append(dict(step=it, loss=float(l), acc=float(a), n=n, K=K, t=time.time() - T0)); say(json.dumps(log[-1]))
            (out / 'progress.jsonl').write_text('\n'.join(json.dumps(r) for r in log))
        if not np.isfinite(float(l)):
            raise SystemExit('loss is not finite')
    swa = p if swa is None else swa
    np.savez(out / 'params.npz', **{k: np.asarray(v) for k, v in p.items()}); np.savez(out / 'params_swa.npz', **{k: np.asarray(v) for k, v in swa.items()})

    def table_of(q_):
        on = np.asarray(q_['W']) > 0; return sorted((rels[a], rels[b], rels[c]) for a, b, c in zip(*np.nonzero(on)))
    tt = true_table(graph['train']); learned_final = table_of(p); learned = table_of(swa); want = sorted((a, b, c) for (a, b), c in tt.items())
    tab = dict(rules_in_training_proofs=len(want), learned=len(learned), learned_and_true=len(set(learned) & set(want)), missing=sorted(set(want) - set(learned)),
               extra=sorted(set(learned) - set(want)), same_with_final_weights=learned == learned_final, learned_with_final_weights=len(learned_final))
    (out / 'table.json').write_text(json.dumps(dict(summary=tab, learned=learned), indent=1))

    rows = []; readers = {}
    for part in job.get('eval_parts', ['validation', 'test']):
        st = parts[part]; byn = {}
        for i, s in enumerate(st):
            byn.setdefault(s['n'], []).append(i)
        res = {}; per_story = [None] * len(st)
        for n, ids in sorted(byn.items()):
            K0 = loops_for(n); Kmax = K0 + 16
            if (n, Kmax) not in readers:
                readers[(n, Kmax)] = jax.jit(lambda q_, I, m, qq, K0=K0, Kmax=Kmax: block_run(q_, I, m, qq, Kmax, (K0, K0 + 4, Kmax)))
            for b0 in range(0, len(ids), 256):
                idx = ids[b0:b0 + 256]; I, m, qq, y = batch_of(st, idx, rid, n); lg, L = readers[(n, Kmax)](swa, I, m, qq); lg = np.asarray(lg); pr = lg.argmax(-1)
                bits = np.asarray(L)[np.arange(len(idx)), :, qq[:, 0], qq[:, 1]]
                for b, i in enumerate(idx):
                    res.setdefault(st[i]['k'], []).append(dict(K=bool(pr[0, b] == y[b]), K4=bool(pr[1, b] == y[b]), K16=bool(pr[2, b] == y[b]), rest=bool((pr[1, b] == pr[2, b])),
                                                               on=int(bits[b].sum()), answer_on=bool(bits[b, y[b]] > 0)))
                    per_story[i] = dict(k=st[i]['k'], answer=rels[int(pr[1, b])], right=bool(pr[1, b] == y[b]), bits_on=[rels[r_] for r_ in np.flatnonzero(bits[b] > 0)])
        for k in sorted(res):
            v = res[k]; f = lambda a: sum(x[a] for x in v)
            rows.append(dict(part=part, k=k, questions=len(v), right=dict(K=f('K'), K4=f('K4'), K16=f('K16')), accuracy_K4=round(f('K4') / len(v), 4), same_answer_K4_K16=f('rest'),
                             no_bit_on=sum(x['on'] == 0 for x in v), several_bits_on=sum(x['on'] > 1 for x in v), answer_bit_on=f('answer_on')))
        tot = [x for v in res.values() for x in v]
        rows.append(dict(part=part, k='all', questions=len(tot), right=dict(K=sum(x['K'] for x in tot), K4=sum(x['K4'] for x in tot), K16=sum(x['K16'] for x in tot)),
                         accuracy_K4=round(sum(x['K4'] for x in tot) / len(tot), 4)))
        say(json.dumps(rows[-1]))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1)); (out / f'answers_{part}.json').write_text(json.dumps(per_story))
    result = dict(split=split, arm=arm, relations=R, steps=steps, phase1_steps=p1, phase1_k=p1k, seed=job['seed'], train_seconds=log[-1]['t'],
                  final_train_loss=log[-1]['loss'], table=tab, seconds=time.time() - T0)
    (out / 'result.json').write_text(json.dumps(result)); return rows


# ============================================================================================================ the read-in
def train_readin(job):
    """The read-in, distilled then trained from answers (module docstring). Writes the probabilities of every pair of every part
    for the distilled read-in and for each seed's trained one (float16 [pairs, 21]), the read-ins (readin_distilled.npz,
    readin_trained_s<seed>.npz: U, b, mu, sd), index_<part>.json and a log. The test part's answers are never read."""
    import torch
    import torch.nn.functional as F
    T0 = time.time(); say = lambda *a: print(*a, f'[{time.time() - T0:.0f}s]', flush=True); out = Path(job['out']); out.mkdir(parents=True, exist_ok=True)
    split = job['split']; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log = {}

    def load(part, folders, labels_from=None, answers=False):
        Hs, idx = [], []
        for f in folders:
            Hs.append(np.load(Path(f) / f'{split}_{part}_H.npy')); idx += json.loads((Path(f) / f'{split}_{part}_index.json').read_text())
        H = np.concatenate(Hs); story = np.concatenate([np.full(len(s['cands']), i) for i, s in enumerate(idx)]); ab = np.array([c for s in idx for c in s['cands']]).reshape(-1, 2)
        assert len(H) == len(story) == len(ab)
        d = dict(H=torch.from_numpy(H.astype(np.float32)), story=torch.from_numpy(story), a=torch.from_numpy(ab[:, 0]), b=torch.from_numpy(ab[:, 1]), n=np.array([s['n'] for s in idx]),
                 q=np.array([s['q'] for s in idx]), ids=[s['id'] for s in idx], first=np.concatenate([[0], np.cumsum([len(s['cands']) for s in idx])]), zs=None, y=None)
        if labels_from:   # the frozen model's own zero-shot reading of these pairs, checked pair by pair
            old = {}
            for f in labels_from:
                old.update({o['id']: o for o in json.loads((Path(f) / f'{split}_{part}_reading.json').read_text())['stories']})
            assert all([c[:2] for c in old[s['id']]['cands']] == s['cands'] for s in idx); d['zs'] = torch.tensor([c[2] for s in idx for c in old[s['id']]['cands']])
        if answers:
            rows = {r['id']: r for r in csv.DictReader(open(Path(job['data']) / split / f'{part}.csv', newline=''))}; d['y'] = torch.tensor([RELS.index(rows[i]['target_text']) for i in d['ids']])
        return d
    tr = load('train', job['states_train'], job['reading_train'], True); va = load('validation', job['states_vt'], job['reading_vt'], True); te = load('test', job['states_vt'])
    mu = tr['H'].mean(0); sd = tr['H'].std(0) + 1e-6; norm = lambda d: ((d['H'] - mu) / sd).to(dev); Xtr, Xva, Xte = norm(tr), norm(va), norm(te); W = Xtr.shape[1]
    say('pairs', len(Xtr), len(Xva), len(Xte), 'width', W)
    probs = lambda U, b, X: torch.cat([torch.softmax(X[i:i + 65536] @ U + b, -1) for i in range(0, len(X), 65536)])
    # 1. distilled: fitted to the model's own reading of the training pairs
    torch.manual_seed(0); U = torch.zeros(W, 21, device=dev, requires_grad=True); b = torch.zeros(21, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([U, b], lr=1e-3, weight_decay=1e-4); ztr = tr['zs'].to(dev); zva = va['zs'].to(dev)
    for ep in range(int(job.get('distil_epochs', 60))):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), 4096):
            jj = perm[i:i + 4096]; loss = F.cross_entropy(Xtr[jj] @ U + b, ztr[jj]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        agree = float(((Xva @ U + b).argmax(-1) == zva).float().mean()); agree_tr = float(((Xtr[:50000] @ U + b).argmax(-1) == ztr[:50000]).float().mean())
    U0, b0 = U.detach().clone(), b.detach().clone(); log['distil'] = dict(agreement_validation=agree, agreement_train=agree_tr, final_loss=float(loss)); say('distilled', json.dumps(log['distil']))

    def rows_of(d, sel):
        rows = np.concatenate([np.arange(d['first'][s], d['first'][s + 1]) for s in sel]); st = np.concatenate([np.full(d['first'][s + 1] - d['first'][s], i) for i, s in enumerate(sel)])
        return rows, torch.from_numpy(st).to(dev)

    def facts_of(d, sel, rows, st, Prows):
        """The stories sel of d as fact tensors: Prows [their pairs, 21] -> In [B, 20, N, N] (relation r at (a, b): b is a's r), questions [B, 2]."""
        N = int(d['n'][sel].max()); In = torch.zeros(len(sel), 20, N, N, device=dev); In[st, :, d['a'][rows].to(dev), d['b'][rows].to(dev)] = Prows[:, :20]
        return In, torch.from_numpy(d['q'][sel]).to(dev)

    def derive(In, ra, rb, rc):
        """The most probable derivation of each relation at each pair: [B, 20, N, N] (derived only)."""
        D = torch.zeros_like(In); N = In.shape[-1]; idx = rc.view(1, -1, 1, 1).expand(In.shape[0], -1, N, N)
        for _ in range(int(math.ceil(math.log2(max(N, 2)))) + 7):
            T = torch.maximum(In, D); D = D.scatter_reduce(1, idx, (T[:, ra].unsqueeze(-1) * T[:, rb].unsqueeze(-3)).amax(3), 'amax', include_self=True)
        return D

    def accuracy(d, P, rules):
        ra, rb, rc = rules; ok = 0
        with torch.no_grad():
            for i in range(0, len(d['n']), 512):
                sel = np.arange(i, min(i + 512, len(d['n']))); rows, st = rows_of(d, sel); In, q = facts_of(d, sel, rows, st, P[torch.from_numpy(rows).to(dev)]); D = derive(In, ra, rb, rc)
                s = D[torch.arange(len(sel), device=dev), :, q[:, 0], q[:, 1]]; ok += int(((s.argmax(-1) == d['y'][sel].to(dev)) & (s.max(-1).values > 0)).sum())
        return ok / len(d['n'])
    save = lambda name, U_, b_: [np.save(out / f'probs_{name}_{part}.npy', probs(U_, b_, X).cpu().numpy().astype(np.float16)) for part, X in (('train', Xtr), ('validation', Xva), ('test', Xte))]
    with torch.no_grad():
        save('distilled', U0, b0)
    np.savez(out / 'readin_distilled.npz', U=U0.cpu().numpy(), b=b0.cpu().numpy(), mu=mu.numpy(), sd=sd.numpy())
    for part, d in (('train', tr), ('validation', va), ('test', te)):
        (out / f'index_{part}.json').write_text(json.dumps(dict(ids=d['ids'], first=d['first'].tolist(), a=d['a'].tolist(), b=d['b'].tolist(), n=d['n'].tolist(), q=d['q'].tolist())))
    # 2. trained from answers through the cost read, training stories only; kept by validation accuracy
    steps, every, pull, B = int(job.get('steps', 12000)), int(job.get('every', 500)), float(job.get('pull', .1)), int(job.get('batch', 64))
    by_n = {int(n): np.flatnonzero(tr['n'] == n) for n in np.unique(tr['n'])}
    for seed, table in job['tables'].items():
        rules = tuple(torch.tensor([RELS.index(t[i]) for t in table], device=dev) for i in range(3)); torch.manual_seed(int(seed)); rng = np.random.default_rng(int(seed))
        U = U0.clone().requires_grad_(True); b = b0.clone().requires_grad_(True)
        opt = torch.optim.AdamW([U, b], lr=float(job.get('lr', 2e-4)), weight_decay=1e-4); curve = []
        with torch.no_grad():
            v0 = accuracy(va, probs(U, b, Xva), rules)
        best = (v0, 0, U.detach().clone(), b.detach().clone()); curve.append(dict(step=0, validation=v0)); say('seed', seed, 'rules', len(table), 'validation with the distilled read-in', round(v0, 4))
        for it in range(1, steps + 1):
            n = int(rng.choice(list(by_n))); sel = rng.choice(by_n[n], B); rows, st = rows_of(tr, sel); rt = torch.from_numpy(rows).to(dev); lg = Xtr[rt] @ U + b
            In, q = facts_of(tr, sel, rows, st, torch.softmax(lg, -1)); D = derive(In, *rules); s = D[torch.arange(B, device=dev), :, q[:, 0], q[:, 1]]
            loss_a = F.cross_entropy(torch.log(s + 1e-9), tr['y'][sel].to(dev)); loss_z = F.cross_entropy(lg, ztr[rt]); loss = loss_a + pull * loss_z; opt.zero_grad(); loss.backward(); opt.step()
            if it % every == 0:
                with torch.no_grad():
                    v = accuracy(va, probs(U, b, Xva), rules); ag = float(((Xva @ U + b).argmax(-1) == zva).float().mean())
                curve.append(dict(step=it, validation=v, loss_answer=float(loss_a), loss_pull=float(loss_z), agreement_with_zero_shot=ag)); say('seed', seed, json.dumps(curve[-1]))
                if v > best[0]:
                    best = (v, it, U.detach().clone(), b.detach().clone())
        with torch.no_grad():
            save(f'trained_s{seed}', best[2], best[3])
        np.savez(out / f'readin_trained_s{seed}.npz', U=best[2].cpu().numpy(), b=best[3].cpu().numpy(), mu=mu.numpy(), sd=sd.numpy())
        log[f'seed{seed}'] = dict(rules=len(table), validation_distilled=v0, validation_best=best[0], kept_at_update=best[1], curve=curve)
        (out / 'learn_log.json').write_text(json.dumps(log, indent=1))
    res = dict(split=split, seconds=time.time() - T0, distil=log['distil'],
               seeds={k: dict(validation_distilled=v['validation_distilled'], validation_best=v['validation_best'], kept_at_update=v['kept_at_update']) for k, v in log.items() if k.startswith('seed')})
    (out / 'learn_result.json').write_text(json.dumps(res)); return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'block':
        train_block(json.loads(sys.argv[2]))
    elif cmd == 'readin':
        print(json.dumps(train_readin(json.loads(sys.argv[2]))))
    elif cmd == 'facts':
        facts(*sys.argv[2:])
    elif cmd == 'reading':
        reading_from_probs(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
