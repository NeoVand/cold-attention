"""Cold attention beside a frozen language model.

A frozen Qwen2.5 model reads the text; a block of cold attention chains what it read and answers a yes-or-no question.
Nothing in the language model is trained or changed. The block keeps 3 bits for every ordered pair of name mentions. In
each loop, bit c of (u, v) turns on when some mention k passes one learned test with both (u, k) and (k, v); bits never
turn off; a test cannot ask for a bit to be off; and the answer can only be pushed towards yes by a bit that is on.

Two tasks:
  kinship     statements such as "Maria Lopez is the mother of Tom Baker." in eight wordings, and the question
              "Is X an ancestor of Y?". The frozen model gives, for every mention, a key (the sum of its input embeddings
              over the name's tokens), and for every sentence its own reading of who the parent is.
  comparison  facts "* A is older than B." over single-token names, and the question "Is X older than Y?". The frozen
              model's input embeddings are compared at the name positions.

Sections: 1 the generators of the evaluation contexts (numpy); 2 the read-in from the frozen model (torch and
transformers; nothing is trained); 3 the block (jax); 4 loading and answering.

  python organ.py demo [kinship/qwen2.5-7b/within/seed0] [--statements 64]    (downloads the frozen model; a GPU is advised)
  python organ.py demo comparison/qwen2.5-0.5b/seed0

Apache 2.0.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = 'mohsenvand/cold-attention-organ'
DEFAULT = 'kinship/qwen2.5-7b/within/seed0'   # the default organ: frozen Qwen2.5-7B, pooled reading, layout 'within'

# ===================================================================================================== 1. generators
# Kinship. A context: n statements, each one parent link (parent, child) said in one wording; statement order shuffled.
# Structures: 'lineage' (1 to 3 lines of descent: deep chains) and 'pedigree' (people in birth order; a person may have a
# stated mother and a stated father, drawn from the few people born just before). People are integer ids, rendered as
# full names. Wordings (text, the person named first is the parent, gender condition):
WORDINGS = (('{P} is the mother of {C}.', True, 'Pf'), ('{P} is the father of {C}.', True, 'Pm'), ('{P} is a parent of {C}.', True, ''),
            ('{C} is the son of {P}.', False, 'Cm'), ('{C} is the daughter of {P}.', False, 'Cf'), ('{C} is a child of {P}.', False, ''),
            ("{C}'s mother is {P}.", False, 'Pf'), ("{C}'s father is {P}.", False, 'Pm'))
NW = len(WORDINGS)
QUESTION = 'Is {X} an ancestor of {Y}?'
STRUCTS = ('lineage', 'pedigree')
KINDS = ('yes', 'reversed', 'unrelated', 'absent')
FEMALE = 'Maria Anna Sofia Emma Laura Julia Sarah Alice Clara Elena Nora Irene Lucy Rosa Helen Diana Marta Olga Paula Rita Sara Tina Vera Wendy Zoe Amy Beth Cora Dana Eva Faye Gina Hana Ida Jane Kate Lena Mia Nina Opal Pia Ruth Sue Tess Uma Val Willa Yara'.split()
MALE = 'Tom Paul Mark John Peter David James Henry Oscar Felix Hugo Ivan Jack Karl Leo Max Noah Otto Pablo Quinn Rob Sam Theo Uri Victor Walt Xavier Yuri Zack Adam Ben Carl Dean Eric Fred Gus Hank Ian Joel Kurt Luke Mike Neil Owen Pete Ray Sean Troy'.split()
LAST = 'Baker Lopez Kowalski Chen Okafor Novak Silva Meyer Tanaka Rossi Haddad Jensen Murphy Petrov Garcia Kim Dubois Larsen Weber Costa Nowak Singh Moreau Fischer Santos Ivanov Keller Romano Bauer Olsen Adams Brooks Carter Diaz Evans Foster Grant Hayes Irwin Jones Knight Lewis Mason Nolan Ortiz Parker Quimby Reed Stone Turner Usher Vance Walsh Young Zimmer Abbott Barnes Clark Dixon Ellis Flynn Gibson Hart Ingram Joyce Kane Lowe Marsh Nash Owens Price Rhodes Shaw Tate Underwood Vega Wade Yates Zhang Ahmed Bello Cruz Dang Endo Farah Gupta Hoang Ito Jovic Khan Lund Mbeki Nair Oduya Park Qureshi Rao Sato Tran Ueda Varga Wolde Xu Yilmaz Zapata Arnold Bishop Conway Drake Egan Finch Gould Holt Ivers Jarvis Kemp Lang Monroe Neal Pratt'.split()
N_PEOPLE = len(FEMALE + MALE) * len(LAST)
TRAIN_WORDINGS_LARGE = [0, 2, 3, 4, 5, 7]   # the wordings in training for the 7B and 1.5B organs; 1 and 6 are never trained on
KINSHIP_SPLITS = dict(
    train=dict(seed=20261201, sizes=list(range(1, 17)), per_size=384),
    val6=dict(seed=20261206, sizes=[16], per_size=48, wordings=TRAIN_WORDINGS_LARGE),
    val16=dict(seed=20261202, sizes=[16], per_size=48), val24=dict(seed=20261203, sizes=[24], per_size=48),
    test=dict(seed=20261204, sizes=[16, 32, 64, 128, 256, 512], per_size=24), test1000=dict(seed=20261205, sizes=[1000], per_size=12))


def person(i):
    """-> (full name, is female) of person id i (0 .. N_PEOPLE - 1)."""
    first = FEMALE + MALE
    f = i % len(first)
    return f'{first[f]} {LAST[i // len(first)]}', f < len(FEMALE)


def _lineage(rng, n):
    c = int(rng.integers(1, 4)) if n >= 6 else 1
    cuts = np.sort(rng.choice(np.arange(1, n), c - 1, replace=False)) if c > 1 else np.array([], int)
    e, off = [], 0
    for L in np.diff(np.concatenate([[0], cuts, [n]])):
        e += [(off + i, off + i + 1) for i in range(L)]
        off += L + 1
    return np.array(e, int), off


def _pedigree(rng, n, female):
    w = int(rng.integers(2, 6)); r = float(rng.uniform(.2, .6)); e = []; anc = [set()]; k = 0
    while len(e) < n:
        k += 1; lo = max(0, k - w); p1 = int(rng.integers(lo, k)); par = [p1]
        if len(e) + 2 <= n and rng.random() < r:
            cand = [p for p in range(lo, k) if female[p] != female[p1] and p not in anc[p1] and p1 not in anc[p]]
            if cand:
                par.append(int(cand[int(rng.integers(len(cand)))]))
        a_ = set()
        for p in par:
            e.append((p, k)); a_ |= anc[p] | {p}
        anc.append(a_)
    return np.array(e, int), k + 1


def _distances(edges, E):
    """-> D [E, E]: the fewest links on a chain from x down to y; -1 if none."""
    nb = [[] for _ in range(E)]
    for a, b in edges:
        nb[a].append(b)
    D = -np.ones((E, E), np.int32)
    for s in range(E):
        D[s, s] = 0; front = [s]; d = 0
        while front:
            d += 1; nxt = []
            for u in front:
                for v in nb[u]:
                    if D[s, v] < 0:
                        D[s, v] = d; nxt.append(v)
            front = nxt
    return D


def kinship_context(rng, n, struct, q_per=8, wordings=None):
    """-> dict: n, struct, facts [n, 2] person ids (parent, child; statement order), cls [n] wording, q [Q, 2], label,
    kind (yes, reversed, unrelated, absent), depth (links on the shortest chain of a yes)."""
    wordings = list(range(NW)) if wordings is None else list(wordings)
    if struct == 'lineage':
        edges, E = _lineage(rng, n); ent = rng.choice(N_PEOPLE, E + 2, replace=False)
    else:
        ent = rng.choice(N_PEOPLE, n + 3, replace=False); female = np.array([person(int(i))[1] for i in ent]); edges, E = _pedigree(rng, n, female)
    D = _distances(edges, E); yes = np.argwhere(D > 0); depths = np.unique(D[D > 0]); qs = []
    for _ in range(q_per // 2):
        d = int(rng.choice(depths)); cand = yes[D[yes[:, 0], yes[:, 1]] == d]; x, y = cand[rng.integers(len(cand))]; qs.append((ent[x], ent[y], 1, 0, d))
    no_un = np.argwhere((D < 0) & (D.T < 0))
    for _ in range(q_per - q_per // 2):
        r = rng.random()
        if r < .4 or (r < .7 and len(no_un) == 0):
            cand = yes[rng.integers(len(yes))]; qs.append((ent[cand[1]], ent[cand[0]], 0, 1, -1))
        elif r < .7:
            x, y = no_un[rng.integers(len(no_un))]; qs.append((ent[x], ent[y], 0, 2, -1))
        else:
            x = ent[int(rng.integers(E))]; out = ent[E + int(rng.integers(2))]; qs.append(((x, out) if rng.random() < .5 else (out, x)) + (0, 3, -1))
    qs = np.array(qs, np.int64)[rng.permutation(len(qs))]; order = rng.permutation(n); facts = ent[edges][order]; cls = np.zeros(n, np.int8)
    for k, (p, c) in enumerate(facts):
        pf, cf = person(int(p))[1], person(int(c))[1]
        ok = [w for w in wordings if WORDINGS[w][2] in ('', 'Pf' if pf else 'Pm', 'Cf' if cf else 'Cm')]
        cls[k] = ok[int(rng.integers(len(ok)))]
    return dict(n=n, struct=STRUCTS.index(struct), facts=facts.astype(np.int32), cls=cls, q=qs[:, :2].astype(np.int32),
                label=qs[:, 2].astype(np.int8), kind=qs[:, 3].astype(np.int8), depth=qs[:, 4].astype(np.int32))


def kinship_split(name):
    """-> the contexts of one evaluation split (as they were generated for training and evaluation)."""
    a = KINSHIP_SPLITS[name]; rng = np.random.default_rng(a['seed'])
    return [kinship_context(rng, n, STRUCTS[i % 2], wordings=a.get('wordings')) for n in a['sizes'] for i in range(a['per_size'])]


def kinship_text(c):
    """-> (statement sentences, question sentences) of a kinship context."""
    st = [WORDINGS[w][0].format(P=person(int(p))[0], C=person(int(ch))[0]) for (p, ch), w in zip(c['facts'], c['cls'])]
    return st, [QUESTION.format(X=person(int(x))[0], Y=person(int(y))[0]) for x, y in c['q']]


# Comparison. A context: n facts "a is older than b" over distinct names, no cycles; families chain, tree, mixed.
FAMILIES = ('chain', 'tree', 'mixed')
COMPARISON_SPLITS = dict(
    train=dict(seed=20261115, sizes=list(range(1, 17)), per_size=384),
    val16=dict(seed=20261116, sizes=[16], per_size=48), val24=dict(seed=20261117, sizes=[24], per_size=48),
    test=dict(seed=20261118, sizes=[16, 32, 64, 128, 256, 512], per_size=24), test1000=dict(seed=20261119, sizes=[1000], per_size=12))


def _comparison_structure(rng, n, family):
    if family == 'chain':
        return _lineage(rng, n)
    if family == 'tree':
        e = []
        for k in range(1, n + 1):
            p = int(rng.integers(0, k)); e.append((p, k) if rng.random() < .5 else (k, p))
        return np.array(e, int), n + 1
    E = max(2, int(round(.7 * n)) + 1)
    while E * (E - 1) // 2 < n:
        E += 1
    seen = set(); e = [(i, i + 1) for i in rng.permutation(E - 1)[:min(n, E - 1)]]; seen.update(e)
    while len(e) < n:
        i = int(rng.integers(0, E - 1)); jn = min(E - 1, i + 1 + int(rng.geometric(.5)))
        if (i, jn) not in seen:
            seen.add((i, jn)); e.append((i, jn))
    e = np.array(e, int); used = np.unique(e); remap = -np.ones(E, int); remap[used] = np.arange(len(used))
    return remap[e], len(used)


def comparison_context(rng, n, family, n_names=1400, q_per=8):
    """-> dict: n, family, facts [n, 2] name ids (the first is older; statement order), q [Q, 2], label, kind, depth."""
    edges, E = _comparison_structure(rng, n, family)
    names = rng.choice(n_names, E + 2, replace=False); ent, spare = names[:E], names[E:]
    D = _distances(edges, E); yes = np.argwhere(D > 0); depths = np.unique(D[D > 0]); unrel = np.argwhere((D < 0) & (D.T < 0)); qs = []
    for _ in range(q_per // 2):
        d = int(rng.choice(depths)); cand = yes[D[yes[:, 0], yes[:, 1]] == d]; x, y = cand[rng.integers(len(cand))]; qs.append((ent[x], ent[y], 1, 0, d))
    for i in range(q_per - q_per // 2):
        kinds = [1] + ([2] if len(unrel) else []) + [3]; k = kinds[i % len(kinds)] if i < len(kinds) else int(rng.choice(kinds))
        if k == 1:
            x, y = yes[rng.integers(len(yes))]; qs.append((ent[y], ent[x], 0, 1, -1))
        elif k == 2:
            x, y = unrel[rng.integers(len(unrel))]; qs.append((ent[x], ent[y], 0, 2, -1))
        else:
            x = int(rng.integers(E)); s = spare[int(rng.integers(2))]; qs.append((ent[x], s, 0, 3, -1) if rng.random() < .5 else (s, ent[x], 0, 3, -1))
    qs = np.array(qs, int); order = rng.permutation(n)
    return dict(n=n, family=FAMILIES.index(family), facts=ent[edges][order].astype(np.int32), q=qs[:, :2].astype(np.int32),
                label=qs[:, 2].astype(np.int8), kind=qs[:, 3].astype(np.int8), depth=qs[:, 4].astype(np.int32))


def comparison_split(name):
    a = COMPARISON_SPLITS[name]; rng = np.random.default_rng(a['seed'])
    return [comparison_context(rng, n, FAMILIES[i % 3]) for n in a['sizes'] for i in range(a['per_size'])]


# ============================================================================================ 2. read-in (frozen model)
PROBES = (' Therefore, the parent is', ' Therefore, the child is')
LEAD = 'Here is what is known about a family.'
WORD = re.compile(r"'s|[A-Za-z]+|[^A-Za-z\s]")


class FrozenModel:
    """The frozen language model at a pinned revision. Input embeddings are taken in float16 (as the read-ins were made,
    on a GPU in float16), whatever the device."""

    def __init__(self, name, revision, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=torch.float16 if self.dev == 'cuda' else torch.float32).to(self.dev).eval()
        self.emb = self.model.get_input_embeddings().weight.detach().to(torch.float16).float().cpu().numpy()
        self.base = getattr(self.model, 'model', None) or self.model.transformer

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def last_logits(self, ids_list, bs=64):
        """-> next-token logits [len(ids_list), vocab] at the last token of each sequence (left padding, length-sorted batches)."""
        torch = self.torch
        order = np.argsort([len(i) for i in ids_list]); out = [None] * len(ids_list)
        for b0 in range(0, len(ids_list), bs):
            idx = order[b0:b0 + bs]; L = max(len(ids_list[i]) for i in idx)
            x = torch.full((len(idx), L), self.tok.pad_token_id or 0, device=self.dev); am = torch.zeros((len(idx), L), dtype=torch.long, device=self.dev)
            for r, i in enumerate(idx):
                x[r, L - len(ids_list[i]):] = torch.tensor(ids_list[i], device=self.dev); am[r, L - len(ids_list[i]):] = 1
            with torch.no_grad():
                lg = self.model.lm_head(self.base(input_ids=x, attention_mask=am, use_cache=False).last_hidden_state[:, -1]).float().cpu().numpy()
            for r, i in enumerate(idx):
                out[i] = lg[r]
        return out


class KinshipReader:
    """Reads kinship sentences with a frozen model. Nothing is trained.
    Who is named: the closed-class words (closed.json: the words common to many training sentences, found without labels)
    separate the names; a mention is a maximal run of other words, and its key is the sum of the model's input embeddings
    over its tokens. Which wording: the same sum over a sentence's closed-class tokens. The model's own reading: the
    sentence followed by ' Therefore, the parent is' and by ' Therefore, the child is'; for each, the logit of the
    first-named person's first token minus the second-named's; the reading is the sign of the difference of the two."""

    def __init__(self, fm, closed):
        self.fm = fm; self.closed_s = set(closed['statements']); self.closed_q = set(closed['questions']); self.cache = {}

    def toks(self, sentence):
        if sentence not in self.cache:
            e = self.fm.tok(' ' + sentence, add_special_tokens=False, return_offsets_mapping=True); self.cache[sentence] = (e['input_ids'], e['offset_mapping'])
        return self.cache[sentence]

    def words(self, sentence):
        """-> [(word, token indices)] of ' ' + sentence; a token belongs to the first word it overlaps."""
        ids, off = self.toks(sentence); out, t = [], 0
        for m_ in WORD.finditer(' ' + sentence):
            ts = []
            while t < len(off) and off[t][0] < m_.end():
                if off[t][1] > m_.start() or off[t][1] == off[t][0]:
                    ts.append(t)
                t += 1
            out.append((m_.group(), ts))
        return out

    def mentions(self, sentence, closed):
        runs, cur = [], []
        for w, ts in self.words(sentence):
            if w in closed:
                if cur:
                    runs.append(cur); cur = []
            else:
                cur += ts
        if cur:
            runs.append(cur)
        return runs

    def keys(self, sentence, closed):
        """-> keys [3, d] (first-named, second-named, wording), as stored (float16, read as float32), the two mentions' token
        indices, and whether exactly two mentions were found."""
        ids, _ = self.toks(sentence); runs = self.mentions(sentence, closed); ok = len(runs) == 2; runs = (runs + [[], []])[:2]
        cl = [t for w, ts in self.words(sentence) if w in closed for t in ts]; d = self.fm.emb.shape[1]
        s = lambda ts: self.fm.emb[[ids[t] for t in ts]].sum(0) if ts else np.zeros(d, np.float32)
        return np.stack([s(runs[0]), s(runs[1]), s(cl)]).astype(np.float16).astype(np.float32), runs, ok

    def readings(self, sentences, runs, bs=64):
        """-> reading [n] int8: 1 the model says the first-named person is the parent, 0 the second, -1 undecided (the two
        names have the same tokens). Names sharing first tokens are compared at the first token that differs."""
        tg = []
        for s, (r1, r2) in zip(sentences, runs):
            ids = self.toks(s)[0]; a = [ids[t] for t in r1]; b = [ids[t] for t in r2]; c = 0
            while c < min(len(a), len(b)) - 1 and a[c] == b[c]:
                c += 1
            tg.append((a[c], b[c], tuple(a[:c])) if a and b else (0, 0, ()))
        m = []
        for probe in PROBES:
            pr = self.fm.enc(probe); lg = self.fm.last_logits([self.toks(s)[0] + pr + list(t[2]) for s, t in zip(sentences, tg)], bs)
            m.append(np.array([float(l[t[0]] - l[t[1]]) for l, t in zip(lg, tg)], np.float32))
        same = np.array([t[0] == t[1] for t in tg])
        return np.where(same, -1, (m[0] - m[1] > 0).astype(np.int8)).astype(np.int8)

    def read(self, contexts, bs=64):
        """contexts: [(statements, questions)] as text. -> dict s_key [statements, 3, d], q_key [contexts, Q, 2, d],
        read [statements], off [contexts + 1], found (exactly two mentions in every statement and question)."""
        sk, qk, runs, sents, found = [], [], [], [], True
        for st, qs in contexts:
            for s in st:
                k, r, ok = self.keys(s, self.closed_s); sk.append(k); runs.append(r); sents.append(s); found &= ok
            q = []
            for s in qs:
                k, _, ok = self.keys(s, self.closed_q); q.append(k[:2]); found &= ok
            qk.append(np.stack(q))
        n = [len(st) for st, _ in contexts]
        return dict(s_key=np.stack(sk), q_key=np.stack(qk) if len({len(q) for q in qk}) == 1 else qk, read=self.readings(sents, runs, bs),
                    off=np.concatenate([[0], np.cumsum(n)]).astype(np.int64), found=bool(found))

    def alone(self, statements, question):
        """The frozen model answering by itself: True when ' Yes' beats ' No' after the whole context and the question."""
        ctx = self.fm.enc(LEAD) + [t for s in statements for t in self.toks(s)[0]] + self.fm.enc('\n\nQuestion: ' + question + '\nAnswer:')
        lg = self.fm.last_logits([ctx], 1)[0]
        return bool(lg[self.fm.enc(' Yes')[0]] > lg[self.fm.enc(' No')[0]])


class ComparisonReader:
    """Reads comparison facts with a frozen model: the input embeddings of the fact lines '* A is older than B.\\n' and of the
    question 'Q: Is X older than Y?\\nA:'. Names are single tokens of the model's vocabulary (names.json)."""
    FACT, QUEST = '* {} is older than {}.\n', 'Q: Is {} older than {}?\nA:'

    def __init__(self, fm, names):
        self.fm = fm; self.names = names['names']; self.TF, self.TQ = names['fact_tokens'], names['question_tokens']

    def read(self, facts, questions):
        """facts [(a, b)], questions [(x, y)] as names. -> X [Q, n + 1, T, d] float32 (item 0 is the question), tok, real."""
        e = self.fm.emb; T = max(self.TF, self.TQ); n = len(facts)
        F = [self.fm.enc(self.FACT.format(a, b)) for a, b in facts]; Qs = [self.fm.enc(self.QUEST.format(x, y)) for x, y in questions]
        assert all(len(f) == self.TF for f in F) and all(len(q) == self.TQ for q in Qs), 'every name must be one token, as in names.json'
        X = np.zeros((len(Qs), n + 1, T, e.shape[1]), np.float32); tok = np.zeros((len(Qs), n + 1, T), bool)
        for i, q in enumerate(Qs):
            X[i, 0, :self.TQ] = e[q]; tok[i, 0, :self.TQ] = True
            for k, f in enumerate(F):
                X[i, k + 1, :self.TF] = e[f]; tok[i, k + 1, :self.TF] = True
        return X, tok, np.ones((len(Qs), n + 1), bool)


# ====================================================================================================== 3. the block
def loops_for(n_items, c=3):
    """The loop rule: ceil(log2 of the number of items) + c."""
    return int(math.ceil(math.log2(max(n_items, 2)))) + c


class Layout:
    """The input planes of a pair (a, b) of mentions, for K statement classes, as patterns. Pattern = (relation, same
    person, a in the question, b in the question, same mention); relation 0: none; 1 + 2c + d: within a statement of class
    c (d = 0: a named first, b second); 2K + 1 + d: the question's X and Y. The last pattern is 'not a real pair'.
    links 'partner': relation 1 + 2c + d when a is named in a statement of class c (d = 0: named first) and the other
    person named in a's statement is b's person (one comparison of two keys); 2K + 1 + d the same for the question."""

    def __init__(self, K, links='within'):
        self.K = K; self.links = links; self.P = 2 * K + 7; R = 2 * K + 3; self.n_pat = R * 16 + 1; pat = np.zeros((self.n_pat, self.P), np.float32)
        for rel in range(R):
            for b in range(16):
                r = pat[rel * 16 + b]; r[self.P - 1] = 1; r[self.P - 2] = b & 1; r[self.P - 3] = b >> 1 & 1; r[self.P - 4] = b >> 2 & 1; r[0] = b >> 3 & 1
                if rel:
                    r[rel] = 1
        self.pat = pat

    def pid(self, sp, sent, slot, cls, real):
        """sp [N, N] same person; sent [N] statement index (-2: the question; -1: padding); slot [N] 0 named first, 1 second;
        cls [N] class -> pattern ids [N, N]."""
        N = len(sent); K = self.K; same = np.eye(N, dtype=bool); inq = sent == -2; pair = sent[:, None] == sent[None]
        if self.links == 'partner':
            partner = np.arange(N); partner[0::2] += 1; partner[1::2] -= 1; partner = np.where(real, np.minimum(partner, N - 1), np.arange(N)); pair = sp[partner] & real[:, None]
        rel = np.where(pair & ~same & (sent[:, None] >= 0), 1 + 2 * cls[:, None] + slot[:, None], 0)
        rel = np.where(pair & ~same & inq[:, None], 2 * K + 1 + slot[:, None], rel)
        p = rel * 16 + ((sp & ~same) * 8 + inq[:, None] * 4 + inq[None] * 2 + same)
        return np.where(real[:, None] & real[None], p, self.n_pat - 1).astype(np.int32)


def frame(n, N):
    """-> sent, slot, real for a question with n statements and N >= 2n + 2 mentions: item 0 is X, item 1 is Y, then
    each statement's first-named and second-named person."""
    sent = -np.ones(N, np.int64); slot = np.zeros(N, np.int64); real = np.zeros(N, bool); sent[:2] = -2; slot[1] = 1
    sent[2:2 * n + 2] = np.repeat(np.arange(n), 2); slot[2:2 * n + 2] = np.tile([0, 1], n); real[:2 * n + 2] = True
    return sent, slot, real


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def wording_classes(w_key, cal, open_new=False):
    """-> the wording class of every statement: the nearest of the training wordings (cosine of the centred closed-class
    keys). open_new: a statement that matches no class at 0.98 opens a new one (a wording never seen in training)."""
    W = _unit((np.asarray(w_key, np.float32) - cal['mu_w']) / cal['sd_w'])
    if not open_new:
        return (W @ cal['centres'].T).argmax(1), len(cal['centres'])
    C_ = cal['centres']; wc = np.full(len(W), -1)
    while True:
        todo = np.flatnonzero(wc < 0)
        if not len(todo):
            break
        S_ = W[todo] @ C_.T; hit = S_.max(1) >= .98; wc[todo[hit]] = S_[hit].argmax(1)
        if hit.all():
            break
        C_ = np.concatenate([C_, W[todo[~hit][0]][None]])
    return wc, len(C_)


def pooled_directions(w_key, read, cal):
    """-> class [statements] (0: the first-named is the parent, 1: the second) and the share per wording class. One
    direction per wording class, by majority of the model's readings over the class's sentences among those read
    together. No label and no answer is used."""
    wc, nc = wording_classes(w_key, cal, open_new=True); dec = read >= 0
    share = np.array([float((read[(wc == c) & dec] == 1).mean()) if ((wc == c) & dec).any() else .5 for c in range(nc)])
    return np.where(share[wc] > .5, 0, 1), wc, share


def statement_classes(cfg, cal, w_key, read):
    """-> the input class of every statement for this organ (cfg['planes']: 'pooled', 'wording', or 'direction': each
    sentence's own reading, class 0 when the model reads the first-named as the parent or is undecided)."""
    if cfg['planes'] == 'pooled':
        return pooled_directions(w_key, read, cal)[0]
    if cfg['planes'] == 'direction':
        return np.where(read == 0, 1, 0)
    assert cfg['planes'] == 'wording', cfg['planes']
    return wording_classes(w_key, cal)[0]


def kinship_item(s_key, q_key, cls, cal, lay, with_sp=False):
    """One question. s_key [n, 3, d] (float32), q_key [2, d], cls [n] -> pattern ids [N, N], real [N] (N = 2n + 2), and
    with_sp the 'same person' bits [N, N]."""
    n = len(s_key); N = 2 * n + 2; sent, slot, real = frame(n, N)
    P = np.zeros((N, s_key.shape[-1]), np.float32); P[:2] = q_key; P[2:] = s_key[:, :2].reshape(2 * n, -1); has = np.abs(P).sum(-1) > 0
    U = (P - cal['mu_p']) / cal['sd_p']; U = U / np.maximum(np.linalg.norm(U, axis=-1, keepdims=True), 1e-9) * has[:, None]
    sp = ((U @ U.T) > float(cal['theta_p'])) & has[:, None] & has[None]
    c = np.zeros(N, np.int64); c[2:] = np.repeat(cls, 2)
    return (lay.pid(sp, sent, slot, c, real), real) + ((sp,) if with_sp else ())


def kinship_run(p, pat, pid, real, K):
    """The block on a batch of questions of one size. pid [B, N, N], real [B, N] -> logits after each loop [K, B] (the
    answer at the pair (X, Y); yes when above 0) and whether the bits repeated [K, B]."""
    import jax
    import jax.numpy as j
    a = p['a']; fi = pat @ a['E'] + a['b'] + jax.nn.gelu(pat @ a['e1'] + a['b1']) @ a['e2']
    I = j.moveaxis(fi[pid], -1, 1); m = real.astype(I.dtype); mix = jax.nn.softplus(a['P']); L = j.zeros_like(I); out, same = [], []
    for _ in range(K):
        x = I + j.einsum('bcij,cd->bdij', L, mix)
        L2 = j.maximum(L, (j.matmul((x > 0) * m[:, None, None, :], (x > 0).astype(I.dtype)) > 0).astype(I.dtype))
        out.append(L2[:, :, 0, 1] @ jax.nn.softplus(p['w_out']) - jax.nn.softplus(p['b_out'])); same.append((L2 == L).all((1, 2, 3))); L = L2
    return j.stack(out), j.stack(same)


def comparison_bits(X, tok, real, ri):
    """The read-in of the comparison organ: one unit key per token (centred per token position, scaled, whitened, from the
    training lines; zero under a floor, as for the words of the sentence pattern) and one comparator per pair of token
    positions (planes): the cosine of two keys against a threshold found without labels. -> bits [B, M, N, N]."""
    import jax.numpy as j
    isq = (j.arange(X.shape[1]) == 0)[None, :, None, None]
    x = (j.where(isq, X - ri['qmu'], X - ri['fmu']) / ri['sd']) @ ri['Wh']
    nr = j.sqrt((x * x).sum(-1, keepdims=True)); k = j.where((nr > ri['floor']) & tok[..., None], x / j.maximum(nr, 1e-9), 0.)
    pl = np.asarray(ri['planes']); ki, kj = j.moveaxis(k[:, :, pl[:, 0]], 2, 1), j.moveaxis(k[:, :, pl[:, 1]], 2, 1)
    cos = j.matmul(ki, j.swapaxes(kj, 2, 3)); N = X.shape[1]
    pair = (real[:, None, :, None] & real[:, None, None, :]) & ~j.eye(N, dtype=bool)[None, None]
    return ((cos > ri['theta'][None, :, None, None]) & pair).astype(X.dtype)


def comparison_run(p, bits, real, K):
    """The block on the comparator bits: items are the question (item 0) and the facts; the answer at (question, question).
    -> logits after each loop [K, B] and whether the bits repeated [K, B]."""
    import jax
    import jax.numpy as j
    lin = lambda t, w, b=None: j.einsum('bcij,cd->bdij', t, w) + (0. if b is None else b[None, :, None, None])
    B, _, N, _ = bits.shape; r = real.astype(bits.dtype); isq = (j.arange(N) == 0).astype(bits.dtype)
    pair = r[:, :, None] * r[:, None, :]; one = lambda x: (x * pair)[:, None]
    F = j.concatenate([bits, one(isq[None, :, None] * j.ones((B, N, N))), one(isq[None, None, :] * j.ones((B, N, N))), one(j.eye(N)[None] * j.ones((B, 1, 1))), one(j.ones((B, N, N)))], 1)
    a = p['a']; m = real.astype(F.dtype); fixed = lin(F, a['E'], a['b']) + lin(jax.nn.gelu(lin(F, a['e1'], a['b1'])), a['e2'])
    s = j.zeros((B, p['w_out'].shape[0], N, N), F.dtype); out, same = [], []
    for _ in range(K):
        x = fixed + lin(s, jax.nn.softplus(a['P']))
        s2 = j.maximum(s, (j.matmul((x > 0) * m[:, None, None, :], (x > 0).astype(F.dtype)) > 0).astype(F.dtype))
        out.append(s2[:, :, 0, 0] @ jax.nn.softplus(p['w_out']) - jax.nn.softplus(p['b_out'])); same.append((s2 == s).all((1, 2, 3))); s = s2
    return j.stack(out), j.stack(same)



def comparison_relations(facts, q):
    """The true-relations control's input: for items (the question first, then the facts), each a pair of names (first,
    second), the four planes [4, n + 1, n + 1]: second(u) = first(v), first = first, second = second, first(u) = second(v)."""
    a = np.concatenate([facts[:, 0], q[:1]]); b = np.concatenate([facts[:, 1], q[1:]])
    r = np.stack([b[:, None] == a[None], a[:, None] == a[None], b[:, None] == b[None], a[:, None] == b[None]])
    return np.roll(np.roll(r, 1, 1), 1, 2).astype(np.float32)   # the question first


def comparison_warm_run(p, bits, real, K):
    """The warm control: a census block (pair state of width 128 that carries strengths: each loop counts witnesses in one
    half, keeps the best witness's score in the other, and adds a gated update and an MLP) on the same input planes.
    The answer: a linear read of the normalised state at (question, question). -> logits after each loop [K, B] and
    whether the state repeated [K, B]."""
    import jax
    import jax.numpy as j
    HI = 'highest'   # full float32 products, as the control was trained and read
    lin = lambda t, w, b=None: j.einsum('bcij,cd->bdij', t, w, precision=HI) + (0. if b is None else b[None, :, None, None])
    norm = lambda t: (t - t.mean(1, keepdims=True)) * jax.lax.rsqrt(t.var(1, keepdims=True) + 1e-6)
    B, _, N, _ = bits.shape; r = real.astype(bits.dtype); isq = (j.arange(N) == 0).astype(bits.dtype)
    pair = r[:, :, None] * r[:, None, :]; one = lambda x: (x * pair)[:, None]
    F = j.concatenate([bits, one(isq[None, :, None] * j.ones((B, N, N))), one(isq[None, None, :] * j.ones((B, N, N))), one(j.eye(N)[None] * j.ones((B, 1, 1))), one(j.ones((B, N, N)))], 1)
    m = real.astype(F.dtype); b = p['blk']; h = b['th_t'].shape[0]
    tt = b['th_t'] + jax.lax.stop_gradient(j.clip(b['th_t'], 0., 1.) - b['th_t'])
    mid = m[:, None, :, None] > 0

    def count_compare(xa, xc):   # more = [N_2i > N_2i+1], same = [N_2i = N_2i+1] over the real witnesses
        A, Cc = (xa > 0).astype(F.dtype), (xc > 0).astype(F.dtype)
        Nn = j.matmul(A * m[:, None, None, :], Cc * m[:, None, :, None], precision=HI); d = Nn[:, 0::2] - Nn[:, 1::2]
        return j.stack([(d > 0).astype(F.dtype), (d == 0).astype(F.dtype)], 2).reshape(Nn.shape)

    def best(xa, xc):            # exp(-min over real k of max(x_a, x_c) + t min(x_a, x_c)), x = -log sigmoid(score)
        ca, cc = jax.nn.softplus(-xa), jax.nn.softplus(-xc)

        def f(args):
            a, c, t1 = args; A, Cc = a[:, :, :, None], c[:, None, :, :]
            return j.where(mid, j.maximum(A, Cc) + t1 * j.minimum(A, Cc), j.inf).min(2)
        return j.exp(-jax.lax.map(f, (ca.swapaxes(0, 1), cc.swapaxes(0, 1), tt)).swapaxes(0, 1))
    s = lin(F, p['E'], p['eb']) + lin(jax.nn.gelu(lin(F, p['e1'], p['eb1'])), p['e2']); out, same = [], []
    for _ in range(K):
        t = norm(s); xa, xc = lin(t, b['a'], b['ba']), lin(t, b['b'], b['bb'])
        y = j.concatenate([count_compare(xa[:, :h], xc[:, :h]), best(xa[:, h:], xc[:, h:])], 1)
        z = s + jax.nn.sigmoid(lin(t, b['g'], b['bg'])) * lin(y, b['o'], b['bo'])
        s2 = z + lin(jax.nn.gelu(lin(norm(z), b['f1'], b['bf1'])), b['f2'], b['bf2'])
        out.append(j.dot(norm(s2)[:, :, 0, 0], p['w_out'], precision=HI) + p['b_out']); same.append((s2 == s).all((1, 2, 3))); s = s2
    return j.stack(out), j.stack(same)

_JIT = {}


def kinship_forward(p, pat, pid, real, K):
    """kinship_run compiled once per K (as the reads were made)."""
    import jax
    if ('kinship', K) not in _JIT:
        _JIT[('kinship', K)] = jax.jit(lambda p, pat, pid, real: kinship_run(p, pat, pid, real, K))
    return _JIT[('kinship', K)](p, pat, pid, real)


def comparison_forward(p, ri, X, tok, real, K):
    """The comparison read-in and block in one compiled function per K (as the reads were made). -> logits [K, B],
    repeated [K, B], bits [B, M, N, N]."""
    import jax
    key = ('comparison', K, id(ri))
    if key not in _JIT:
        def f(p, X, tok, real):
            bits = comparison_bits(X, tok, real, ri); lg, same = (comparison_warm_run if 'blk' in p else comparison_run)(p, bits, real, K)
            return lg, same, bits
        _JIT[key] = jax.jit(f)
    return _JIT[key](p, X, tok, real)



def comparison_bits_forward(p, bits, real, K):
    """The block on given input planes (the true-relations control), compiled once per K. -> logits [K, B], repeated [K, B]."""
    import jax
    if ('bits', K) not in _JIT:
        _JIT[('bits', K)] = jax.jit(lambda p, bits, real: (comparison_warm_run if 'blk' in p else comparison_run)(p, bits, real, K))
    return _JIT[('bits', K)](p, bits, real)

def batch_size(N):
    """Questions read at once for N items (as in the evaluation)."""
    return max(1, min(32, 4096 // N))


# ===================================================================================================== 4. load, answer
def _fetch(folder, files, repo=REPO, revision=None, token=None):
    """-> a local folder holding `files`: `folder` itself when it exists here, else that folder of the Hugging Face repo."""
    if Path(folder).exists():
        return Path(folder)
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(repo, f'{folder}/{f}', revision=revision, token=token)) for f in files][0].parent


def load(folder, repo=REPO, revision=None, token=None):
    """-> (params, config, read-in numbers, extra) of one organ: a local folder, or e.g. 'kinship/qwen2.5-7b/within/seed0'
    in the Hugging Face repo. extra: closed.json (kinship) or names.json (comparison)."""
    import jax.numpy as j
    from safetensors.numpy import load_file
    d = _fetch(folder, ['config.json', 'model.safetensors'], repo, revision, token); cfg = json.loads((d / 'config.json').read_text())
    params = {}
    for k, v in load_file(str(d / 'model.safetensors')).items():
        *path, leaf = k.split('/'); q = params
        for part in path:
            q = q.setdefault(part, {})
        q[leaf] = j.array(v)
    up = str(Path(folder).parent.parent if cfg['task'] == 'kinship' else Path(folder).parent)
    if cfg['task'] == 'kinship':
        ri_dir = _fetch(up, ['readin.npz', 'closed.json'], repo, revision, token); extra = json.loads((ri_dir / 'closed.json').read_text())
        z = np.load(ri_dir / 'readin.npz'); ri = {k: (float(z[k]) if k == 'theta_p' else z[k]) for k in z.files}
    elif cfg.get('input') == 'true_relations':   # the control without a frozen model: the true 'same name' planes
        ri, extra = None, None
    else:
        z = np.load(_fetch(folder, ['readin.npz'], repo, revision, token) / 'readin.npz')
        ri = {k: (z[k] if k == 'planes' else j.array(z[k])) for k in z.files}
        extra = json.loads((_fetch(up, ['names.json'], repo, revision, token) / 'names.json').read_text())
    return params, cfg, ri, extra


class Organ:
    """One trained organ with its frozen model. ask() reads text and answers; the frozen model is loaded on first use."""

    def __init__(self, folder=DEFAULT, repo=REPO, revision=None, token=None, device=None, parts=None):
        """folder: a local folder or a folder of the Hugging Face repo (default: DEFAULT); parts: (params, config, read-in,
        extra) instead."""
        self.params, self.cfg, self.ri, self.extra = parts if parts is not None else load(folder, repo, revision, token); self.device = device; self.reader = None
        if self.cfg['task'] == 'kinship':
            self.lay = Layout(self.cfg['classes'], self.cfg['layout']); self.pat = self.lay.pat

    def _reader(self):
        if self.reader is None:
            fm = FrozenModel(self.cfg['frozen_model']['name'], self.cfg['frozen_model']['revision'], self.device)
            self.reader = KinshipReader(fm, self.extra) if self.cfg['task'] == 'kinship' else ComparisonReader(fm, self.extra)
        return self.reader

    def answer_read(self, r, extra_loops=4):
        """Kinship: answers from a read-in (KinshipReader.read). The directions are pooled over every statement read
        together. -> one list of answers (True: yes) per context, read after K + extra_loops loops."""
        import jax.numpy as j
        cls = statement_classes(self.cfg, self.ri, r['s_key'][:, 2], r['read']); out = []; pat = j.array(self.pat)
        for ci in range(len(r['off']) - 1):
            o0, o1 = int(r['off'][ci]), int(r['off'][ci + 1]); n = o1 - o0; N = 2 * n + 2; K0 = loops_for(N); bs = batch_size(N); ans = []
            items = [kinship_item(r['s_key'][o0:o1], q, cls[o0:o1], self.ri, self.lay) for q in r['q_key'][ci]]
            for i in range(0, len(items), bs):
                part = items[i:i + bs]
                lg, _ = kinship_forward(self.params, pat, j.array(np.stack([x[0] for x in part])), j.array(np.stack([x[1] for x in part])), K0 + extra_loops)
                ans += [bool(v > 0) for v in np.asarray(lg[-1])]
            out.append(ans)
        return out

    def ask(self, contexts, extra_loops=4):
        """kinship: contexts [(statements, questions)] as text; comparison: [(facts [(a, b)], questions [(x, y)])] as names.
        -> one list of answers (True: yes) per context."""
        import jax.numpy as j
        if self.cfg.get('input') == 'true_relations':
            return self._ask_true(contexts, extra_loops)
        rd = self._reader()
        if self.cfg['task'] == 'kinship':
            r = rd.read(contexts)
            assert r['found'], 'a sentence without exactly two names; see closed.json for the words that are not names'
            return self.answer_read(r, extra_loops)
        out = []
        for facts, qs in contexts:
            X, tok, real = rd.read(facts, qs); N = len(facts) + 1; K0 = loops_for(N); bs = batch_size(N); ans = []
            for i in range(0, len(qs), bs):
                lg, _, _ = comparison_forward(self.params, self.ri, j.array(X[i:i + bs]), j.array(tok[i:i + bs]), j.array(real[i:i + bs]), K0 + extra_loops)
                ans += [bool(v > 0) for v in np.asarray(lg[-1])]
            out.append(ans)
        return out

    def _ask_true(self, contexts, extra_loops=4):
        """The true-relations control: the planes come from which names are equal, not from a model."""
        import jax.numpy as j
        out = []
        for facts, qs in contexts:
            ids = {}; f = np.array([[ids.setdefault(a, len(ids)), ids.setdefault(b, len(ids))] for a, b in facts], int)
            N = len(facts) + 1; K0 = loops_for(N); bs = batch_size(N); ans = []
            rel = np.stack([comparison_relations(f, np.array([ids.setdefault(x, len(ids)), ids.setdefault(y, len(ids))], int)) for x, y in qs])
            for i in range(0, len(qs), bs):
                lg, _ = comparison_bits_forward(self.params, j.array(rel[i:i + bs]), j.ones((len(rel[i:i + bs]), N), bool), K0 + extra_loops)
                ans += [bool(v > 0) for v in np.asarray(lg[-1])]
            out.append(ans)
        return out


def demo(folder, statements=64, seed=0):
    organ = Organ(folder); rng = np.random.default_rng(seed)
    if organ.cfg['task'] == 'kinship':
        c = kinship_context(rng, statements, 'pedigree'); st, qs = kinship_text(c)
        print('\n'.join(st[:6]), f'\n... ({len(st)} statements)\n')
        got = organ.ask([(st, qs)])[0]
        for q, g, lab, d in zip(qs, got, c['label'], c['depth']):
            print(f'{q:<60} organ: {"yes" if g else "no ":<4} truth: {"yes" if lab else "no"}' + (f'  (a chain of {d})' if lab else ''))
    else:
        names = organ.extra['names']; c = comparison_context(rng, statements, 'mixed')
        facts = [(names[a], names[b]) for a, b in c['facts']]; qs = [(names[x], names[y]) for x, y in c['q']]
        got = organ.ask([(facts, qs)])[0]
        for (x, y), g, lab in zip(qs, got, c['label']):
            print(f'Is {x} older than {y}?'.ljust(40), 'organ:', 'yes' if g else 'no ', ' truth:', 'yes' if lab else 'no')


if __name__ == '__main__':
    if sys.argv[1:2] == ['demo']:
        args = sys.argv[2:]; n = int(args[args.index('--statements') + 1]) if '--statements' in args else 64
        demo(args[0] if args and not args[0].startswith('--') else DEFAULT, n)
    else:
        print(__doc__)
