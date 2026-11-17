"""Cold attention beside a frozen language model, on CLUTRR (Sinha and others, 2019).

A CLUTRR story states a chain of k family relations between people whose names are in brackets; the question is the
relation between two of them, one of 20 words. A frozen Qwen2.5-7B-Instruct reads, for every pair of people named close
together, which relation the story states between them. A block of cold attention chains those readings with a rule
table it learned from answers, and answers. Nothing in the language model is trained or changed.

Three organs, for each of two CLUTRR splits (gen_train234_test2to10, gen_train23_test2to10) and three seeds:
  graph      the block on the benchmark's own graph of each story (no language model). 20 bits per ordered pair of people,
             one per relation; a learned rule (a, b -> c) turns bit c on at (i, k) when a is on at (i, j) and b at (j, k)
             for some person j; bits stay on. The table it learned is exactly CLUTRR's 62 rules on every split and seed.
  zero_shot  the model's own zero-shot reading of each pair (a word and its probability); the cost read: each read
             relation costs -log of its probability, a derived relation costs its cheapest derivation under the table the
             block learned from that reading, and the answer is the cheapest relation at the question's pair. A tie, or
             nothing derived, takes the bit block's answer.
  trained    a read-in, one linear map from the model's last-layer state to 21 scores (20 relations, nothing stated),
             trained from the stories' answers through the cost read; then the same cost read, with the zero-shot stage's
             table (the teacher table) and every probability kept.

Sections: 1 the data and the candidate pairs; 2 the frozen model (torch, transformers; nothing is trained); 3 the bit
block and the cost read (numpy); 4 loading and answering.

  python clutrr.py demo [trained | zero_shot | graph] [gen_train234_test2to10] [seed]   (a GPU is advised)

Code: Apache 2.0. The CLUTRR data is CC BY-NC 4.0; it is downloaded from its own repository, not shipped here.
"""
import ast
import csv
import hashlib
import io
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = 'mohsenvand/cold-attention-clutrr'
MODEL = ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28')
RELS = ['aunt', 'brother', 'daughter', 'daughter-in-law', 'father', 'father-in-law', 'granddaughter', 'grandfather', 'grandmother',
        'grandson', 'husband', 'mother', 'mother-in-law', 'nephew', 'niece', 'sister', 'son', 'son-in-law', 'uncle', 'wife']
SPLITS = ('gen_train234_test2to10', 'gen_train23_test2to10')

# ============================================================================================= 1. data, candidate pairs
DATA_URL = 'https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/{split}/{part}.csv'   # what the Hub's CLUTRR/v1 loads
SHA256 = {('gen_train234_test2to10', 'test'): 'e139a9ebeece6901', ('gen_train234_test2to10', 'train'): 'a6123f6e305cc528',
          ('gen_train234_test2to10', 'validation'): '414390d3322b9fd2', ('gen_train23_test2to10', 'test'): 'd49c7cbe4f575b48',
          ('gen_train23_test2to10', 'train'): '38963d01a20789ac', ('gen_train23_test2to10', 'validation'): '418d712579e6c9bb'}


def load_clutrr(split, part, folder=None):
    """-> the rows of one CLUTRR file (dicts: id, story, query, target_text, task_name, story_edges, edge_types,
    query_edge, ...), checked against the sha256 of the copy the results were read on. folder: a local copy
    (<folder>/<split>/<part>.csv); else downloaded."""
    csv.field_size_limit(10 ** 9)
    if folder:
        raw = (Path(folder) / split / f'{part}.csv').read_bytes()
    else:
        with urllib.request.urlopen(DATA_URL.format(split=split, part=part)) as r:
            raw = r.read()
    h = hashlib.sha256(raw).hexdigest()[:16]
    assert h == SHA256[(split, part)], f'{split}/{part}.csv is not the file the results were read on ({h})'
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8'), newline='')))


NAME = re.compile(r'\[([^\]]+)\]')


def story_items(story):
    """-> names (the bracketed names, in order of first appearance) and candidates [(a, b, context)]: every ordered pair of
    different people whose names are at most two apart in the story's sequence of names, with the text from the start of
    the sentence of the pair's first close mention to the end of the sentence of its last."""
    ment = [(m.group(1), m.start(), m.end()) for m in NAME.finditer(story)]; names = list(dict.fromkeys(n for n, _, _ in ment)); idx = {n: i for i, n in enumerate(names)}
    cuts = [0] + [m.end() for m in re.finditer(r'[.!?]\s+', story)] + [len(story)]
    sent_of = lambda pos: max(i for i, c in enumerate(cuts[:-1]) if c <= pos)
    best = {}
    for t, (n1, s1, e1) in enumerate(ment):
        for u in (t + 1, t + 2):
            if u < len(ment) and ment[u][0] != n1:
                a, b = idx[n1], idx[ment[u][0]]; gap = ment[u][1] - e1
                for key in ((a, b), (b, a)):
                    best[key] = (gap, s1, ment[u][2]) if key not in best else (min(gap, best[key][0]), min(s1, best[key][1]), max(ment[u][2], best[key][2]))
    return names, [(a, b, story[cuts[sent_of(lo)]:cuts[sent_of(hi - 1) + 1]].strip()) for (a, b), (_, lo, hi) in sorted(best.items())]


def question(row, names):
    qa, qb = ast.literal_eval(row['query']); return [names.index(qa), names.index(qb)]


def graph_story(row):
    """The benchmark's own graph of a story: -> n people, facts [(i, j, relation)] (j is i's relation), question (i, j)."""
    se, et = ast.literal_eval(row['story_edges']), ast.literal_eval(row['edge_types']); q = tuple(ast.literal_eval(row['query_edge']))
    return max(max(a, b) for a, b in se + [q]) + 1, [(a, b, t) for (a, b), t in zip(se, et)], q


# ======================================================================================== 2. the frozen model (no training)
INSTR = ("You are given a short text about a family and a sentence to complete with ONE word. The sentence has the form '[B] is [A]'s ___': say what B is to A. "
         "Example: for the text '[Tom] went fishing with his daughter [Ann].', the sentence '[Ann] is [Tom]'s ___' is completed with 'daughter', and "
         "'[Tom] is [Ann]'s ___' is completed with 'father'. Use he, she, his and her in the text to tell whether a person is male or female. "
         "If the text directly states how the two people are related, give the family word. If the text does not directly state how these two people are related, "
         "complete the sentence with the word: unknown. Do not work the relation out from other relations.")   # one made-up example of direction; not from the benchmark


class Reader:
    """Qwen2.5-7B-Instruct at the pinned revision, float16 on a GPU. For a pair (A, B) the prompt is the chat template with
    INSTR as the system message and 'Text: <context>\\n\\nSentence to complete: [B] is [A]'s ___' as the user's, and the reply
    begun with "[B] is [A]'s". zero_shot(): the model's next word among the 20 relation words and 'unknown' (a word of
    several tokens followed token by token), with the probability of its first token. states(): the last-layer state there."""

    def __init__(self, device=None, name=MODEL[0], revision=MODEL[1]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch; self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=torch.float16 if self.dev == 'cuda' else torch.float32).to(self.dev).eval()
        self.base = getattr(self.model, 'model', None) or self.model.transformer
        self.words = RELS + ['unknown']; self.seqs = [self.enc(' ' + r) for r in self.words]; self.firsts = sorted({s[0] for s in self.seqs})
        self.group = {}
        for r, s in enumerate(self.seqs):
            self.group.setdefault(s[0], []).append(r)
        self.keep = sorted({t for s in self.seqs for t in s}); self.col = {t: i for i, t in enumerate(self.keep)}

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def prompt(self, ctx, a, b):
        tail = f"[{b}] is [{a}]'s"; msg = [{'role': 'system', 'content': INSTR}, {'role': 'user', 'content': f'Text: {ctx}\n\nSentence to complete: {tail} ___'}]
        return self.enc(self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + tail)

    def _last(self, ids_list, bs, f):
        torch = self.torch; order = np.argsort([len(i) for i in ids_list]); res = [None] * len(ids_list)
        for b0 in range(0, len(ids_list), bs):
            idx = order[b0:b0 + bs]; L = max(len(ids_list[i]) for i in idx)
            x = torch.full((len(idx), L), self.tok.pad_token_id or 0, device=self.dev); am = torch.zeros((len(idx), L), dtype=torch.long, device=self.dev)
            for r, i in enumerate(idx):
                x[r, L - len(ids_list[i]):] = torch.tensor(ids_list[i], device=self.dev); am[r, L - len(ids_list[i]):] = 1
            with torch.no_grad():
                h = self.base(input_ids=x, attention_mask=am, use_cache=False).last_hidden_state[:, -1]; v = f(h)
            for r, i in enumerate(idx):
                res[i] = v[r]
        return res

    def next_probs(self, ids_list, bs=64):
        keep = self.torch.tensor(self.keep, device=self.dev)
        return self._last(ids_list, bs, lambda h: self.torch.softmax(self.model.lm_head(h).float(), -1)[:, keep].cpu())

    def states(self, prompts, bs=128):
        """-> the last-layer state at the end of each prompt [n, width] (float16)."""
        return np.stack(self._last(prompts, bs, lambda h: h.float().cpu().numpy().astype(np.float16)))

    def zero_shot(self, prompts, bs=64):
        """-> per prompt [relation index (20: unknown), p of its first token, mass on the relation words' first tokens,
        runner-up's p], rounded to 4 decimals."""
        torch = self.torch; P = self.next_probs(prompts, bs); ft = torch.tensor([self.col[t] for t in self.firsts]); out = []; again = []
        for i, pr in enumerate(P):
            v = pr[ft]; top = int(v.argmax()); g = self.group[self.firsts[top]]
            out.append([g[0], float(v[top]), float(v.sum()), float(torch.topk(v, 2).values[1])])
            if len(g) > 1:
                again.append((i, g, 1, prompts[i] + [self.firsts[top]]))
        while again:   # words that share their tokens so far: follow the model one more token
            Q = self.next_probs([a[3] for a in again], bs); nxt = []
            for (i, g, depth, ids), pr in zip(again, Q):
                ends = [r for r in g if len(self.seqs[r]) == depth]; cont = {}
                for r in g:
                    if len(self.seqs[r]) > depth:
                        cont.setdefault(self.seqs[r][depth], []).append(r)
                pc = {t: float(pr[self.col[t]]) for t in cont}; p_end = 1. - sum(pc.values()); t_best = max(pc, key=pc.get) if pc else None
                if ends and (t_best is None or p_end >= pc[t_best]):
                    out[i][0] = ends[0]
                else:
                    out[i][0] = cont[t_best][0]
                    if len(cont[t_best]) > 1:
                        nxt.append((i, cont[t_best], depth + 1, ids + [t_best]))
            again = nxt
        return [[r, round(p, 4), round(m, 4), round(s, 4)] for r, p, m, s in out]


def readin_probs(H, ri):
    """The read-in: softmax(((H - mu) / sd) @ U + b) -> [pairs, 21] (index 20: nothing stated), kept in float16 as the
    results were."""
    X = (np.asarray(H, np.float32) - ri['mu']) / ri['sd']; z = X @ ri['U'] + ri['b']; z = np.exp(z - z.max(-1, keepdims=True))
    return (z / z.sum(-1, keepdims=True)).astype(np.float16)


# =========================================================================================== 3. the bit block, the cost read
def loops_for(n, c=3):
    return int(math.ceil(math.log2(max(n, 2)))) + c


def bit_block(p, n, facts, q, extra=4, rest=16):
    """The block with hard bits on one story: n people, facts [(i, j, relation index)], question (i, j). -> (answer index,
    read after ceil(log2 n) + 3 + extra loops; the relations whose bit is on at the question's pair after + rest loops)."""
    W = np.asarray(p['W']) > 0; R = W.shape[0]; I = np.zeros((R, n, n), np.float32)
    for a, c, t in facts:
        if a != c and a < n and c < n:
            I[t, a, c] = 1
    L = np.zeros_like(I); Wf = W.astype(np.float32); g = np.asarray(p['g'], np.float32); sp = np.logaddexp(g, np.float32(0)).astype(np.float32); ans = None
    for it in range(loops_for(n) + max(extra, rest)):
        T = np.maximum(I, L); J = (np.einsum('aij,cjk->acik', T, T) > 0).astype(np.float32)
        L = np.maximum(L, (np.einsum('acik,acr->rik', J, Wf) > 0).astype(np.float32))
        if it + 1 == loops_for(n) + extra:
            ans = int(np.argmax(L[:, q[0], q[1]] * sp + np.asarray(p['bias'], np.float32)))
    return ans, [RELS[r] for r in np.flatnonzero(L[:, q[0], q[1]] > 0)]


def table_of(p):
    """-> the rules (a, b, c) whose weight is on, as relation names."""
    W = np.asarray(p['W']) > 0; return sorted((RELS[a], RELS[b], RELS[c]) for a, b, c in zip(*np.nonzero(W)))


def one_rule_per_pair(rules):
    """The zero-shot stage's cost read keeps one rule per pair of relations (a, b): where the table has two, the later in
    the sorted table. Its reported numbers were made so; it matters for one table (gen_train23_test2to10, seed 1)."""
    d = {}
    for a, b, c in rules:
        d[(a, b)] = c
    return [(a, b, c) for (a, b), c in d.items()]


def cost_read(n, costs, q, rules, fallback):
    """The cost read on one story. costs: {(i, j, relation index): cost}; rules: [(a, b, c)] indices; fallback: the
    answer index used on a tie (when it is among the tied) or when nothing is derived. -> (answer index, how,
    {relation: cost} of the cheapest four derived at the question's pair)."""
    I = np.full((20, n, n), np.inf)
    for (a, b, r), c in costs.items():
        I[r, a, b] = min(I[r, a, b], c)
    ra, rb, rc = (np.array([t[i] for t in rules], int) for i in range(3)); D = np.full((20, n, n), np.inf)
    for _ in range(loops_for(n) + 4):
        T = np.minimum(I, D); cand = (T[ra][:, :, :, None] + T[rb][:, None, :, :]).min(2); Dn = D.copy(); np.minimum.at(Dn, rc, cand)
        if np.array_equal(Dn, D):
            break
        D = Dn
    at = D[:, q[0], q[1]]; top = {RELS[r]: round(float(at[r]), 4) for r in np.argsort(at, kind='stable')[:4] if np.isfinite(at[r])}
    if not np.isfinite(at).any():
        return fallback, 'nothing derived', top
    best = np.flatnonzero(at <= at.min() + 1e-9)
    if len(best) == 1:
        return int(best[0]), 'cheapest', top
    return (fallback, 'tie', top) if fallback in best else (int(best[0]), 'tie', top)


def costs_from_reading(cands, threshold):
    """Zero-shot reading [[a, b, relation index, p, ...]] -> {(a, b, r): -log p} for every relation read (not 'unknown')
    with p >= threshold."""
    out = {}
    for a, b, r, p, *_ in cands:
        if r < 20 and p >= threshold:
            out[(a, b, r)] = min(out.get((a, b, r), np.inf), -math.log(max(1e-6, min(1., p))))
    return out


def costs_from_probs(pairs, P, threshold=0.):
    """Read-in probabilities [pairs, 21] -> {(a, b, r): -log p} for the 20 relations with p >= max(threshold, 1e-6)."""
    P = np.asarray(P, np.float16).astype(np.float64)[:, :20]; out = {}
    for (a, b), pc in zip(pairs, P):
        for r in np.flatnonzero(pc >= max(threshold, 1e-6)):
            out[(a, b, int(r))] = float(-np.log(np.clip(pc[r], 1e-6, 1.)))
    return out


def facts_from_reading(cands, threshold):
    return [(a, b, r) for a, b, r, p, *_ in cands if r < 20 and p >= threshold]


def facts_from_probs(pairs, P, threshold):
    """The bit block's facts from read-in probabilities: every relation with p >= 0.02 whose p, rounded to 4 decimals,
    is at least the threshold."""
    P = np.asarray(P, np.float16).astype(np.float32); out = []
    for (a, b), pc in zip(pairs, P):
        for r in np.flatnonzero(pc[:20] >= .02):
            if round(float(pc[r]), 4) >= threshold:
                out.append((a, b, int(r)))
    return out


# ===================================================================================================== 4. load, answer
def _fetch(folder, files, repo=REPO, revision=None):
    if Path(folder).exists():
        return Path(folder)
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(repo, f'{folder}/{f}', revision=revision)) for f in files][0].parent


def _params(path):
    from safetensors.numpy import load_file
    return load_file(str(path))


class Organ:
    """One organ: kind 'graph', 'zero_shot' or 'trained'; split; seed. root: a local copy of the repository (else files are
    fetched from the Hugging Face repo). answer() takes CLUTRR rows (load_clutrr) and returns, per story, the answer and
    how it was reached."""

    def __init__(self, kind='trained', split=SPLITS[0], seed=0, root=None, device=None, reader=None):
        self.kind, self.split, self.seed, self.device, self._reader = kind, split, seed, device, reader
        base = (Path(root) / ('graph' if kind == 'graph' else 'text') / split) if root else f"{'graph' if kind == 'graph' else 'text'}/{split}"
        if kind == 'graph':
            d = _fetch(f'{base}/seed{seed}', ['config.json', 'model.safetensors']); self.block = _params(d / 'model.safetensors'); self.cfg = json.loads((d / 'config.json').read_text())
            return
        z = _fetch(f'{base}/zero_shot/seed{seed}', ['config.json', 'model.safetensors']); self.zs_block = _params(z / 'model.safetensors'); self.zs_cfg = json.loads((z / 'config.json').read_text())
        self.teacher = [tuple(RELS.index(x) for x in t) for t in table_of(self.zs_block)]   # the zero-shot stage's table: the teacher table
        self.teacher_zs = one_rule_per_pair(self.teacher)   # as the zero-shot stage's cost read used it
        if kind == 'trained':
            t = _fetch(f'{base}/trained/seed{seed}', ['config.json', 'readin.npz', 'fallback.safetensors']); self.cfg = json.loads((t / 'config.json').read_text())
            ri = np.load(t / 'readin.npz'); self.ri = {k: ri[k] for k in ri.files}; self.fallback = _params(t / 'fallback.safetensors')
        else:
            self.cfg = self.zs_cfg

    def reader(self):
        if self._reader is None:
            self._reader = Reader(self.device)
        return self._reader

    def read(self, rows):
        """-> per story: names, question, candidate pairs, and the model's reading of them (zero-shot words, or read-in
        probabilities)."""
        rd = self.reader(); items = []; prompts = []
        for row in rows:
            names, cands = story_items(row['story']); items.append(dict(names=names, q=question(row, names), pairs=[(a, b) for a, b, _ in cands], first=len(prompts)))
            prompts += [rd.prompt(ctx, names[a], names[b]) for a, b, ctx in cands]
        if self.kind == 'trained':
            P = readin_probs(rd.states(prompts), self.ri)
            for it in items:
                it['probs'] = P[it['first']:it['first'] + len(it['pairs'])]
        else:
            Z = rd.zero_shot(prompts)
            for it in items:
                it['cands'] = [[a, b] + Z[it['first'] + c] for c, (a, b) in enumerate(it['pairs'])]
        return items

    def answer_items(self, items):
        out = []
        for it in items:
            n = len(it['names']); q = it['q']
            if self.kind == 'trained':
                fb, _ = bit_block(self.fallback, n, facts_from_probs(it['pairs'], it['probs'], self.cfg['bit_threshold']), q)
                a, how, top = cost_read(n, costs_from_probs(it['pairs'], it['probs'], self.cfg['cost_threshold']), q, self.teacher, fb)
            else:
                fb, _ = bit_block(self.zs_block, n, facts_from_reading(it['cands'], self.zs_cfg['bit_threshold']), q)
                a, how, top = cost_read(n, costs_from_reading(it['cands'], self.zs_cfg['cost_threshold']), q, self.teacher_zs, fb)
            out.append(dict(answer=RELS[a], how=how, bits_answer=RELS[fb], costs=top))
        return out

    def answer(self, rows):
        if self.kind == 'graph':
            out = []
            for row in rows:
                n, facts, q = graph_story(row); a, on = bit_block(self.block, n, [(i, j, RELS.index(t)) for i, j, t in facts], q); out.append(dict(answer=RELS[a], bits_on=on))
            return out
        return self.answer_items(self.read(rows))


def demo(kind='trained', split=SPLITS[0], seed=0, stories=8):
    rows = load_clutrr(split, 'test'); rows = [r for r in rows if int(r['task_name'].split('.')[1]) == 10][:stories]
    organ = Organ(kind, split, seed); got = organ.answer(rows)
    for r, g in zip(rows, got):
        print(r['story'][:160].replace('\n', ' '), '...'); print('   question', r['query'], '| answer', g['answer'], '| truth', r['target_text'], '\n')
    print(sum(g['answer'] == r['target_text'] for g, r in zip(got, rows)), 'of', len(rows), 'right (stories of 10 relations)')


if __name__ == '__main__':
    if sys.argv[1:2] == ['demo']:
        a = sys.argv[2:]; demo(a[0] if a else 'trained', a[1] if len(a) > 1 else SPLITS[0], int(a[2]) if len(a) > 2 else 0)
    else:
        print(__doc__)
