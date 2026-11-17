"""The paper's numbers, each beside the value the paper prints: recomputed from the released models and published sources,
or read from the files of measured values in data/measured/.

  python numbers/numbers.py                  # reads the small JSON files of the Hugging Face repositories (downloaded once)
  python numbers/numbers.py --local DIR      # a local copy of the repositories: DIR/<folder>/..., folders as in REPOS
  python numbers/numbers.py --recomputed     # only the recomputed numbers

Every row is: the number's key, the paper's value, the value here, its source, and whether the two match.
  model      recomputed from the scores the released models record (each model's config.json / index.json). Every released
             model was read again with this repository's code and gives exactly its recorded predictions (REPRODUCE.md), so
             these scores are the models' own; the averages, ranges, counts and comparisons are recomputed here.
  published  recomputed from other papers' numbers (data/published.json, with their sources).
  read       read from data/measured/<family>.json: a measurement that is not a stored model score (reads on extra files,
             training logs, analyses, costs). Each file says what was measured; its values are the measured values as the
             paper prints them, so they are shown, not checked.
The paper's own values are in data/paper.json (one entry per number the paper prints, under the keys used here).
"""
import os
import sys
# This file shares its name with the standard library's numbers module; keep its folder off the import path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != os.path.dirname(os.path.abspath(__file__))]
import argparse  # noqa: E402
import json  # noqa: E402
import statistics as st  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
REPOS = {'models': 'mohsenvand/cold-attention-clrs30', 'shared': 'mohsenvand/cold-attention-clrs30-shared',
         'variants': 'mohsenvand/cold-attention-clrs30-variants', 'eleven': 'mohsenvand/cold-attention-clrs30-eleven',
         'scc-committee': 'mohsenvand/cold-attention-scc-committee', 'max_retrieval': 'mohsenvand/cold-attention-max-retrieval',
         'csp': 'mohsenvand/cold-attention-csp', 'organ': 'mohsenvand/cold-attention-organ', 'clutrr': 'mohsenvand/cold-attention-clutrr'}

# the paper's task order and names
FAMILY = [
    ('Graphs', ['bfs', 'dfs', 'topological_sort', 'articulation_points', 'bridges', 'strongly_connected_components']),
    ('Paths and trees', ['bellman_ford', 'dijkstra', 'dag_shortest_paths', 'floyd_warshall', 'mst_kruskal', 'mst_prim']),
    ('Order', ['insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 'minimum', 'binary_search', 'quickselect']),
    ('Greedy', ['activity_selector', 'task_scheduling']),
    ('Dynamic programming', ['find_maximum_subarray_kadane', 'lcs_length', 'matrix_chain_order', 'optimal_bst']),
    ('Strings and geometry', ['kmp_matcher', 'naive_string_matcher', 'graham_scan', 'jarvis_march', 'segments_intersect'])]
TASKS = [t for _, ts in FAMILY for t in ts]
PRETTY = {'activity_selector': 'Activity selector', 'articulation_points': 'Articulation points', 'bellman_ford': 'Bellman–Ford',
          'bfs': 'Breadth-first search', 'binary_search': 'Binary search', 'bridges': 'Bridges', 'bubble_sort': 'Bubble sort',
          'dag_shortest_paths': 'DAG shortest paths', 'dfs': 'Depth-first search', 'dijkstra': 'Dijkstra',
          'find_maximum_subarray_kadane': 'Maximum subarray (Kadane)', 'floyd_warshall': 'Floyd–Warshall', 'graham_scan': 'Graham scan',
          'heapsort': 'Heapsort', 'insertion_sort': 'Insertion sort', 'jarvis_march': 'Jarvis march', 'kmp_matcher': 'KMP matcher',
          'lcs_length': 'LCS length', 'matrix_chain_order': 'Matrix chain order', 'minimum': 'Minimum', 'mst_kruskal': 'Kruskal',
          'mst_prim': 'Prim', 'naive_string_matcher': 'Naive string matcher', 'optimal_bst': 'Optimal BST', 'quickselect': 'Quickselect',
          'quicksort': 'Quicksort', 'segments_intersect': 'Segments intersect', 'strongly_connected_components': 'Strong components',
          'task_scheduling': 'Task scheduling', 'topological_sort': 'Topological sort'}
key = lambda t: t.replace('_', '-')

N, SRC = {}, {}       # key -> printed value; key -> 'model' | 'published' | 'data'


def put(k, v, fmt=None, src='model'):
    if fmt == 'pct':
        v = f'{v:.2f}'
    elif fmt == 'pct1':
        v = f'{v:.1f}'
    elif fmt == 'int':
        v = f'{int(round(v)):,}'
    N[k], SRC[k] = str(v), src


def wtl(ours, theirs, tasks=TASKS):
    w = sum(ours[t] > theirs[t] + 1e-9 for t in tasks)
    ti = sum(abs(ours[t] - theirs[t]) <= 1e-9 for t in tasks)
    return w, ti, len(tasks) - w - ti


def put_wtl(prefix, ours, theirs, src='model'):
    for x, v in zip('wtl', wtl(ours, theirs)):
        put(f'{prefix}.{x}', v, src=src)


class Repos:
    """The repositories' JSON files, from a local copy or from Hugging Face (only *.json is downloaded)."""

    def __init__(self, local=None):
        self.local, self.paths = Path(local) if local else None, {}

    def root(self, name):
        if name not in self.paths:
            if self.local:
                self.paths[name] = self.local / name
            else:
                from huggingface_hub import snapshot_download
                self.paths[name] = Path(snapshot_download(REPOS[name], allow_patterns=['*.json']))
        return self.paths[name]

    def json(self, name, rel):
        return json.loads((self.root(name) / rel).read_text())


# ------------------------------------------------------------------------------------------------ published baselines
def published():
    P = json.loads((DATA / 'published.json').read_text())
    fn = {t: P['floydnet'][t]['full'] for t in TASKS}
    fnnh = {t: P['floydnet'][t]['no_hints'] for t in TASKS}
    oth = {t: P['best_other'][t]['score'] for t in TASKS}
    env = {t: max(fn[t], fnnh[t], oth[t]) for t in TASKS}
    for t in TASKS:
        put(f'fn.{key(t)}', fn[t], 'pct', 'published'); put(f'fnnh.{key(t)}', fnnh[t], 'pct', 'published')
        put(f'oth.{key(t)}', oth[t], 'pct', 'published'); put(f'othname.{key(t)}', P['best_other'][t]['model'], src='published')
    put('fn.mean', st.mean(fn.values()), 'pct', 'published'); put('fn.mean1', st.mean(fn.values()), 'pct1', 'published')
    put('fnnh.mean', st.mean(fnnh.values()), 'pct', 'published')
    put('fn.hinted.n', sum(fn[t] != fnnh[t] for t in TASKS), src='published')
    put('fn.hinted.lower', ', '.join(PRETTY[t] for t in TASKS if fnnh[t] > fn[t]), src='published')
    put('fnbest.mean', st.mean(max(fn[t], fnnh[t]) for t in TASKS), 'pct', 'published')
    put('oth.mean', st.mean(oth.values()), 'pct', 'published')
    put('env.mean', st.mean(env.values()), 'pct', 'published'); put('env2.mean', st.mean(env.values()), 'pct', 'published')
    dp = {int(d): v for d, v in P['floydnet_params_millions_by_depth'].items()}
    put('fn.params.min', f'{dp[min(dp)]:.1f}', src='published'); put('fn.params.max', f'{dp[max(dp)]:.0f}', src='published')
    put('fn.params.32', f'{dp[32]:.1f}', src='published')
    put('fn.depth.min', min(dp), src='published'); put('fn.depth.max', max(dp), src='published')
    return dict(fn=fn, fnnh=fnnh, oth=oth, env=env, fn_params=(dp[min(dp)] * 1e6, dp[max(dp)] * 1e6))


# ------------------------------------------------------------------------------------------------ one block per task
def per_task(R, pub):
    fn, fnnh, env = pub['fn'], pub['fnnh'], pub['env']
    idx = R.json('models', 'index.json')
    by = {}
    for m in idx:
        by.setdefault((m['block'], m['task']), {})[m['seed']] = m
    # the general block: three seeds per task, read after 8 loops
    seeds = {t: [by[('census', t)][s]['scores']['test'] for s in (0, 1, 2)] for t in TASKS}
    gen = {t: st.mean(v) for t, v in seeds.items()}
    per_seed = [st.mean(seeds[t][s] for t in TASKS) for s in (0, 1, 2)]
    put('gen.mean', st.mean(gen.values()), 'pct'); put('gen.mean1', st.mean(gen.values()), 'pct1')
    for s, v in enumerate(per_seed):
        put(f'gen.seed{s}', v, 'pct')
    put('gen.seed.lo', min(per_seed), 'pct'); put('gen.seed.hi', max(per_seed), 'pct')
    put('gen.seeds.above.fn', sum(v > st.mean(fn.values()) for v in per_seed))
    for t in TASKS:
        put(f'gen.{key(t)}', gen[t], 'pct'); put(f'gen.{key(t)}.lo', min(seeds[t]), 'pct'); put(f'gen.{key(t)}.hi', max(seeds[t]), 'pct')
        put(f'gen.{key(t)}.sd', st.pstdev(seeds[t]), 'pct')
    for nm, ref in (('fn', fn), ('fnnh', fnnh), ('env', env)):
        put_wtl(f'gen.vs.{nm}', gen, ref)
        put(f'gen.minus.{nm}', st.mean(gen.values()) - st.mean(ref.values()), 'pct')
    put('gen.below1.n', sum(gen[t] < fn[t] - 1.0 for t in TASKS)); put('gen.above1.n', sum(gen[t] > env[t] + 1.0 for t in TASKS))
    # parameters: the released models keep only what a task's outputs read. The training code also carries the read-outs no
    # output of the task reads (the edge read-out P, R and biases; the node-pointer and mask-one vectors) and two shape vectors
    # the block never reads (64 + 64); the paper's per-task count includes them ("counting an edge read-out that only edge
    # tasks use").
    par = {t: by[('census', t)][1]['parameters'] for t in TASKS}
    carried = {}
    for t in TASKS:
        c = R.json('models', f"census/{t}/seed1/config.json")
        kinds = {o['kind'] for o in c['outputs']}
        carried[t] = par[t] + 128 + (0 if 'edge' in kinds else 2 * 128 * 128 + 2 * 128) + (0 if 'node' in kinds else 128) + \
            (0 if 'one' in kinds and c['features'] is None else 128)
    put('params.model.lo', f'{min(par.values()) / 1e6:.2f}'); put('params.model.hi', f'{max(par.values()) / 1e6:.2f}')
    put('params.task.min', min(carried.values()), 'int'); put('params.task.max', max(carried.values()), 'int')
    put('ratio.lo', pub['fn_params'][0] / max(carried.values()), 'int'); put('ratio.hi', pub['fn_params'][1] / min(carried.values()), 'int')

    # the cold blocks: the order and greedy tasks compare their numbers (compare_first), the path and tree tasks build pair numbers
    ORDER = ['insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 'minimum', 'binary_search', 'quickselect', 'activity_selector', 'task_scheduling']
    PATH = ['bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_kruskal', 'floyd_warshall', 'mst_prim']
    cold, rest_med, rest_max = {}, [], []
    for t in ORDER + PATH:
        ms = by[('compare_first' if t in ORDER else 'pair_numbers', t)]
        cold[t] = [ms[s]['scores']['test'] for s in sorted(ms)]
        for s in sorted(ms):
            put(f'cold.{key(t)}.s{s}', ms[s]['scores']['test'], 'pct')
            lu = R.json('models', f"{ms[s]['name']}/config.json")['loops_used']['test']
            rest_med.append(lu['median']); rest_max.append(lu['max'])
        put(f'cold.{key(t)}', st.mean(cold[t]), 'pct'); put(f'cold.{key(t)}.n', len(cold[t]))
    put('cold.models', sum(len(v) for v in cold.values())); put('cold.models.exact', sum(sum(x == 100.0 for x in v) for v in cold.values()))
    put('cold.tasks', len(cold)); put('cold.exact.tasks', sum(len(v) >= 3 and min(v) == 100.0 for v in cold.values()))
    put('cold.rest.median.lo', min(rest_med), 'int'); put('cold.rest.median.hi', max(rest_med), 'int'); put('cold.rest.max', max(rest_max), 'int')
    put('cold.activity', st.mean(cold['activity_selector']), 'pct')
    put('cold.prim', st.mean(cold['mst_prim']), 'pct'); put('cold.fw', st.mean(cold['floyd_warshall']), 'pct')
    put('cold.prim.wrong', max(round((100 - x) / 100 * 32 * 64) for x in cold['mst_prim']), 'int')   # 32 test graphs of 64 nodes
    put('cold.models.all', sum(len(v) for v in cold.values()))
    put('dfs.gen', gen['dfs'], 'pct'); put('gen.minus.fnnh.r', round(st.mean(gen.values()) - st.mean(fnnh.values()), 2), 'pct1')
    put('hint.only.losses', ', '.join(PRETTY[t] for t in TASKS if fnnh[t] < gen[t] < fn[t]))   # behind only FloydNet's hinted runs
    # the bit loop with positions encoded (the cold block on topological sort and strong components)
    for t in ('topological_sort', 'strongly_connected_components'):
        put(f'bits.{key(t)}', st.mean(m['scores']['test'] for m in by[('cold', t)].values()), 'pct')
    put('bits.topo.exact', sum(m['scores']['test'] == 100.0 for m in by[('cold', 'topological_sort')].values()))
    # the block: width 128, half 64; its parameters (the shared block's count) plus the two 64-vectors the training code
    # carries and never reads
    w = R.json('models', 'census/bfs/seed0/config.json')['model']['width']
    put('block.half', w // 2)
    blk = R.json('shared', 'shared-units-dfs/seed0/config.json')['parameters_shared']
    put('params.block', blk + w, 'int'); put('params.block.unused', w, 'int')
    return dict(by=by, gen=gen, gen_seeds=seeds, cold=cold, ORDER=ORDER, PATH=PATH)


# ------------------------------------------------------------------------------------------------ set by the theory
def theory(R, pub, pt):
    """Each task in the setting the theory prescribes, fixed before the scores were read: the cold block where the answer
    depends on the numbers only through comparisons (order and greedy tasks: numbers compared; depth-first search, strong
    components and topological sort: positions compared, read after n loops), built pair numbers on the path and tree tasks,
    the general block elsewhere."""
    fn, fnnh, env, by, gen = pub['fn'], pub['fnnh'], pub['env'], pt['by'], pt['gen']
    mean = lambda b, t: st.mean(m['scores']['test'] for m in by[(b, t)].values())
    pc = [m for m in R.json('variants', 'positions_compared_only/index.json')]
    pc_mean = lambda t: st.mean(m['scores']['test'] for m in pc if m['task'] == t)
    th = {}
    for t in TASKS:
        if t in pt['ORDER']:
            th[t] = mean('compare_first', t)
        elif t in pt['PATH']:
            th[t] = mean('pair_numbers', t)
        elif t == 'dfs':
            th[t] = mean('compare_first', t)
        elif t in ('strongly_connected_components', 'topological_sort'):
            th[t] = pc_mean(t)
        else:
            th[t] = gen[t]
    put('theory.mean', st.mean(th.values()), 'pct'); put('theory.env', st.mean(env.values()), 'pct', 'published')
    for nm, ref in (('fn', fn), ('fnnh', fnnh), ('env', env)):
        put_wtl(f'theory.vs.{nm}', th, ref)
    put('theory.quickselect', th['quickselect'], 'pct'); put('theory.dfs', th['dfs'], 'pct'); put('theory.task-scheduling', th['task_scheduling'], 'pct')
    # no route arithmetic built in: the path and tree tasks take the general block
    l1 = {t: gen[t] if t in pt['PATH'] else th[t] for t in TASKS}
    put('theory.l1.mean', st.mean(l1.values()), 'pct'); put('theory.l1.minus.fnnh', st.mean(l1.values()) - st.mean(fnnh.values()), 'pct')
    put_wtl('theory.l1.vs.fn', l1, fn); put_wtl('theory.l1.vs.fnnh', l1, fnnh)
    put('theory.l1.cold.n', len(pt['ORDER']) + 3); put('theory.l1.gen.n', len(TASKS) - len(pt['ORDER']) - 3)
    # the benchmark's standard inputs only: task scheduling and quickselect were trained with two input changes; the general block
    std = {t: gen[t] if t in ('task_scheduling', 'quickselect') else l1[t] for t in TASKS}
    put('theory.std.mean', st.mean(std.values()), 'pct'); put('theory.std.minus.fn', st.mean(std.values()) - st.mean(fn.values()), 'pct')
    put_wtl('theory.std.vs.fn', std, fn)
    return dict(th=th, l1=l1, std=std)


# ------------------------------------------------------------------------------------------------ published models without hints
def nohint(pub, pt, th):
    P = json.loads((DATA / 'published.json').read_text())['no_hint_baselines']
    gen, l1, std, fnnh = pt['gen'], th['l1'], th['std'], pub['fnnh']
    bev = P['bevilacqua2023']; rod = {t: max(x for x in v if x is not None) for t, v in P['rodionov2023'].items()}
    rod0 = {t: v[0] for t, v in P['rodionov2023'].items()}; mah = {t: max(v) for t, v in P['mahdavi2023'].items()}
    dear = {t: v[3] for t, v in P['georgiev2024'].items()}
    dear_all = {t: max([x for x in v if x is not None] + [P['dear_triplet'].get(t, -1)] + P['dear_parallel'].get(t, [])) for t, v in P['georgiev2024'].items()}

    def wtl2(ours, theirs):   # the published values have two decimals: ours rounded to two before comparing
        w = ti = l = 0
        for t, v in theirs.items():
            o = round(ours[t] + 1e-9, 2)
            if abs(o - v) < 0.005: ti += 1
            elif o > v: w += 1
            else: l += 1
        return w, ti, l
    for nm, d in (('bev', bev), ('rod', rod), ('mah', mah), ('dear', dear)):
        put(f'nohint.{nm}.n', len(d), src='published'); put(f'nohint.{nm}.mean', st.mean(d.values()), 'pct', 'published')
        put(f'nohint.{nm}.gen', st.mean(gen[t] for t in d), 'pct'); put(f'nohint.{nm}.l1', st.mean(l1[t] for t in d), 'pct')
        for x, v in zip('wtl', wtl2(gen, d)):
            put(f'nohint.{nm}.gen.{x}', v)
    put('nohint.rod0.mean', st.mean(rod0.values()), 'pct', 'published'); put('nohint.dnar.bfs', P['dnar2025']['bfs'], 'int', 'published')
    env = {}
    for d in (bev, rod, mah, dear_all, P['dnar2025'], fnnh):
        for t, v in d.items():
            env[t] = max(env.get(t, -1), v)
    put('nohint.env.mean', st.mean(env.values()), 'pct', 'published')
    for nm, ours in (('gen', gen), ('l1', l1), ('std', std)):
        for x, v in zip('wtl', wtl2(ours, env)):
            put(f'nohint.env.{nm}.{x}', v)


# ------------------------------------------------------------------------------------------------ the variants
def variants(R, pub, pt, th):
    fn, fnnh, gen, by = pub['fn'], pub['fnnh'], pt['gen'], pt['by']
    V = lambda v: R.json('variants', f'{v}/index.json')
    test = lambda ms, t=None: [m['scores']['test'] for m in sorted(ms, key=lambda m: m['seed']) if t is None or m['task'] == t]
    # the registered-only theory column: each row at the recipe registered before its results
    reg = dict(th['th'])
    reg['task_scheduling'] = gen['task_scheduling']                                   # before the move to the cold block
    reg['dfs'] = st.mean(test(V('n_loops')))                                          # the general block read after n loops
    for t in ('topological_sort', 'strongly_connected_components'):                  # the bit loop with positions encoded
        reg[t] = st.mean(m['scores']['test'] for m in by[('cold', t)].values())
    qs = test(V('first_registered_seeds'))
    reg['quickselect'] = st.mean(qs)
    put('theory.reg.mean', st.mean(reg.values()), 'pct'); put('theory.reg.qs', reg['quickselect'], 'pct')
    put_wtl('theory.reg.vs.fn', reg, fn); put_wtl('theory.reg.vs.fnnh', reg, fnnh)
    for s, v in zip((1, 2), qs):
        put(f'qs.first-registered.s{s}', v, 'pct')
    allq = pt['cold']['quickselect']
    put('qs.all.n', len(allq)); put('qs.all.exact', sum(x == 100.0 for x in allq))
    put('qs.fn', fn['quickselect'], 'pct', 'published'); put('qs.oth', pub['oth']['quickselect'], 'pct', 'published')
    # task scheduling: the general block with deadlines in units of n; the cold block with the sampler's random positions
    u = test(V('deadline_units'))
    put('ts.units.mean', st.mean(u), 'pct'); put('ts.units.lo', min(u), 'pct'); put('ts.units.hi', max(u), 'pct')
    r = test(V('random_positions'))
    put('ts.random-positions.mean', st.mean(r), 'pct'); put('ts.random-positions.lo', min(r), 'pct'); put('ts.random-positions.hi', max(r), 'pct')
    put('ts.random-positions.n', len(r)); put('ts.random-positions.gap', th['th']['task_scheduling'] - st.mean(r), 'pct')
    # positions compared in a register (topological sort, strong components), and strong components with the multi-entry pool
    pc = V('positions_compared_only')
    for t in ('topological_sort', 'strongly_connected_components'):
        v = test(pc, t)
        put(f'reg.{key(t)}', st.mean(v), 'pct'); put(f'reg.{key(t)}.lo', min(v), 'pct'); put(f'reg.{key(t)}.hi', max(v), 'pct')
    put('reg.topo.exact', sum(x == 100.0 for x in test(pc, 'topological_sort')))
    d = test(V('scc_multi_entry_pool'))
    put('scc.doors', st.mean(d), 'pct'); put('scc.doors.lo', min(d), 'pct'); put('scc.doors.hi', max(d), 'pct')
    for s, v in enumerate(d):
        put(f'scc.doors.s{s}', v, 'pct')
    put('scc.doors.gain', st.mean(d) - st.mean(test(pc, 'strongly_connected_components')), 'pct')
    # the string matchers with a stored count given as a program (tests start open)
    sc = V('stored_count')
    for t in ('naive_string_matcher', 'kmp_matcher'):
        v = test(sc, t)
        put(f'str.count.open.{key(t)}', st.mean(v), 'pct'); put(f'str.count.open.{key(t)}.lo', min(v), 'pct')
    put('str.count.open.runs', len(sc)); put('str.count.open.exact', sum(m['scores']['test'] == 100.0 for m in sc))
    # the dynamic programs with the recurrence given: fixed tests, and tests learned
    fx = V('recurrence_given')
    put('dp.fixed.models', len(fx)); put('dp.fixed.exact', sum(m['scores']['test'] == 100.0 for m in fx))
    # the cold count: the order, path and tree tasks at rest, and the three depth-first-order tasks with positions compared
    A = {t: v for t, v in pt['cold'].items()}
    A['dfs'] = [m['scores']['test'] for m in by[('compare_first', 'dfs')].values()]
    for t in ('topological_sort', 'strongly_connected_components'):
        A[t] = test(pc, t)
    for pre, cnt in (('cold.all', A), ('cold.no-route-sums', {t: v for t, v in A.items() if t not in pt['PATH']})):
        exact = [t for t, v in cnt.items() if len(v) >= 3 and min(v) == 100.0]
        put(f'{pre}.tasks', len(cnt)); put(f'{pre}.exact', len(exact))
        put(f'{pre}.short', ', '.join(PRETTY[t] for t in TASKS if t in cnt and t not in exact))
    put('cold.all.mean', st.mean(st.mean(v) for v in A.values()), 'pct')
    for m in V('recurrence_given_learned_tests'):
        put(f"dp.rand.{key(m['task'])}", m['scores']['test'], 'pct')
    # segment intersection: the triangle pairs on a product pair number (the learned form), and bilinear tests
    sg = test(V('product_pair_number'))
    put('seg.product.learn.mean', st.mean(sg), 'pct'); put('seg.product.learn.hi', max(sg), 'pct'); put('seg.product.learn.zero', sum(x == 0 for x in sg))
    put('seg.bilin.mean', st.mean(m['scores']['test'] for m in by[('census_bilinear', 'segments_intersect')].values()), 'pct')
    # item memory: one state per item (the tie-aware read is the registered score; plain: the library scorer)
    items = {m['task']: R.json('variants', f"{m['name']}/config.json") for m in V('item_memory')}
    th_ = th['th']
    IT = {'item': {t: items[t]['scores']['test'] for t in TASKS}, 'item.plain': {t: items[t]['scores_plain']['test'] for t in TASKS}}
    for pre, it in IT.items():
        near = [t for t in TASKS if th_[t] - it[t] <= 0.5 + 1e-9]
        rest = [t for t in TASKS if t not in near]
        put(f'{pre}.mean', st.mean(it.values()), 'pct'); put(f'{pre}.near.n', len(near)); put(f'{pre}.rest.n', len(rest))
        put(f'{pre}.near.list', ', '.join(PRETTY[t] for t in near))
        put(f'{pre}.rest.mean', st.mean(it[t] for t in rest), 'pct'); put(f'{pre}.rest.th.mean', st.mean(th_[t] for t in rest), 'pct')
        put(f'{pre}.exact', sum(it[t] == 100.0 for t in TASKS))
        for t in TASKS:
            put(f'{pre}.{key(t)}', it[t], 'pct')
    put('item.reads.differ', sum(IT['item'][t] != IT['item.plain'][t] for t in TASKS))
    put('item.minus.theory', st.mean(th_.values()) - st.mean(IT['item'].values()), 'pct')


# ------------------------------------------------------------------------------------------------ Max Retrieval
def max_retrieval(R):
    """Ten seeds of each network; % of test sets answered right at every size, by read."""
    P = json.loads((DATA / 'published.json').read_text())['max_retrieval']
    cfg = [dict(R.json('max_retrieval', f"{m['name']}/config.json"), folder=m['name'].split('/')[0]) for m in R.json('max_retrieval', 'index.json')]
    sizes = sorted({int(n) for c in cfg for r in c['scores'].values() for n in r})
    tab = lambda net, read: {n: [c['scores'][read][str(n)] for c in cfg if c['folder'] == net] for n in sizes}
    P_, Sc, Sw = tab('cold_count', 'cold'), tab('soft_count', 'cold'), tab('soft_count', 'warm')
    W, W2C, W1 = tab('softmax', 'softmax'), tab('softmax', 'cold'), tab('softmax_half_penalty', 'softmax')
    put('maxret.cold.min', min(min(v) for v in P_.values()), 'pct'); put('maxret.cold.mean.min', min(st.mean(v) for v in P_.values()), 'pct')
    put('maxret.cold.4096', st.mean(P_[4096]), 'pct')
    put('maxret.readcold.min', min(min(v) for v in Sc.values()), 'pct'); put('maxret.readcold.mean.min', min(st.mean(v) for v in Sc.values()), 'pct')
    put('maxret.warm.4096', st.mean(Sw[4096]), 'pct'); put('maxret.warm.4096.r', st.mean(Sw[4096]), 'int'); put('maxret.warm.16384', st.mean(Sw[16384]), 'pct')
    put('maxret.top', max(sizes), 'int'); put('maxret.train', 16); put('maxret.seeds', len(P_[16]))
    put('maxret.seeds.w', 'zero one two three four five six seven eight nine ten'.split()[len(P_[16])])
    put('maxret.w2c.4096', st.mean(W2C[4096]), 'pct1'); put('maxret.w.4096', st.mean(W[4096]), 'pct1')
    pub = {int(n): v for n, v in P['softmax_table1'].items()}
    for arm, T in (('W', W), ('W1', W1)):   # our softmax replication against the published table, sizes 64 to 4,096
        put(f'maxret.repl.{arm}.gap', max(pub[n] - st.mean(T[n]) for n in sizes if 64 <= n <= 4096), 'pct1')
    put('maxret.pub.4096', pub[4096], 'pct1', 'published'); put('maxret.best.4096', P['asentmax_4096'], 'pct1', 'published')


# ------------------------------------------------------------------------------------------------ shared blocks
def shared(R, pub, pt):
    fn, fnnh, gen = pub['fn'], pub['fnnh'], pt['gen']

    def seeds(variant, sub=''):
        out = []
        for s in (0, 1, 2):
            p = R.root('shared') / variant / sub / f'seed{s}' / 'config.json'
            if p.exists():
                out.append(json.loads(p.read_text())['tasks'])
        return out
    # the paper's shared block: deadlines in units of n for task scheduling, depth-first search read after n loops
    S = seeds('shared-units-dfs')
    per = [st.mean(T[t]['scores']['test'] for t in TASKS) for T in S]
    task = {t: st.mean(T[t]['scores']['test'] for T in S) for t in TASKS}
    put('shared.mean', st.mean(per), 'pct'); put('shared.lo', min(per), 'pct'); put('shared.hi', max(per), 'pct')
    put('shared.all8', st.mean(st.mean(T[t].get('scores_after_8_loops', T[t]['scores'])['test'] for t in TASKS) for T in S), 'pct')
    W = seeds('shared-units-dfs', 'weight_averaged')
    put('shared.swa', st.mean(st.mean(T[t]['scores']['test'] for t in TASKS) for T in W), 'pct')
    put('shared.minus.gen', st.mean(per) - st.mean(gen.values()), 'pct'); put('shared.minus.fn', st.mean(per) - st.mean(fn.values()), 'pct')
    put('shared.minus.fnnh', st.mean(per) - st.mean(fnnh.values()), 'pct')
    put('shared.seeds.above.fn', sum(m > st.mean(fn.values()) for m in per)); put('shared.seeds.above.gen', sum(m > st.mean(gen.values()) for m in per))
    for t in TASKS:
        put(f'shared.task.{key(t)}', task[t], 'pct')
    put('shared.ts', task['task_scheduling'], 'pct'); put('shared.dfs', task['dfs'], 'pct')
    for s, T in enumerate(S):
        put(f'shared.dfs.s{s}', T['dfs']['scores']['test'], 'pct')
    put('shared.dfs.8', st.mean(T['dfs']['scores_after_8_loops']['test'] for T in S), 'pct')
    put('shared.dag.min', min(T['dag_shortest_paths']['scores']['test'] for T in S), 'pct')
    put('shared.dag.swa', st.mean(T['dag_shortest_paths']['scores']['test'] for T in W), 'pct')
    # the first shared block (no units, depth-first search after 8 loops)
    F = seeds('shared')
    fper = [st.mean(T[t]['scores']['test'] for t in TASKS) for T in F]
    ftask = {t: st.mean(T[t]['scores']['test'] for T in F) for t in TASKS}
    put('shared.first.mean', st.mean(fper), 'pct'); put('shared.first.lo', min(fper), 'pct'); put('shared.first.hi', max(fper), 'pct')
    for s, v in enumerate(fper):
        put(f'shared.first.s{s}', v, 'pct')
    put('shared.first.minus.fnnh', st.mean(fper) - st.mean(fnnh.values()), 'pct')
    put('shared.first.seeds.above.fn', sum(m > st.mean(fn.values()) for m in fper))
    put_wtl('shared.first.vs.fn', ftask, fn)
    for t in TASKS:
        put(f'shared.first.task.{key(t)}', ftask[t], 'pct')
    put('shared.delta', round(st.mean(per), 2) - round(st.mean(fper), 2), 'pct')
    put('shared.ts.old', ftask['task_scheduling'], 'pct'); put('shared.dfs.old', ftask['dfs'], 'pct')
    losses = {t: task[t] - ftask[t] for t in TASKS if task[t] < ftask[t] - 1e-9}
    put('shared.losses.n', len(losses))
    for t in ('dag_shortest_paths', 'lcs_length', 'activity_selector', 'binary_search', 'optimal_bst'):
        put(f'shared.loss.{key(t)}', -losses[t], 'pct')
    # the general-and-cold shared block: weight-averaged parameters at its registered read, and final parameters after 8 loops
    G = seeds('shared-warm-plus-cold'); G8 = seeds('shared-warm-plus-cold-final')
    put('shared.general-and-cold.mean', st.mean(st.mean(T[t]['scores']['test'] for t in TASKS) for T in G), 'pct')
    g8 = [st.mean(T[t]['scores']['test'] for t in TASKS) for T in G8]
    put('shared.general-and-cold.final8', st.mean(g8), 'pct'); put('shared.general-and-cold.gap', st.mean(fper) - st.mean(g8), 'pct')
    # five tasks: one shared block against the five one-task general blocks
    T5 = seeds('shared-five-tasks')[0]
    five = sorted(T5)
    put('shared.five-tasks.mean', st.mean(T5[t]['scores']['test'] for t in five), 'pct')
    single = st.mean(pt['gen_seeds'][t][0] for t in five)       # seed 0 of the one-task general blocks
    put('single.five-tasks.mean', single, 'pct'); put('shared.five-tasks.gap', single - st.mean(T5[t]['scores']['test'] for t in five), 'pct')
    return dict(task=task, first=ftask)


# ------------------------------------------------------------------------------------------------ constraint problems
def csp(R):
    """2-SAT (trained on at most 16 variables) and two-colouring (32), three seeds each, weight-averaged parameters."""
    cfg = lambda t, k, v='cold': R.json('csp', f'{v}/{t}/seed{k}/config.json')
    for t, nm in (('two_sat', '2sat'), ('two_colouring', '2col')):
        C = [cfg(t, k) for k in range(3)]
        top = [c['scores']['n500'] for c in C]
        put(f'csp.{nm}.lo', min(top), 'pct'); put(f'csp.{nm}.exact', sum(v == 100.0 for v in top))
        put(f'csp.{nm}.train', max(C[0]['train']['variables']))
    put('csp.vars', max(sp['variables'] for sp in cfg('two_sat', 0)['splits'].values()))
    first = cfg('two_sat', 0, 'cold_first_registered')        # the first registered 2-SAT run
    put('csp.2sat.orig.n500', first['scores']['n500'], 'pct'); put('csp.2sat.orig.test', first['scores']['test'], 'pct')


# ------------------------------------------------------------------------------------------------ the block beside a frozen model
def organ(R):
    I = R.json('organ', 'index.json')
    O = {o['name']: o for o in I['organs']}
    cfg = lambda name: R.json('organ', f'{name}/config.json')
    # comparison: three seeds on frozen Qwen2.5-0.5B's comparator bits, read after K + 4 loops
    C = [cfg(n) for n in sorted(O) if n.startswith('comparison/')]
    tests = lambda c: [(int(n), v) for sp in ('test', 'test1000') for n, v in c['scores'][sp].items()]
    assert all(v['right_pct']['at_K_plus_4'] == 100.0 for c in C for _, v in tests(c))
    put('organ.seeds', len(C)); put('organ.top', max(n for n, _ in tests(C[0])), 'int'); put('organ.train', max(C[0]['train']['facts']))
    put('organ.questions', sum(v['questions'] for c in C for _, v in tests(c)) // len(C), 'int')
    W = {n: v for n, v in tests(cfg('comparison/controls/warm/seed0'))}
    for n in (64, 256, 512, 1000):
        put(f'organ.warm.{n}', W[n]['right_pct']['at_K_plus_4'], 'pct')
    put('organ.warm.256.r', W[256]['right_pct']['at_K_plus_4'], 'int')
    put('organ.warm.512.q', W[512]['questions']); put('organ.warm.1000.q', W[1000]['questions'])
    alone = I['frozen_model_alone_right_pct']
    a = [v for sp in ('test', 'test1000') for v in alone['comparison/qwen2.5-0.5b'][sp].values()]
    put('organ.alone.lo', min(a), 'pct1'); put('organ.alone.hi', max(a), 'pct1')
    # kinship: the block beside frozen Qwen2.5-7B and -1.5B; directions from the model's own reading
    words = 'zero one two three four five six seven eight nine ten'.split()
    pooled7 = [n for n in O if n.startswith('kinship/qwen2.5-7b/') and '/per_sentence' not in n]
    small = [n for n in O if n.startswith('kinship/qwen2.5-1.5b/')]
    sent = [n for n in O if n.startswith('kinship/qwen2.5-7b/per_sentence')]
    put('kinship.seeds.7b', len(pooled7)); put('kinship.seeds.1p5b', len(small))
    put('kinship.seeds.7b.w', words[len(pooled7)]); put('kinship.seeds.1p5b.w', words[len(small)])
    K = {n: cfg(n) for n in pooled7 + small + sent}
    assert all(v['right_pct']['at_K_plus_4'] == 100.0 for n in pooled7 + small for sp in ('test', 'test1000') for v in K[n]['scores'][sp].values())
    put('kinship.top', max(int(x) for sp in ('test', 'test1000') for x in K[pooled7[0]]['scores'][sp]), 'int')
    put('kinship.train', max(K[pooled7[0]]['train']['statements']))
    put('kinship.sent.1000', min(K[n]['scores']['test1000']['1000']['right_pct']['at_K_plus_4'] for n in sent), 'pct')
    learned = [K[n]['scores']['val6']['16']['right_pct']['at_K_plus_4'] == 100.0 for n in K]   # a start learned: all of val6 right
    put('kinship.learned.model', sum(learned)); put('kinship.starts.model', len(learned))
    S = I['kinship_qwen2_5_0_5b_every_start']                  # starts on the 0.5B read-ins, directions learned from answers
    put('kinship.learned.answers', sum(e['learned'] for e in S)); put('kinship.starts.answers', len(S))
    a7 = alone['kinship/qwen2.5-7b']['test1000']['1000']
    put('kinship.alone7.1000', a7, 'pct'); put('kinship.alone7.1000.r', a7, 'int')


# ------------------------------------------------------------------------------------------------ depth, and the committee
def depth_and_committee(R):
    """The eleven-task shared blocks on the deeper graphs (Bellman-Ford and Prim; read 4 loops after the registered read,
    weight-averaged parameters), and strong components trained on committee-chosen cases."""
    for b, folder in (('warm1', 'warm'), ('warm2', 'warm_plus_warm'), ('cold', 'warm_plus_cold'), ('tied', 'warm_plus_cold_tied')):
        T = R.json('eleven', f'{folder}/seed0/config.json')['tasks']
        for t, nm in (('bellman_ford', 'bf'), ('mst_prim', 'prim')):
            for sp in ('test', 'chain64', 'deep128', 'deep256'):
                put(f'deep.{nm}.{sp}.{b}', T[t]['scores_4_loops_later'][sp], 'pct1')
    c = [m['scores']['test'] for m in sorted(R.json('scc-committee', 'index.json'), key=lambda m: m['seed'])]
    put('committee.scc.after', st.mean(c), 'pct'); put('committee.scc.after.r', st.mean(c), 'pct1')
    put('committee.scc.lo', min(c), 'pct'); put('committee.scc.hi', max(c), 'pct')
    pc = [m['scores']['test'] for m in R.json('variants', 'positions_compared_only/index.json') if m['task'] == 'strongly_connected_components']
    put('committee.scc.before', st.mean(pc), 'pct'); put('committee.scc.before.r', st.mean(pc), 'pct1')
    put('committee.scc.hand', st.mean(m['scores']['test'] for m in R.json('variants', 'scc_multi_entry_pool/index.json')), 'pct')


# ------------------------------------------------------------------------------------------------ CLUTRR
def clutrr(R):
    """Two splits (trained on k = 2, 3, 4 or on 2, 3; tested on 2 to 10), three seeds each: the block on the true graph; the
    zero-shot cost read on the frozen model's reading; the trained read-in; the frozen model alone."""
    P = json.loads((DATA / 'published.json').read_text())['clutrr']
    words = 'zero one two three four five six seven eight nine ten'.split()
    I = R.json('clutrr', 'index.json')
    cfg = lambda rel: R.json('clutrr', f'{rel}/config.json')
    pct = lambda d: 100 * d['right'] / d['questions']
    for s, sp in (('234', 'gen_train234_test2to10'), ('23', 'gen_train23_test2to10')):
        Z = [cfg(f'text/{sp}/zero_shot/seed{k}')['scores'] for k in range(3)]
        T = [cfg(f'text/{sp}/trained/seed{k}')['scores'] for k in range(3)]
        G = [cfg(f'graph/{sp}/seed{k}')['scores'] for k in range(3)]
        cs = [pct(z['cost_read']['test']['all']) for z in Z]
        put(f'clutrr.cost.{s}', st.mean(cs), 'pct1'); put(f'clutrr.cost.{s}.lo', min(cs), 'pct1'); put(f'clutrr.cost.{s}.hi', max(cs), 'pct1')
        put(f'clutrr.cost.{s}.k10', st.mean(pct(z['cost_read']['test']['10']) for z in Z), 'int')
        put(f'clutrr.cost.{s}.val', 100 * st.mean(z['cost_read']['validation']['all']['accuracy'] for z in Z), 'pct1')
        put(f'clutrr.test.{s}', Z[0]['cost_read']['test']['all']['questions'], 'int')
        put(f'clutrr.bits.{s}', st.mean(pct(z['bits']['test']['all']) for z in Z), 'pct1')
        put(f'clutrr.bits.{s}.k10', st.mean(pct(z['bits']['test']['10']) for z in Z), 'int')
        ls = [pct(t['cost_read']['test']['all']) for t in T]
        put(f'clutrr.learn.{s}', st.mean(ls), 'pct1'); put(f'clutrr.learn.{s}.lo', min(ls), 'pct1'); put(f'clutrr.learn.{s}.hi', max(ls), 'pct1')
        put(f'clutrr.learn.{s}.val', 100 * st.mean(t['cost_read']['validation']['all']['accuracy'] for t in T), 'pct1')
        put(f'clutrr.learn.{s}.k10', st.mean(pct(t['cost_read']['test']['10']) for t in T), 'int')
        put(f'clutrr.learn.{s}.bits', st.mean(pct(t['fallback_bits']['test']['all']) for t in T), 'pct1')
        put(f'clutrr.learn.{s}.gain', st.mean(ls) - st.mean(cs), 'pct1')
        A = I['frozen_model_alone'][sp]
        put(f'clutrr.alone.steps.{s}', 100 * A['test_steps']['all']['accuracy'], 'pct1')
        put(f'clutrr.alone.word.{s}', 100 * A['test_word']['all']['accuracy'], 'pct1')
        put(f'clutrr.alone.steps.{s}.k10', 100 * A['test_steps']['10']['accuracy'], 'int')
        right = [sum(g['test'][k]['right'] for k in g['test'] if k != 'all') for g in G]
        assert len(set(right)) == 1, right
        put(f'clutrr.graph.right.{s}', right[0], 'int'); put(f'clutrr.graph.of.{s}', sum(G[0]['test'][k]['questions'] for k in G[0]['test'] if k != 'all'), 'int')
        if s == '234':
            put('clutrr.alone.steps.234.r', 100 * A['test_steps']['all']['accuracy'], 'int')
            put('clutrr.learn.234.r', st.mean(ls), 'int')
            put('clutrr.learn.234.overalone', round(st.mean(ls), 1) - round(100 * A['test_steps']['all']['accuracy'], 1), 'int')
            put('clutrr.learn.234.k4', st.mean(pct(t['cost_read']['test']['4']) for t in T), 'pct1')
            put('clutrr.cost.234.k4', st.mean(pct(z['cost_read']['test']['4']) for z in Z), 'pct1')
            put('clutrr.graph.miss.234.w', words[sum(G[0]['test'][k]['questions'] for k in G[0]['test'] if k != 'all') - right[0]])
            put('clutrr.graph.k5to10', sum(G[0]['test'][k]['questions'] for k in G[0]['test'] if k != 'all' and int(k) >= 5), 'int')
        else:
            put('clutrr.learn.23.above', words[sum(v > P['dsr_lm'] for v in ls)]); put('clutrr.cost.23.above', words[sum(v > P['dsr_lm'] for v in cs)])
    graphs = [o for o in I['organs'] if o['organ'] == 'graph']
    put('clutrr.graph.runs', len(graphs)); put('clutrr.graph.runs.w', words[len(graphs)])
    put('clutrr.rules', len(cfg(graphs[0]['name'])['rules']))
    rev = cfg(f"text/gen_train234_test2to10/zero_shot/seed0")['frozen_model']['revision']
    put('clutrr.revision', rev); put('clutrr.revision.short', rev[:12])
    put('clutrr.dsr', P['dsr_lm'], src='published'); put('clutrr.gpt3', P['gpt3_zero_shot'], src='published')
    rl = [v for k, v in P['k10'].items() if k != 'GAT']
    put('clutrr.pub.k10.lo', f'{min(rl):.2f}', src='published'); put('clutrr.pub.k10.hi', f'{max(rl):.2f}', src='published')
    put('clutrr.gat.k10', f"{P['k10']['GAT']:.2f}", src='published')
    pt = P['path_of_thoughts']
    put('pot.gpt4o.cot', pt['gpt4o_cot'], src='published'); put('pot.gpt4o.sym', pt['gpt4o_symbolic'], src='published')
    put('pot.gpt5', pt['gpt5_zero_shot'], src='published'); put('pot.llama70', pt['llama3_70b_pot_llm'], src='published')
    put('pot.gpt4o.best', pt['gpt4o_pot_llm'], src='published')
    put('popper.qwen.long', round(P['popper']['qwen3_5_4b']), src='published'); put('popper.gpt.long', round(P['popper']['gpt5_4']), src='published')


# ------------------------------------------------------------------------------------------------ run and compare
def measured():
    """Every number read from data/measured/<family>.json -> {key: family}."""
    fam = {}
    for f in sorted((DATA / 'measured').glob('*.json')):
        for k, v in json.loads(f.read_text())['values'].items():
            assert k not in N, f'{k} is both recomputed and read'
            put(k, v, src='read'); fam[k] = f.stem
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--local', help='a local copy of the repositories (DIR/<folder>, folders as in REPOS)')
    ap.add_argument('--recomputed', action='store_true', help='only the recomputed numbers')
    a = ap.parse_args()
    R = Repos(a.local)
    pub = published()
    pt = per_task(R, pub)
    th = theory(R, pub, pt)
    nohint(pub, pt, th)
    variants(R, pub, pt, th)
    shared(R, pub, pt)
    max_retrieval(R)
    csp(R)
    organ(R)
    depth_and_committee(R)
    clutrr(R)
    recomputed = set(N)
    fam = measured()
    paper = json.loads((DATA / 'paper.json').read_text())
    keys = sorted(k for k in N if not a.recomputed or k in recomputed)
    w = max(len(k) for k in keys)
    print(f'{"key":{w}}  {"paper":>14}  {"here":>14}  {"source":9}  match')
    bad = []
    for k in keys:
        p, v = paper.get(k), N[k]
        ok = 'read' if SRC[k] == 'read' else 'yes' if p == v else 'not in paper' if p is None else 'NO'
        if ok == 'NO':
            bad.append(k)
        print(f'{k:{w}}  {p or "":>14}  {v:>14}  {SRC[k]:9}  {ok}')
    rc = [k for k in recomputed if k in paper]
    by_src = {s: sum(1 for k in rc if SRC[k] == s) for s in ('model', 'published')}
    print(f'\nThe paper prints {len(paper)} numbers.')
    print(f'Recomputed: {len(rc)} ({by_src["model"]} from the released models, {by_src["published"]} from published numbers); '
          f'{len(rc) - len(bad)} match, {len(bad)} differ' + (f' ({", ".join(bad)})' if bad else '') + '.')
    print(f'Read from data/measured: {len(fam)} numbers in {len(set(fam.values()))} families ({", ".join(sorted(set(fam.values())))}).')
    missing = sorted(set(paper) - set(N))
    if missing:
        print('Neither recomputed nor read:', ', '.join(missing))
    return 1 if bad or missing else 0

if __name__ == '__main__':
    sys.exit(main())
