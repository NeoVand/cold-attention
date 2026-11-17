"""The results tables (README and model card) from the released models' scores.

  python scripts/tables.py <models dir or index.json> [results.json] [--shared <shared block dir>] [--variants <variants dir>]

Writes results.json (every model's test and val score) when asked, and prints the markdown tables: the headline (against
FloydNet: the shared general block when --shared is given; one general block per task, three seeds; and, with
--variants, each task in the setting the theory prescribes) and the task-by-task table.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from witness_clrs.tasks import FLOYDNET, NAMES  # noqa: E402

# The block each task opens on in the demo: the one its symmetry calls for (the demo's site/src/lib/demo/tasks.ts).
DEFAULTS_NOTE = ('**Default block per task** (the one the task\'s symmetry calls for; the demo opens on it): the cold block with '
                 'numbers compared for the sorts, minimum, quickselect, binary search, activity selector, task scheduling and '
                 'depth-first search (positions in its register); the cold block with pair numbers for Bellman-Ford, Dijkstra, '
                 'DAG shortest paths, Floyd-Warshall, Kruskal and Prim; the cold block for topological sort and strongly '
                 'connected components; the general block for the rest.')

BLOCK_NAMES = {'census': 'General block', 'census_bilinear': 'General block, bilinear tests', 'cold': 'Cold block',
               'compare_first': 'Cold block, numbers compared', 'pair_numbers': 'Cold block, pair numbers'}

# The setting the theory prescribes (fixed before the scores were read): the cold block where a task's answer depends on its
# numbers only through comparisons, with loops in proportion to n for depth-first search; the general block elsewhere.
THEORY_COLD = ('insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 'minimum', 'binary_search', 'activity_selector',
               'quickselect', 'task_scheduling')
OFF_STANDARD = ('task_scheduling', 'quickselect')   # their cold runs changed the inputs (deadlines in units of n; all sizes)

def load(src):
    p = Path(src)
    if p.is_dir():
        p = p / 'index.json'
    return json.loads(p.read_text())


def wtl(col):
    return [sum(c > FLOYDNET[t] + 1e-9 for t, c in col.items()), sum(abs(c - FLOYDNET[t]) <= 1e-9 for t, c in col.items()),
            sum(c < FLOYDNET[t] - 1e-9 for t, c in col.items())]


def shared_scores(directory):
    """{seed: {task: test score at the registered read}} of a shared block's seed folders."""
    out = {}
    for c in sorted(Path(directory).glob('seed*/config.json')):
        cfg = json.loads(c.read_text())
        out[int(c.parent.name[4:])] = {t: v['scores']['test'] for t, v in cfg['tasks'].items()}
    return out


def theory(by, variants):
    """Each task in the setting the theory prescribes; strongly connected components and depth-first search with positions
    compared, topological sort on the cold block. -> (as trained, on the standard inputs)."""
    pc = [m['scores']['test'] for m in load(Path(variants) / 'positions_compared_only') if m['task'] == 'strongly_connected_components']
    mean = lambda b, t: st.mean(by[(b, t)])
    out = []
    for off in ((), OFF_STANDARD):
        th = {}
        for t in FLOYDNET:
            if t in THEORY_COLD:
                th[t] = mean('census', t) if t in off else mean('compare_first', t)
            elif t == 'dfs':
                th[t] = mean('compare_first', t)
            elif t == 'strongly_connected_components':
                th[t] = st.mean(pc)
            elif t == 'topological_sort':
                th[t] = mean('cold', t)
            else:
                th[t] = mean('census', t)
        out.append(th)
    return out


def tables(index, shared=None, variants=None):
    by = {}
    for m in index:
        by.setdefault((m['block'], m['task']), []).append(m['scores']['test'])
    tasks = sorted(FLOYDNET, key=lambda t: NAMES[t].lower())
    census = {t: by[('census', t)] for t in tasks if ('census', t) in by}
    sh = shared_scores(shared) if shared else None
    L = []
    if len(census) == 30:
        mF = st.mean(FLOYDNET.values())
        L += ['| | Average accuracy, 30 tasks, 64 items | Above / equal / below FloydNet |', '|---|---:|---:|']
        if sh:
            seeds = [st.mean(v.values()) for _, v in sorted(sh.items())]
            col = {t: st.mean(sh[s][t] for s in sh) for t in tasks}
            L.append(f'| One general block shared by all 30 tasks ({len(sh)} seeds) | **{st.mean(seeds):.2f}** (seeds {", ".join(f"{q:.2f}" for q in seeds)}) | {" / ".join(map(str, wtl(col)))} |')
        seeds = [st.mean(m['scores']['test'] for m in index if m['block'] == 'census' and m['seed'] == s) for s in range(3)]
        cm = {t: st.mean(v) for t, v in census.items()}
        L.append(f'| One general block per task (3 seeds) | **{st.mean(cm.values()):.2f}** (seeds {", ".join(f"{q:.2f}" for q in seeds)}) | {" / ".join(map(str, wtl(cm)))} |')
        if variants:
            full, std = theory(by, variants)
            L.append(f'| Each task in the setting the theory prescribes, on the standard inputs | **{st.mean(std.values()):.2f}** | {" / ".join(map(str, wtl(std)))} |')
            L.append(f'| The same, with the two input changes (task scheduling, quickselect) | {st.mean(full.values()):.2f} | {" / ".join(map(str, wtl(full)))} |')
        L += [f'| FloydNet, one model per task, best of runs with and without hints | {mF:.2f} | |', '']
    blocks = [b for b in ('census', 'census_bilinear', 'cold', 'compare_first', 'pair_numbers') if any(k[0] == b for k in by)]
    names = (['Shared general block'] if sh else []) + [BLOCK_NAMES[b] for b in blocks]
    L += ['| Task | FloydNet | ' + ' | '.join(names) + ' |', '|---|---:|' + '---:|' * len(names)]
    for t in tasks:
        cells = [f'{st.mean(sh[s][t] for s in sh):.2f}'] if sh else []
        for b in blocks:
            v = by.get((b, t))
            cells.append(f'{st.mean(v):.2f}' + (f' ({len(v)})' if len(v) != 3 else '') if v else '')
        L.append(f'| {NAMES[t]} | {FLOYDNET[t]:.2f} | ' + ' | '.join(cells) + ' |')
    L += ['', DEFAULTS_NOTE]
    L += ['', 'Test accuracy (%) on the official CLRS-30 test split (64 items; 80 for the string matchers), the mean over seeds '
          '(3 unless a count is given in brackets), read by each block\'s registered rule and scored with the library\'s scorer. '
          'The shared general block is read after 8 loops, depth-first search after n. FloydNet: arXiv 2601.19094, Table 9 '
          '(best of runs with and without hints).']
    return '\n'.join(L)

if __name__ == '__main__':
    args, opt = [a for a in sys.argv[1:]], {}
    for k in ('--shared', '--variants'):
        if k in args:
            i = args.index(k); opt[k[2:]] = args[i + 1]; del args[i:i + 2]
    idx = load(args[0])
    if len(args) > 1:
        Path(args[1]).write_text(json.dumps(idx, indent=1) + '\n')
    print(tables(idx, **opt))
