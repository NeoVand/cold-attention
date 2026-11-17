"""Check that this code reproduces the released models' predictions on the official CLRS-30 splits.

For every model directory (model.safetensors, config.json, reference_predictions.npz) the registered read and the
prediction after every loop are recomputed and compared entry by entry with the reference predictions (those the model
made when it was trained). Prints one line per model and writes a JSON report.

  python scripts/check_release.py <models dir> <data dir> [report.json] [filter]

<data dir>/<task>/{test,val}.npz are the official splits (scripts/generate.py). The reference predictions were made on an
NVIDIA L4 with jax 0.8.0; on other hardware a few entries away from the registered read can differ in the last float bit
of a near tie (the report counts them).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from witness_clrs import data as D, evaluate as E, weights as W  # noqa: E402


def check(model_dir, data_dir, splits=None):
    p, cfg = W.load(model_dir)
    outs = [(o['name'], o['kind']) for o in cfg['outputs']]
    ref = np.load(Path(model_dir) / 'reference_predictions.npz')
    res, t0 = dict(name=cfg['name']), time.time()
    splits = splits or tuple(cfg.get('splits', ('test', 'val')))
    for sp in splits:
        d = D.load_split(Path(data_dir) / cfg['task'] / f'{sp}.npz')
        r = {}
        trim = False
        if cfg['model'].get('read', {}).get('rule') == 'log2':  # each input after its own ceil(log2 n) + plus loops
            B = max(1, min(16, 1024 // d['A'].shape[-1]))      # inputs per batch, so that the largest graphs fit
            read, K, per = E.predict_log2(p, cfg['model'], outs, d, B)
            r['loops_differ'] = int((K != ref[f'{sp}/loops']).sum())
            trim = True  # compare the loops run here (up to the split's largest count) with the same loops of the reference
        elif cfg['model'].get('read', {}).get('rule') == 'n':   # each input after its own n loops
            read, K, per = E.predict_n(p, cfg['model'], outs, d, cfg['train'].get('eval_batch', 16))
        elif cfg['model']['block'] == 'census':
            per = E.predict_loops(p, cfg['model'], outs, d, 16, cfg['train'].get('eval_batch', 16))
            read = {k: v[E.CENSUS_LOOPS - 1] for k, v in per.items()}
        else:
            rr = E.predict_rest(p, cfg['model'], outs, d, cfg['train'].get('eval_batch', 16))
            per, read = rr['per_loop'], rr['pred']
            r['loops_differ'] = int((rr['loops'] != ref[f'{sp}/loops']).sum())
        names = [name for name, _ in outs] + [f'{name}#unique' for name, _ in outs if f'{sp}/per_loop/{name}#unique' in ref.files]
        for name in names:
            a, b = per[name], ref[f'{sp}/per_loop/{name}']
            b = b[:a.shape[0]] if trim else b
            r[name] = dict(entries=int(read[name].size), differ_at_read=int((read[name] != ref[f'{sp}/read/{name}']).sum()),
                           differ_any_loop=int((a != b).sum()) if a.shape == b.shape else -1)
        r['score'] = round(100 * (E.score_ties(read, d, outs) if len(names) > len(outs) else E.score(read, d, outs)), 4)
        res[sp] = r
    res['seconds'] = round(time.time() - t0, 1)
    res['read_exact'] = all(v['differ_at_read'] == 0 and res[sp].get('loops_differ', 0) == 0 for sp in splits for v in res[sp].values() if isinstance(v, dict))
    res['all_loops_exact'] = res['read_exact'] and all(v['differ_any_loop'] == 0 for sp in splits for v in res[sp].values() if isinstance(v, dict))
    return res


def check_shared(model_dir, data_dir, tasks=None, splits=None):
    """A shared block: every task's model (the shared block with the task's own parameters), read by its rule (after 8
    loops, after each input's own n loops, or after ceil(log2 n) + c loops), against the reference predictions; the same
    report per task as check(). splits: every split the reference holds for the task (the library's test and val from
    data_dir; any further split from the release's own data/ folder, its first inputs as many as the reference has)."""
    p, cfg = W.load(model_dir)
    ref = np.load(Path(model_dir) / 'reference_predictions.npz')
    extra = Path(model_dir).parents[1] / 'data'
    out = []
    for t in tasks or sorted(cfg['tasks']):
        pt, ct = W.shared_task(p, cfg, t)
        outs = [(o['name'], o['kind']) for o in ct['outputs']]
        res, t0 = dict(name=ct['name']), time.time()
        have = sorted({k.split('/')[1] for k in ref.files if k.startswith(t + '/')}, key=lambda sp: (sp not in ('test', 'val'), sp))
        for sp in splits or have:
            d = D.load_split((Path(data_dir) if sp in ('test', 'val') else extra) / t / f'{sp}.npz')
            k = ref[f'{t}/{sp}/read/{outs[0][0]}'].shape[0]
            d = {kk: v[:k] for kk, v in d.items()}
            if ct['model'].get('read', {}).get('rule') == 'n':
                read, _, per = E.predict_n(pt, ct['model'], outs, d, 16)
            elif ct['model'].get('read', {}).get('rule') == 'log2':
                per = E.predict_loops(pt, ct['model'], outs, d, 16, 16)
                K = E.log2_loops(d['n'], ct['model']['read']['plus']); i = np.arange(len(K))
                read = {kk: v[K - 1, i] for kk, v in per.items()}
            else:
                per = E.predict_loops(pt, ct['model'], outs, d, 16, 16)
                read = {k: v[E.CENSUS_LOOPS - 1] for k, v in per.items()}
            r = {}
            for name, _ in outs:
                a, b = per[name], ref[f'{t}/{sp}/per_loop/{name}']
                r[name] = dict(entries=int(read[name].size), differ_at_read=int((read[name] != ref[f'{t}/{sp}/read/{name}']).sum()),
                               differ_any_loop=int((a != b).sum()) if a.shape == b.shape else -1)
            r['score'] = round(100 * E.score(read, d, outs), 4)
            res[sp] = r
        res['seconds'] = round(time.time() - t0, 1)
        sps = splits or have
        res['read_exact'] = all(res[sp][n]['differ_at_read'] == 0 for sp in sps for n, _ in outs)
        res['all_loops_exact'] = res['read_exact'] and all(res[sp][n]['differ_any_loop'] == 0 for sp in sps for n, _ in outs)
        out.append(res)
    return out


def main(models, data, report='check_release.json', filt=''):
    dirs = sorted(q.parent for q in Path(models).rglob('config.json') if filt in str(q))
    out = []
    for q in dirs:
        r = check(q, data)
        out.append(r)
        print(json.dumps({k: r[k] for k in ('name', 'read_exact', 'all_loops_exact', 'seconds')} | {'test': r['test']['score']}), flush=True)
        Path(report).write_text(json.dumps(out, indent=1))
    print(f"read exact: {sum(r['read_exact'] for r in out)} of {len(out)}; every loop exact: {sum(r['all_loops_exact'] for r in out)} of {len(out)}")
    return out


if __name__ == '__main__':
    main(*sys.argv[1:])
