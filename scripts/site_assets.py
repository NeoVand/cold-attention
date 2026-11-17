"""Copy released models into the demo page's static folder and write their reference predictions as JSON.

  python scripts/site_assets.py <models dir> <site static dir> [seeds=0]
  python scripts/site_assets.py --shared <shared block dir> <site static dir>   (e.g. shared-units-dfs/seed0: every task's
                                                    model from the shared block, as <static>/models/shared/<task>/seed0)

<static>/models/<block>/<task>/seed<k>/{model.safetensors, config.json, reference.json}; reference.json holds the
registered read (and loops used) on the official test and val splits, for the page's parity test.
<static>/models/index.json lists them; <static>/models/scores.json holds every model's test and val score (all seeds),
for the page's score grid.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main(models, static, seeds='0'):
    keep = {int(s) for s in seeds.split(',')}
    index, scores = [], []
    for cfgp in sorted(Path(models).rglob('config.json')):
        cfg = json.loads(cfgp.read_text())
        if cfg['model'].get('bilinear_tests'):  # bilinear tests: not in the page's engines (Python only)
            continue
        seed = int(cfg['name'].rsplit('seed', 1)[1])
        scores.append(dict(name=cfg['name'], task=cfg['task'], block=cfg['block'], seed=seed, test=cfg['scores']['test'], val=cfg['scores']['val']))
        if seed not in keep:
            continue
        dest = Path(static) / 'models' / cfg['name']
        dest.mkdir(parents=True, exist_ok=True)
        for f in ('model.safetensors', 'config.json'):
            shutil.copyfile(cfgp.parent / f, dest / f)
        ref = np.load(cfgp.parent / 'reference_predictions.npz')
        doc = {k: ref[k].astype(int).tolist() for k in ref.files if '/read/' in k or k.endswith('/loops')}
        (dest / 'reference.json').write_text(json.dumps(doc, separators=(',', ':')))
        index.append({k: cfg[k] for k in ('name', 'task', 'task_name', 'block', 'parameters', 'scores')})
    (Path(static) / 'models' / 'index.json').write_text(json.dumps(index, indent=1))
    (Path(static) / 'models' / 'scores.json').write_text(json.dumps(scores, indent=1))
    print(len(index), 'models')


def shared(block_dir, static):
    """Every task of a shared block as its own census model (the shared blk with the task's own parameters), for the page."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from witness_clrs import weights as W
    params, config = W.load(block_dir)
    ref = np.load(Path(block_dir) / 'reference_predictions.npz')
    seed = int(Path(block_dir).name.replace('seed', ''))
    idx = Path(static) / 'models' / 'index.json'
    index = [e for e in json.loads(idx.read_text()) if e['block'] != 'shared'] if idx.exists() else []
    for t in config['tasks']:
        p, cfg = W.shared_task(params, config, t)
        name = f'shared/{t}/seed{seed}'
        cfg = dict(cfg, name=name, block='shared', parameters=int(sum(np.asarray(v).size for v in W.flatten(p).values())),
                   parameters_shared=config['parameters_shared'])
        dest = Path(static) / 'models' / name
        W.save(dest, p, cfg)
        pre = f'{t}/'
        doc = {k[len(pre):]: ref[k].astype(int).tolist() for k in ref.files if k.startswith(pre) and '/read/' in k}
        (dest / 'reference.json').write_text(json.dumps(doc, separators=(',', ':')))
        index.append({k: cfg[k] for k in ('name', 'task', 'task_name', 'block', 'parameters', 'scores')})
    idx.write_text(json.dumps(index, indent=1))
    sc = Path(static) / 'models' / 'scores.json'   # every seed's scores, for the page's score grid
    scores = [e for e in json.loads(sc.read_text()) if e['block'] != 'shared'] if sc.exists() else []
    for q in sorted(Path(block_dir).parent.glob('seed*/config.json')):
        c = json.loads(q.read_text()); k = int(q.parent.name.replace('seed', ''))
        scores += [dict(name=f'shared/{t}/seed{k}', task=t, block='shared', seed=k, test=v['scores']['test'], val=v['scores']['val']) for t, v in c['tasks'].items()]
    sc.write_text(json.dumps(scores, indent=1))
    print(len(config['tasks']), 'shared-block task models')


if __name__ == '__main__':
    if sys.argv[1] == '--shared':
        shared(*sys.argv[2:4])
        sys.exit()
    main(*sys.argv[1:])
