"""Score predictions with DeepMind's own scorer, clrs.evaluate (dm-clrs 2.0.3), and check our copy agrees.

  python -I scripts/score_library.py <models dir> [data dir] [filter]

Run in an environment with the library. For each model the official test and val splits are regenerated with the
library exactly as witness_clrs.generate makes them, and their sha256 is compared with the split files' manifests when
a data dir is given; the model's registered read (reference_predictions.npz, or a run's predictions.npz) is scored by
clrs.evaluate and by witness_clrs.evaluate.score.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clrs  # noqa: E402
from witness_clrs import generate  # noqa: E402
from witness_clrs.scoring import score  # noqa: E402
from witness_clrs.tasks import OUTPUTS  # noqa: E402


def library_split(task, split):
    c = clrs.CLRS30[split]
    num = c['num_samples'] * clrs.CLRS_30_ALGS_SETTINGS[task]['num_samples_multiplier']
    L = generate.split_length(task, split)
    fb = generate.sample(task, num, L, c['seed'])
    d = generate.pack(fb, L) if 'adj' in {dp.name for dp in fb.features.inputs} else generate.pack_items(fb)
    return fb, d


def evaluate(fb, pred_by_name):
    outputs, preds = list(fb.outputs), {}
    if any(dp.type_ == clrs.Type.SHOULD_BE_PERMUTATION for dp in outputs):
        # sorting: the truth as the library's run.py hands it to clrs.evaluate (a permutation pointer plus a mask_one on
        # the first item); our pointer p is given as one-hot rows plus the mask p_i = i
        from clrs._src import samplers
        outputs = samplers._preprocess_permutations(outputs, True)
    perms = {dp.name for dp in outputs if dp.type_ == clrs.Type.PERMUTATION_POINTER}
    for dp in outputs:
        if dp.type_ == clrs.Type.MASK_ONE and dp.name.endswith('_mask') and dp.name[:-5] in perms:
            continue
        a = pred_by_name[dp.name]
        if dp.type_ == clrs.Type.PERMUTATION_POINTER:
            n = dp.data.shape[-1]
            preds[dp.name + '_mask'] = clrs.DataPoint(name=dp.name + '_mask', location=dp.location, type_=clrs.Type.MASK_ONE,
                                                      data=(a == np.arange(n)).astype(np.float32))
            a = np.eye(n)[a]
        elif dp.type_ in (clrs.Type.MASK_ONE, clrs.Type.CATEGORICAL):
            a = np.eye(dp.data.shape[-1])[a]
        elif dp.type_ == clrs.Type.MASK:
            a = a.astype(np.float32)
        preds[dp.name] = clrs.DataPoint(name=dp.name, location=dp.location, type_=dp.type_, data=a)
    return float(clrs.evaluate(tuple(outputs), preds)['score'])


def main(models, data=None, filt=''):
    out, cache = [], {}
    for cfgp in sorted(Path(models).rglob('config.json')):
        if filt not in str(cfgp):
            continue
        cfg = json.loads(cfgp.read_text())
        task = cfg['task']
        f = cfgp.parent / ('reference_predictions.npz' if (cfgp.parent / 'reference_predictions.npz').exists() else 'predictions.npz')
        pz = np.load(f)
        res = dict(name=cfg['name'])
        for split in ('test', 'val'):
            if (task, split) not in cache:
                cache[task, split] = library_split(task, split)
                if data:  # the split files the models were checked on hold the same arrays
                    ref = np.load(Path(data) / task / f'{split}.npz')
                    assert generate.digest(dict(ref)) == generate.digest(cache[task, split][1]), (task, split, 'split differs from the library')
            fb, d = cache[task, split]
            names = [dp.name for dp in fb.outputs]
            pred = {nm: pz[f'{split}/read/{nm}'] for nm in names}
            lib = evaluate(fb, pred)
            ours = score(pred, d, OUTPUTS[task])
            assert abs(lib - ours) < 1e-6, (cfg['name'], split, lib, ours)
            res[split] = round(100 * lib, 4)
        out.append(res)
        print(json.dumps(res), flush=True)
    return out


if __name__ == '__main__':
    main(*sys.argv[1:])
