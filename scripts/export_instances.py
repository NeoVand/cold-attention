"""Write the official CLRS-30 splits for the demo page: one JSON file per task and split, every array exactly as in the
split file (padded, same dtype), base64 little-endian, so the page feeds the model the very bits Python does.

  python scripts/export_instances.py <data dir> <out dir> [splits=test,val]

<out>/<task>/<split>.json = {task, split, count, arrays: {name: {dtype, shape, b64}}}; dtype one of f32, f64, i16, u8.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

DT = {np.dtype('float32'): 'f32', np.dtype('float64'): 'f64', np.dtype('int16'): 'i16', np.dtype('uint8'): 'u8'}


def encode(a):
    a = np.ascontiguousarray(a)
    if a.dtype not in DT:
        raise ValueError(a.dtype)
    return dict(dtype=DT[a.dtype], shape=list(a.shape), b64=base64.b64encode(a.astype(a.dtype.newbyteorder('<')).tobytes()).decode())


def main(data, out, splits='test,val'):
    for task_dir in sorted(Path(data).iterdir()):
        for sp in splits.split(','):
            f = task_dir / f'{sp}.npz'
            if not f.exists():
                continue
            d = np.load(f)
            doc = dict(task=task_dir.name, split=sp, count=int(len(d['n'])), arrays={k: encode(d[k]) for k in d.files})
            dest = Path(out) / task_dir.name / f'{sp}.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(doc, separators=(',', ':')))
            print(dest, dest.stat().st_size // 1024, 'kB')


if __name__ == '__main__':
    main(*sys.argv[1:])
