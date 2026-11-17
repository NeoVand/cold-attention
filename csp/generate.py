"""Write the splits of one constraint class in the block's graph layout (common.py) -> <out>/<task>/<split>.npz, with a
manifest <split>.json holding the arrays' sha256 (common.digest).

  python -m csp.generate <out dir> <task> [split ...]      (task: two_sat, two_colouring, parity; no split: all)

Splits per class:
  two_sat:        train_L<nv> for nv 2..9 (main pool, padded to 19 items) and big_L<nv> for nv 10..16 (big pool, padded to
                  33; drawn with share 7/15, so the sizes are uniform over 2..16), 16,000 each; val nv 24 (256); test nv 64
                  (256); n256 (128); n500 (128, 1,001 items). Twins make up half of every size where the depth rule can hold
                  (nv >= 9), deep twins (chain >= nv / 2) in val and test.
  two_colouring:  train_L<nv> for nv 3..18 (padded to 19) and big_L<nv> for 19..32 (padded to 33), 8,000 each; val 48; test
                  64; n256; n500 (501 items).
  parity:         train_L4, train_L6 (main pool, padded to 76 items), big_L8 (big pool, 101 items, share 1/3), 16,000 each;
                  val 10 vertices (64); test 16 vertices (64); n8, n12 (64 each); cube, petersen (32 each); fit = the first
                  32 instances of each training file (96, scored as "fits its training instances").
"""
import json
import sys
import time
from pathlib import Path

from . import common, parity, two_colouring, two_sat

MOD = {'two_sat': two_sat, 'two_colouring': two_colouring, 'parity': parity}


def plan(task):
    """{split: (function () -> rows, M)}"""
    m = MOD[task]; code = m.SPEC['code']; it = m.SPEC['items']; P = {}
    if task in ('two_sat', 'two_colouring'):
        sizes = range(2, 17) if task == 'two_sat' else range(3, 33)
        cut = 9 if task == 'two_sat' else 18
        per = 16000 if task == 'two_sat' else 8000
        for nv in sizes:
            name = f'train_L{nv}' if nv <= cut else f'big_L{nv}'
            P[name] = ((lambda nv=nv: m.sample(per, nv, common.seed_train(code, nv, 0))), it(cut) if nv <= cut else it(max(sizes)))
        ev = dict(val=(24 if task == 'two_sat' else 48, 256, 1), test=(64, 256, 2), n256=(256, 128, 3), n500=(500, 128, 4))
        for name, (nv, num, tag) in ev.items():
            P[name] = ((lambda nv=nv, num=num, tag=tag: m.sample(num, nv, common.seed_eval(code, tag), deep=True)), it(nv))
    else:
        for nvert, name in ((4, 'train_L4'), (6, 'train_L6'), (8, 'big_L8')):
            P[name] = ((lambda nvert=nvert: m.sample(16000, [nvert], common.seed_train(code, nvert, 0))), it(6) if nvert < 8 else it(8))
        for name, (sizes, num, tag) in dict(val=([10], 64, 1), test=([16], 64, 2), n8=([8], 64, 3), n12=([12], 64, 4)).items():
            P[name] = ((lambda sizes=sizes, num=num, tag=tag: m.sample(num, sizes, common.seed_eval(code, tag))), it(max(sizes)))
        for name, tag in (('cube', 5), ('petersen', 6)):
            P[name] = ((lambda name=name, tag=tag: m.sample(32, None, common.seed_eval(code, tag), graph=name)), it(8 if name == 'cube' else 10))
        P['fit'] = (lambda: [r for nvert in (4, 6, 8) for r in m.sample(16000, [nvert], common.seed_train(code, nvert, 0))[:32]], it(8))
    return P


def run(out, task, splits=None):
    P = plan(task); res = {}
    for name in (splits or list(P)):
        t0 = time.time(); f, M = P[name]; rows = f()
        meta = common.write(Path(out) / task, name, rows, M, dict(task=task, seconds=None))
        meta['seconds'] = round(time.time() - t0, 1); (Path(out) / task / f'{name}.json').write_text(json.dumps(meta))
        res[name] = dict(meta['label_stats'], sha256=meta['sha256'], M=M, seconds=meta['seconds'])
        print(task, name, json.dumps(res[name]), flush=True)
    return res


if __name__ == '__main__':
    out, task = sys.argv[1], sys.argv[2]
    r = run(out, task, sys.argv[3:] or None)
    if not sys.argv[3:]:
        (Path(out) / task / 'summary.json').write_text(json.dumps(r, indent=1))
