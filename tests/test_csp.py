"""The constraint-problem generators (csp/): the hand cases have their stated answers; on small sampled instances the
standard routine, an independent brute force and the answer decoded from the block's own input arrays agree; twin pairs
differ in their answer. Seconds on a laptop CPU:  python -m pytest tests/test_csp.py"""
import numpy as np
import pytest

from csp import common, parity, two_colouring, two_sat

MODULES = [two_sat, two_colouring, parity]


@pytest.mark.parametrize('m', MODULES, ids=lambda m: m.SPEC['name'])
def test_hand_cases(m):
    for name, x, want in m.HAND:
        assert m.reference(x) == want, name
        assert m.decode(x) == want, name


@pytest.mark.parametrize('m,size', [(two_sat, 6), (two_colouring, 8), (parity, [4])], ids=lambda v: getattr(v, '__name__', str(v)))
def test_sampled_instances_agree(m, size):
    rows = m.sample(12, size, common.seed_eval(m.SPEC['code'], 999))
    for x in rows:
        r = m.reference(x)
        assert m.decode(x) == r
        assert int(x['unsat'][x['s']]) == r['unsat']
        if x['meta']['nv'] <= 12:
            assert m.brute(x) == r
    labels = np.array([int(x['unsat'][x['s']]) for x in rows])
    assert 0 < labels.mean() < 1, 'both answers occur'


def test_twins_differ():
    rows = two_sat.sample(8, 12, common.seed_eval(two_sat.SPEC['code'], 998))
    pairs = {}
    for x in rows:
        if x['meta']['kind'] == 1:
            pairs.setdefault(x['meta']['pair'], []).append(int(x['unsat'][x['s']]))
    assert pairs and all(sorted(v) == [0, 1] for v in pairs.values())
