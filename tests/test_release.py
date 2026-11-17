"""Small checks that run on a laptop CPU in seconds:  python -m pytest tests

- every block initialises with the parameter layout of the released models and runs a forward pass;
- the resting blocks come to rest (their bits stop changing) on a small input;
- with WITNESS_MODELS and WITNESS_DATA set (a models folder and the official splits), a few released models reproduce
  their reference predictions on the val split.
"""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from witness_clrs import model as M
from witness_clrs.data import feature_counts, load_split, model_inputs
from witness_clrs.evaluate import predict_rest, registered_read
from witness_clrs.tasks import OUTPUTS

CFG = dict(width=128, mlp=312, edge_mlp=32)


def tiny_graph(n=6, seed=0):
    rng = np.random.default_rng(seed)
    adj = (rng.random((1, n, n)) < 0.5).astype(np.uint8)
    adj = np.maximum(adj, adj.transpose(0, 2, 1)) | np.eye(n, dtype=np.uint8)[None]
    A = (adj * rng.random((1, n, n))).astype(np.float32)
    A = np.maximum(A, A.transpose(0, 2, 1))
    return dict(n=np.array([n], np.int16), pos=(np.arange(n) / n)[None].astype(np.float32), A=A, adj=adj,
                s=np.array([0], np.int16), pi=np.zeros((1, n), np.int16))


def tiny_array(n=6, seed=0):
    rng = np.random.default_rng(seed)
    key = rng.random((1, n))
    return dict(n=np.array([n], np.int16), x_pos=(np.arange(n) / n)[None], x_key=key, pred=np.argsort(key).astype(np.int16))


@pytest.mark.parametrize('block,task,make,extra', [
    ('census', 'bellman_ford', tiny_graph, {}),
    ('census', 'bellman_ford', tiny_graph, {'weight_scale': 'graph'}),
    ('census', 'insertion_sort', tiny_array, {}),
    ('compare_first', 'insertion_sort', tiny_array, {'registers': [[1], [2]]}),
    ('pair_numbers', 'bellman_ford', tiny_graph, {'arithmetic': 'plus'}),
    ('pair_numbers', 'mst_kruskal', tiny_graph, {'arithmetic': 'max'}),
    ('cold', 'dfs', tiny_graph, {}),
    ('pair_numbers', 'floyd_warshall', tiny_graph, {'arithmetic': 'plus', 'table': 'via', 'self_loops': True}),
    ('pair_numbers', 'mst_prim', tiny_graph, {'arithmetic': 'max', 'table': 'hops'}),
])
def test_blocks_run(block, task, make, extra):
    d = make()
    if task == 'mst_kruskal':
        d = {k: v for k, v in d.items() if k not in ('s', 'pi')} | dict(in_mst=np.zeros_like(d['adj'], np.int16))
    if task == 'floyd_warshall':
        d = {k: v for k, v in d.items() if k not in ('s', 'pi')} | dict(Pi=np.zeros_like(d['adj'], np.int16))
    cfg = dict(CFG, block=block, **extra)
    outs = OUTPUTS[task]
    p = M.init_params(jax.random.PRNGKey(0), cfg, outs, feature_counts(d))
    assert 130_000 < M.count_params(p) < 200_000
    x = {k: jnp.array(v) for k, v in model_inputs(cfg, d, np.arange(1)).items()}
    lg = M.forward(p, cfg, x, 3, outs)
    assert all(np.isfinite(np.asarray(v)).all() for v in lg.values())
    if block != 'census':
        r = predict_rest(p, cfg, outs, d)
        assert r['loops'][0] >= 1


MODELS, DATA = os.environ.get('WITNESS_MODELS'), os.environ.get('WITNESS_DATA')


@pytest.mark.skipif(not (MODELS and DATA), reason='set WITNESS_MODELS and WITNESS_DATA')
@pytest.mark.parametrize('name', ['census/bfs/seed0', 'compare_first/insertion_sort/seed0', 'pair_numbers/mst_kruskal/seed0',
                                  'cold/dfs/seed0', 'pair_numbers/floyd_warshall/seed0', 'pair_numbers/mst_prim/seed0'])
def test_reference_predictions(name):
    from witness_clrs.weights import load
    p, cfg = load(Path(MODELS) / name)
    outs = [(o['name'], o['kind']) for o in cfg['outputs']]
    d = load_split(Path(DATA) / cfg['task'] / 'val.npz')
    d = {k: v[:8] for k, v in d.items()}
    pred, _ = registered_read(p, cfg['model'], outs, d)
    ref = np.load(Path(MODELS) / name / 'reference_predictions.npz')
    for k, _ in outs:
        assert np.array_equal(pred[k], ref[f'val/read/{k}'][:8]), k
