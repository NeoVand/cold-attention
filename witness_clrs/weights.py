"""Saving and loading a trained block: model.safetensors (parameters, float32) and config.json.

Parameter names are paths joined by '/', e.g. 'blk/a', 'in30/e1', 'one30/min'.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def flatten(p, prefix=''):
    out = {}
    for k, v in p.items():
        if isinstance(v, dict):
            out |= flatten(v, f'{prefix}{k}/')
        else:
            out[f'{prefix}{k}'] = np.asarray(v, np.float32)
    return out


def unflatten(flat):
    p = {}
    for k, v in flat.items():
        *path, leaf = k.split('/')
        q = p
        for part in path:
            q = q.setdefault(part, {})
        q[leaf] = jnp.array(v)
    return p


def save(directory, params, config):
    from safetensors.numpy import save_file
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    save_file(flatten(params), str(d / 'model.safetensors'), metadata={'format': 'np'})
    (d / 'config.json').write_text(json.dumps(config, indent=1) + '\n')


def load(directory):
    """-> (params, config) of a saved block (a local directory)."""
    from safetensors.numpy import load_file
    d = Path(directory)
    return unflatten(load_file(str(d / 'model.safetensors'))), json.loads((d / 'config.json').read_text())


def load_hub(repo_id, subfolder, revision=None, token=None):
    """-> (params, config) from a Hugging Face model repo, e.g. load_hub(REPO, 'census/bfs/seed0')."""
    from huggingface_hub import hf_hub_download
    paths = [Path(hf_hub_download(repo_id, f'{subfolder}/{f}', revision=revision, token=token)) for f in ('config.json', 'model.safetensors')]
    return load(paths[1].parent)


def shared_task(params, config, task):
    """A shared block (one block trained on many tasks, each with its own input encoder and read-outs) -> (params, config)
    of one task's model: the shared 'blk' with the task's own parameters (and, for two blocks, the second block with the
    task's own second parameters), and the task's config, ready for the forward and its read (after 8 loops, unless the
    task names its own read in its model config)."""
    tasks = config['tasks']
    if task not in tasks:
        raise KeyError(f'{task} is not one of the shared block\'s tasks: {sorted(tasks)}')
    p = dict(params['task'][task])
    p['blk'] = params['blk']
    if 'second' in params:   # two blocks: the second block with the task's own second embedding and read-outs, and the map between them
        p['second'] = dict(params['second']['task'][task], blk=params['second']['blk'])
        p['couple'] = params['couple']
    tc = tasks[task]
    cfg = dict(name=f"{config['name']}/{task}", task=task, task_name=tc['task_name'], block=config['block'],
               model=dict(config['model'], **tc.get('model', {})), outputs=tc['outputs'], features=tc['features'],
               read=tc.get('model', {}).get('read', config['read']), scores=tc.get('scores', {}))
    return p, cfg


def load_shared(directory, task):
    """-> (params, config) of one task's model from a saved shared block (a local directory)."""
    return shared_task(*load(directory), task)


def load_shared_hub(repo_id, subfolder, task, revision=None, token=None):
    """-> (params, config) of one task's model from a shared block on Hugging Face, e.g.
    load_shared_hub(REPO, 'shared/seed0', 'dijkstra')."""
    from huggingface_hub import hf_hub_download
    paths = [Path(hf_hub_download(repo_id, f'{subfolder}/{f}', revision=revision, token=token)) for f in ('config.json', 'model.safetensors')]
    return load_shared(paths[1].parent, task)
