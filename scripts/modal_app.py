"""Run the release on Modal (https://modal.com): generate the data with the library, train, and check the released models.

  modal run scripts/modal_app.py::check                 # recompute every released model's predictions on an L4 GPU
  modal run scripts/modal_app.py::generate --task bfs   # the official splits and the training pools of one task
  modal run scripts/modal_app.py::generate_all          # all 30 tasks
  modal run scripts/modal_app.py::train --config configs/census/bfs.json --seed 0
  modal run scripts/modal_app.py::train_all --block census --seeds 0,1,2   # the per-task general-block table (90 runs)
  modal run scripts/modal_app.py::train_shared --config configs/shared/units_dfs.json --seed 0   # the shared block (B200)
  modal run scripts/modal_app.py::generate_csp --task two_sat               # the constraint problems' splits (csp/)
  modal run scripts/modal_app.py::train --config configs/csp/two_sat.json --seed 0 --gpu H100

The models and data the check reads are taken from release/models and release/data (scripts/download_models.py and
scripts/generate.py fill them). Data and runs live on the volume 'witness-clrs' (/vol/data, /vol/runs).
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('witness-clrs')
vol = modal.Volume.from_name('witness-clrs', create_if_missing=True)
PKG = dict(local_path=ROOT / 'witness_clrs', remote_path='/root/witness_clrs')

gen_image = (modal.Image.debian_slim(python_version='3.12').pip_install('dm-clrs==2.0.3', 'numpy==2.5.4')
             .add_local_dir(**PKG))
csp_image = modal.Image.debian_slim(python_version='3.12').pip_install('numpy==2.3.4').add_local_dir(ROOT / 'csp', '/root/csp')
gpu_image = (modal.Image.debian_slim(python_version='3.12')
             .pip_install('jax[cuda12]==0.8.0', 'optax==0.2.6', 'numpy==2.3.4', 'safetensors==0.8.0')
             .env(dict(XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONDONTWRITEBYTECODE='1'))
             .add_local_dir(**PKG).add_local_dir(ROOT / 'scripts', '/root/scripts'))
check_image = gpu_image
for sub in sorted(q.name for q in (ROOT / 'release').iterdir() if q.is_dir()) if (ROOT / 'release').exists() else []:
    check_image = check_image.add_local_dir(ROOT / 'release' / sub, f'/root/release/{sub}')


@app.function(image=gen_image, cpu=2, memory=8192, timeout=3600, volumes={'/vol': vol})
def gen_one(task, what, per_size=80000):
    import sys
    sys.path.insert(0, '/root')
    from witness_clrs import generate
    meta = generate.main('/vol/data', task, what, per_size)
    vol.commit()
    return meta


@app.function(image=gpu_image, gpu='L4', cpu=4, memory=16384, timeout=4 * 3600, volumes={'/vol': vol})
def train_one(config, seed):
    import sys
    sys.path.insert(0, '/root')
    from witness_clrs import train as T
    vol.reload()
    dest = Path('/vol/runs') / config['model']['block'] / config['task'] / f'seed{seed}'
    out = T.main(config, '/vol/data', dest, seed)
    vol.commit()
    return out


@app.function(image=gpu_image, gpu='B200', cpu=8, memory=65536, timeout=24 * 3600, volumes={'/vol': vol})
def train_shared_one(config, seed):
    """One shared block (witness_clrs.train_shared); the paper's took about 13 hours on a B200."""
    import sys
    sys.path.insert(0, '/root')
    from witness_clrs import train_shared as TS
    vol.reload()
    out = TS.main(config, '/vol/data', Path('/vol/runs') / 'shared' / config.get('name', 'shared') / f'seed{seed}', seed)
    vol.commit()
    return out


@app.function(image=csp_image, cpu=2, memory=8192, timeout=3600, volumes={'/vol': vol})
def gen_csp_one(task, split):
    import sys
    sys.path.insert(0, '/root')
    from csp import generate
    r = generate.run('/vol/data', task, [split])
    vol.commit()
    return r


@app.function(image=check_image, gpu='L4', cpu=4, memory=16384, timeout=3600)
def check_task(filt, root='models'):
    import sys
    sys.path.insert(0, '/root'); sys.path.insert(0, '/root/scripts')
    import check_release
    models = Path('/root/release') / root
    data = models / 'data' if (models / 'data').exists() else Path('/root/release/data')
    dirs = sorted(q.parent for q in models.rglob('config.json') if f'/{filt}/' in str(q))
    return [check_release.check(q, data) for q in dirs]


@app.function(image=check_image, gpu='H100', cpu=4, memory=65536, timeout=3600)
def check_task_h100(filt, root='models'):
    return check_task.local(filt, root)


@app.local_entrypoint()
def check(report: str = 'release/check_release_l4.json', tasks: str = '', root: str = 'models', gpu: str = 'L4'):
    """tasks: a comma-separated subset (default: every task in release/<root>); root: the models folder under release/;
    gpu: L4, or H100 (the constraint problems' largest graphs need its memory)."""
    tasks = tasks.split(',') if tasks else sorted({q.parent.parent.name for q in (ROOT / 'release' / root).rglob('config.json')})
    fn = check_task_h100 if gpu == 'H100' else check_task
    out = [r for rs in fn.starmap([(t, root) for t in tasks]) for r in rs]
    out.sort(key=lambda r: r['name'])
    Path(report).write_text(json.dumps(out, indent=1))
    for r in out:
        print(r['name'], 'read exact' if r['read_exact'] else 'READ DIFFERS', '| every loop exact' if r['all_loops_exact'] else '| loops differ', r['test']['score'])
    print(f"read exact: {sum(r['read_exact'] for r in out)} of {len(out)}; every loop exact: {sum(r['all_loops_exact'] for r in out)} of {len(out)}")


def sizes_for(task):
    """The training pool sizes a task needs: the library's (4, 7, 11, 13, 16; 20 for the string matchers) and every size a
    config of the task trains on (compare-first quickselect: every size from 4 to 16)."""
    from witness_clrs.generate import train_sizes
    s = set(train_sizes(task))
    for c in (ROOT / 'configs').glob(f'*/{task}.json'):
        s |= set(json.loads(c.read_text())['train']['sizes'])
    return sorted(s)


@app.function(image=check_image, gpu='L4', cpu=4, memory=16384, timeout=3600)
def check_shared_task(task, variant='', root='shared'):
    import sys
    sys.path.insert(0, '/root'); sys.path.insert(0, '/root/scripts')
    import check_release
    dirs = sorted(q.parent for q in Path(f'/root/release/{root}').rglob('config.json') if (not variant or q.parent.parent.name in variant.split(',')) and task in json.loads(q.read_text()).get('tasks', {}))
    return [r for q in dirs for r in check_release.check_shared(q, '/root/release/data', [task])]


@app.function(image=check_image, gpu='B200', cpu=4, memory=16384, timeout=3600)
def check_shared_task_b200(task, variant='', root='shared'):
    return check_shared_task.local(task, variant, root)


@app.local_entrypoint()
def check_shared(report: str = 'release/check_shared_l4.json', tasks: str = '', gpu: str = 'L4', variant: str = '', root: str = 'shared'):
    """Every shared block in release/shared (or only release/shared/<variant>): every task, against its reference
    predictions, on an L4 (or gpu='B200', the GPU the shared blocks were trained and read on)."""
    from witness_clrs.tasks import TASKS
    want = tasks.split(',') if tasks else list(TASKS)
    fn = check_shared_task_b200 if gpu == 'B200' else check_shared_task
    if root != 'shared' and not tasks:
        want = sorted({t for q in (ROOT / 'release' / root).rglob('config.json') for t in json.loads(q.read_text()).get('tasks', {})})
    out = sorted((r for rs in fn.starmap([(t, variant, root) for t in want]) for r in rs), key=lambda r: r['name'])
    Path(report).write_text(json.dumps(out, indent=1))
    for r in out:
        print(r['name'], 'read exact' if r['read_exact'] else 'READ DIFFERS', '| every loop exact' if r['all_loops_exact'] else '| loops differ', r['test']['score'])
    print(f"read exact: {sum(r['read_exact'] for r in out)} of {len(out)}; every loop exact: {sum(r['all_loops_exact'] for r in out)} of {len(out)}")


@app.local_entrypoint()
def generate(task: str, per_size: int = 80000):
    whats = ['test', 'val'] + [f'train:{L}' for L in sizes_for(task)]
    for meta in gen_one.starmap([(task, w, per_size) for w in whats]):
        print(json.dumps(meta))


@app.local_entrypoint()
def train(config: str, seed: int = 0, gpu: str = 'L4'):
    cfg = json.loads(Path(config).read_text())
    f = train_one if gpu == 'L4' else train_one.with_options(gpu=gpu)
    print(json.dumps(f.remote(cfg, seed)))


@app.local_entrypoint()
def train_shared(config: str = 'configs/shared/units_dfs.json', seed: int = 0):
    cfg = json.loads(Path(config).read_text())
    print(json.dumps(train_shared_one.remote(cfg, seed)))


@app.local_entrypoint()
def generate_csp(task: str = 'two_sat'):
    """Every split of one constraint class (csp/generate.py), one container per split."""
    import sys
    sys.path.insert(0, str(ROOT))
    from csp.generate import plan
    for r in gen_csp_one.starmap([(task, sp) for sp in plan(task)]):
        print(json.dumps(r))


@app.local_entrypoint()
def generate_all(per_size: int = 80000):
    """The official splits and the training pools of all 30 tasks."""
    from witness_clrs.tasks import TASKS
    calls = [(t, w, per_size) for t in TASKS for w in ['test', 'val'] + [f'train:{L}' for L in sizes_for(t)]]
    for meta in gen_one.starmap(calls, return_exceptions=True):
        print(json.dumps(meta) if isinstance(meta, dict) else f'ERROR {meta}')


@app.local_entrypoint()
def train_all(block: str = 'census', seeds: str = '0,1,2', tasks: str = ''):
    """Every config of a block (or the listed tasks), every seed, in parallel; prints each run's scores."""
    cfgs = sorted((ROOT / 'configs' / block).glob('*.json'))
    if tasks:
        cfgs = [c for c in cfgs if c.stem in tasks.split(',')]
    calls = [(json.loads(c.read_text()), int(s)) for c in cfgs for s in seeds.split(',')]
    for out in train_one.starmap(calls, return_exceptions=True):
        print(json.dumps(out) if isinstance(out, dict) else f'ERROR {out}')
