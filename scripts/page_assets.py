"""Fetch the demo page's models from the public Hugging Face repositories, at pinned revisions, and write them into the
page's static folder as scripts/site_assets.py does from a local copy. The Pages workflow runs it before the build.

  python scripts/page_assets.py <site static dir>

The per-task models (general block, cold blocks; seed 0, every seed's scores) come from cold-attention-clrs30; the
shared general block's per-task models from cold-attention-clrs30-shared (shared-units-dfs, seed 0, every seed's scores).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import site_assets  # noqa: E402

PER_TASK = ('mohsenvand/cold-attention-clrs30', 'ac9e618459e38622f6a21c64242411f0af46e884')
SHARED = ('mohsenvand/cold-attention-clrs30-shared', 'efcc353e1bd73f8134e7b96f9eac841a52b6e42c')


def main(static):
    from huggingface_hub import snapshot_download
    repo, rev = PER_TASK
    per_task = snapshot_download(repo, revision=rev, allow_patterns=['*/seed*/config.json', '*/seed0/model.safetensors',
                                                                     '*/seed0/reference_predictions.npz'])
    site_assets.main(per_task, static, '0')
    repo, rev = SHARED
    shared = snapshot_download(repo, revision=rev, allow_patterns=['shared-units-dfs/seed*/config.json',
                                                                   'shared-units-dfs/seed0/model.safetensors',
                                                                   'shared-units-dfs/seed0/reference_predictions.npz'])
    site_assets.shared(Path(shared) / 'shared-units-dfs' / 'seed0', static)


if __name__ == '__main__':
    main(*sys.argv[1:])
