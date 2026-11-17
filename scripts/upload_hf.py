"""Upload the models folder to a Hugging Face model repository, created private if it does not exist.

  python scripts/upload_hf.py <models dir> <repo id> [--allow-public]

The token is read by huggingface_hub from its usual place (HF_TOKEN or the cached login); it is never printed.
A repository that already exists keeps its visibility. A public repository is refused unless --allow-public is given
for that one upload; with it, files are only added or replaced, never deleted.
"""
import sys

from huggingface_hub import HfApi


def main(models, repo, *flags):
    allow_public = '--allow-public' in flags
    api = HfApi()
    api.create_repo(repo, repo_type='model', private=True, exist_ok=True)
    info = api.model_info(repo)
    print(repo, 'private' if info.private else 'PUBLIC')
    if not info.private and not allow_public:
        raise SystemExit(f'{repo} is public: not uploading (pass --allow-public for an upload meant to be public)')
    api.upload_folder(folder_path=models, repo_id=repo, repo_type='model', commit_message='Upload trained models',
                      delete_patterns=None if not info.private else ['*/seed*/*'])
    files = api.list_repo_files(repo)
    print(len(files), 'files in', repo, '(private)' if api.model_info(repo).private else '(PUBLIC)')


if __name__ == '__main__':
    main(*sys.argv[1:])
