# Upload this prepared repository

The local package has an initialized Git repository on branch `main`, with no commit, no remote and no push. It contains only final-paper V2 experiments; all file hashes in SHA256SUMS describe the release payload before future edits.

After reviewing the contents, run from the package root:

```bash
git status --short
python analysis/verify_package.py
git add .
git diff --cached --stat
git commit -m "Package final path-unified V2 experiments and CPU reproduction"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your actual repository. The remote should be empty; do not force-push over existing history. If Git asks for an author identity, configure your own `user.name` and `user.email`; none was invented during packaging. No token belongs in this directory or in the remote URL.

The companion source tarball excludes `.git`. After extracting the tarball, initialize Git first:

```bash
git init -b main
```

`.gitignore` excludes model weights, environments, local outputs and credentials. Compressed JSONL evidence under `data/` is intentional and should be committed. This prepared snapshot has no file above 20 MiB; it does not require Git LFS for the included payload. Large model/prompt/landscape artifacts are listed separately in `provenance/EXTERNAL_ARTIFACTS.json`.

`analysis/recompute.py` is the portable CPU entrypoint. Archived GPU scripts remain evidence of the actual experiments and require external model/data/path configuration; do not advertise them as a ready-to-run GPU release.
