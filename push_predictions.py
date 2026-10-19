#!/usr/bin/env python3
"""
Push your predictions to a PRIVATE Hugging Face repo and write the pointer manifest for Codabench.

This is the RECOMMENDED submission route. Instead of uploading ~1.9 GB to Codabench, you push the
.npz to a private Hugging Face dataset repo and submit a small manifest.json naming a pinned
commit plus a read token scoped to that one repo. The scorer pulls from Hugging Face's CDN.

Why it is worth the extra step, measured from the scoring machine:

    Codabench file storage   ~0.5 MB/s   ->  a 1.9 GB submission spends ~1 h in transfer
    Hugging Face CDN         ~50 MB/s    ->  the same predictions arrive in under a minute

The scoring worker runs one job at a time, so that hour is queue time everyone shares.

The repo stays PRIVATE -- your predictions are never visible to other teams.

The FIRST run is two steps, because Hugging Face cannot scope a token to a repo that does not
exist yet:

    # 1. create the private repo (nothing is uploaded)
    uv run python push_predictions.py --repo your-username/fusion-eq-predictions
    # 2. scope a read token to it (the script prints how), then upload for real
    uv run python push_predictions.py --repo your-username/fusion-eq-predictions --read-token hf_...

Every run after that is just step 2.

You need TWO tokens, both from https://huggingface.co/settings/tokens :

  1. A WRITE token, to upload. Log in with it once:  uv run huggingface-cli login
     It never leaves your machine. Note you can only create repos in a namespace you own --
     use your own username unless you have write access to the organization.
  2. A FINE-GRAINED READ token scoped to your predictions repo ONLY. This one goes into
     manifest.json and is submitted, so the scorer can read your private repo. Pass it with
     --read-token or set HF_READ_TOKEN.

     New token -> Fine-grained -> under "Repository permissions" pick your predictions repo
     -> tick "Read access to contents of selected repos" -> nothing else. Scoping it to your
     whole username instead works, but hands over read access to all your private repos.

This script REFUSES a write token or a classic read token for #2, because both grant far more
than the scorer needs and both are stored with your submission. Revoke the read token after the
competition.
"""
from __future__ import annotations
import argparse
import json
import os
import zipfile
import sys
from datetime import date, datetime, timezone
from pathlib import Path

CONFIG_FILENAMES = [
    "diii_d_public_test.npz", "mast_public_test.npz",
    "diii_d_private_test.npz", "mast_private_test.npz",
]

# The Development phase closed on 18 October 2026 (00:00 UTC). From then on only the Final files
# (*_private_test.npz) are scored, and a Final submission holding only Development files scores
# nothing. Several teams lost a submission that way on the Final's first day.
DEV_CLOSED = date(2026, 10, 18)


def development_closed() -> bool:
    return datetime.now(timezone.utc).date() >= DEV_CLOSED

TOKEN_HELP = (
    "Create one at https://huggingface.co/settings/tokens -> New token -> Fine-grained,\n"
    "       grant ONLY 'Read access to contents of selected repos' on your predictions repo."
)


def validate_read_token(api, token: str, repo: str) -> int:
    """Refuse anything broader than fine-grained read. Returns 0 if acceptable."""
    try:
        who = api.whoami(token=token)
    except Exception as exc:
        print(f"ERROR: the read token was rejected by Hugging Face ({type(exc).__name__}).\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1

    at = (who.get("auth") or {}).get("accessToken") or {}
    role = at.get("role")

    if role in ("write", "admin"):
        print(f"ERROR: that is a {role.upper()} token. It would be stored with your submission and\n"
              f"       could modify your repos. Use a read-only fine-grained token instead.\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1
    if role == "read":
        print("ERROR: that is a classic read token. It grants read access to ALL of your private\n"
              "       repos, not just your predictions, and it is stored with your submission.\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1
    if role == "fineGrained":
        scoped = (at.get("fineGrained") or {}).get("scoped") or []
        perms = [p for s in scoped for p in (s.get("permissions") or [])]
        bad = sorted({p for p in perms if "write" in p or "admin" in p})
        if bad:
            print(f"ERROR: that fine-grained token grants {bad}. Read access is all the scorer\n"
                  f"       needs, and the token is stored with your submission.\n"
                  f"       {TOKEN_HELP}", file=sys.stderr)
            return 1
        names = {(s.get("entity") or {}).get("name") for s in scoped}
        namespace = repo.split("/")[0]
        if repo in names:
            if len(scoped) > 1:
                print(f"NOTE: the token is scoped to {len(scoped)} entities; just {repo} "
                      "would be enough.")
        elif namespace in names:
            # Scoped to the whole user/org rather than one repo. It works, but it hands the
            # organizers read access to every private repo in that namespace -- the same
            # over-reach we reject a classic read token for. Warn, do not block: it is their
            # call, and failing here would be worse than the over-sharing.
            print(f"WARNING: the token is scoped to the whole '{namespace}' namespace, so it can\n"
                  f"         read ALL your private repos there -- not just {repo}. It is stored\n"
                  f"         with your submission. Consider re-scoping it to {repo} alone.")
        else:
            print(f"WARNING: the token is scoped to {sorted(n for n in names if n)}, which does not\n"
                  f"         obviously include {repo}. Scoring will fail if it cannot read that repo.")
        return 0

    print(f"NOTE: could not determine the token's scope (role={role!r}). Continuing -- but please\n"
          "      confirm it is a fine-grained, read-only token for your predictions repo.")
    return 0


def push_and_write_pointer(repo: str, read_token: str | None, sub_dir: Path,
                           out: Path, dry_run: bool = False,
                           allow_development: bool = False) -> int:
    """Upload sub_dir/*.npz to `repo` and write the pointer zip. Returns a process exit code.

    submission_skeleton.py calls this directly so one command can build and submit; keeping it a
    function rather than duplicating the logic means the token checks cannot drift between the
    two entry points.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    npz = sorted(p for p in sub_dir.glob("*.npz") if p.name in CONFIG_FILENAMES)
    if not npz:
        print(f"ERROR: no recognised .npz in {sub_dir.resolve()}.\n"
              f"       Expected one or more of: {', '.join(CONFIG_FILENAMES)}\n"
              f"       Build them first: python submission_skeleton.py --out {sub_dir} --max-shots 0",
              file=sys.stderr)
        return 1

    final = [p for p in npz if p.name.endswith("_private_test.npz")]
    dev = [p for p in npz if p.name.endswith("_public_test.npz")]
    if not final and development_closed() and not allow_development:
        print(f"ERROR: {sub_dir.resolve()} holds only Development files "
              f"({', '.join(p.name for p in dev)}).\n"
              "       The Development phase closed on 18 October. A Final submission needs\n"
              "       diii_d_private_test.npz and/or mast_private_test.npz; Development files are\n"
              "       not scored in the Final. Build the Final files with:\n"
              f"           python submission_skeleton.py --final --max-shots 0 --out {sub_dir} "
              f"--repo {repo} --read-token hf_...\n"
              "       (To push Development files anyway, add --development.)", file=sys.stderr)
        return 1
    if final and dev:
        print(f"Pushing only the Final files; ignoring {', '.join(p.name for p in dev)} "
              f"in {sub_dir}, and removing any *_public_test.npz from {repo} in this commit.")
        npz = final

    total = sum(p.stat().st_size for p in npz)
    print(f"Found {len(npz)} file(s) in {sub_dir.resolve()}  ({total / 1e6:.0f} MB total):")
    for p in npz:
        print(f"    {p.name:28s} {p.stat().st_size / 1e6:8.0f} MB")

    api = HfApi()
    if read_token:
        if validate_read_token(api, read_token, repo):
            return 1
        print("Read token accepted (fine-grained, read-only).")

    if dry_run:
        print(f"\n--dry-run: would create {repo} (PRIVATE dataset) and upload the files above.")
        return 0

    who = api.whoami()
    print(f"\nLogged in as {who['name']}. Creating/reusing PRIVATE dataset repo {repo}...")
    try:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    except Exception as exc:
        ns = repo.split("/")[0]
        hint = ""
        if "403" in str(exc) or "rights" in str(exc).lower():
            hint = (f"\n       Your login is '{who['name']}'. If '{ns}' is an organization, you need\n"
                    f"       write access to it -- or just use your own namespace:\n"
                    f"           --repo {who['name']}/{repo.split('/')[-1]}")
        print(f"ERROR: could not create {repo} ({type(exc).__name__}).{hint}\n\n{exc}",
              file=sys.stderr)
        return 1

    # A fine-grained token cannot be scoped to a repo that does not exist yet, so the first run
    # is deliberately two steps: create the repo, then scope a token to it and re-run.
    if not read_token:
        print(f"\nCreated {repo}. Now scope a read token to it and re-run:\n"
              f"\n  1. https://huggingface.co/settings/tokens -> New token -> Fine-grained\n"
              f"  2. Under 'Repository permissions', select {repo}\n"
              f"  3. Tick ONLY 'Read access to contents of selected repos'\n"
              f"  4. re-run with --repo {repo} --read-token hf_...\n"
              f"\nNothing was uploaded yet. The token is what lets the scorer read your private\n"
              f"predictions; it goes into manifest.json and is submitted to Codabench.")
        return 0

    # exist_ok=True does NOT flip an existing public repo to private. Say so loudly rather than
    # leaving a team to discover their predictions were readable all along.
    info = api.repo_info(repo, repo_type="dataset")
    if not getattr(info, "private", True):
        print(f"\nWARNING: {repo} already existed and is PUBLIC. Your predictions will be\n"
              f"         visible to other teams. Scoring still works, but consider making it\n"
              f"         private in the repo settings, or using a fresh --repo name.\n")

    print("Uploading (this is the slow part, but it only happens once per submission)...")
    # Upload exactly the files listed above. With Final files, also delete any Development files
    # left in the repo by earlier pushes: the scorer downloads every recognised file at the pinned
    # commit, so stale *_public_test.npz only cost transfer time on the one shared worker.
    commit = api.upload_folder(folder_path=str(sub_dir), repo_id=repo, repo_type="dataset",
                               allow_patterns=[p.name for p in npz],
                               delete_patterns=["*_public_test.npz"] if final else None)
    sha = commit.oid
    if not (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha.lower())):
        print(f"ERROR: expected a 40-character commit SHA, got {sha!r}", file=sys.stderr)
        return 1

    # Verify with the READ token exactly what the scorer will see, before a submission slot is
    # spent on it. This catches a token scoped to the wrong repo, which is the likely mistake.
    try:
        at_rev = api.repo_info(repo, revision=sha, repo_type="dataset", token=read_token)
    except RepositoryNotFoundError:
        print(f"ERROR: the upload succeeded, but the READ token cannot see {repo}.\n"
              f"       Re-scope it to that repo -- otherwise scoring will fail.\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1
    root = {s.rfilename for s in at_rev.siblings if "/" not in s.rfilename}
    missing = [p.name for p in npz if p.name not in root]
    if missing:
        print(f"ERROR: uploaded, but these are not at the repo ROOT at {sha}: {missing}\n"
              f"       Files found at root: {sorted(root)}", file=sys.stderr)
        return 1

    manifest = {"repo_id": repo, "revision": sha, "token": read_token}
    mpath = sub_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    # zipfile, not a `zip` subprocess: the CLI is absent on plenty of Linux installs and on
    # essentially every Windows one. "w" truncates, so re-running never appends a stale manifest.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(mpath, arcname="manifest.json")   # must sit at the archive ROOT

    print(f"\nPushed {len(npz)} file(s) at commit {sha}")
    print(f"Verified with the read token, at the repo root: "
          f"{', '.join(sorted(p.name for p in npz))}")
    print(f"\n==> Upload this to Codabench:  {out.resolve()}  ({out.stat().st_size} bytes)")
    print(json.dumps({**manifest, "token": "hf_***redacted***"}, indent=2))
    print(f"\nThat zip IS the whole submission -- the scorer pulls your predictions from {repo}\n"
          f"at the pinned commit. {mpath} now holds your read token: it is meant to be submitted,\n"
          "but do not commit it to a public git repo, and revoke it after the competition.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="Hugging Face dataset repo, e.g. your-username/fusion-eq-predictions")
    ap.add_argument("--read-token", default=os.environ.get("HF_READ_TOKEN"),
                    help="fine-grained READ token scoped to --repo (or set HF_READ_TOKEN)")
    ap.add_argument("--dir", type=Path, default=Path("submission"),
                    help="directory holding the .npz (default: submission)")
    ap.add_argument("--out", type=Path, default=Path("submission_pointer.zip"),
                    help="pointer zip to upload to Codabench (default: submission_pointer.zip)")
    ap.add_argument("--dry-run", action="store_true", help="check everything, upload nothing")
    ap.add_argument("--development", action="store_true",
                    help="allow pushing Development files (*_public_test.npz) alone after the "
                         "Development phase closed")
    args = ap.parse_args()
    return push_and_write_pointer(args.repo, args.read_token, args.dir, args.out, args.dry_run,
                                  args.development)


if __name__ == "__main__":
    raise SystemExit(main())
