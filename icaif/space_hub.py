"""The holdout's three HuggingFace repos: deploying the pages and submitting entries.

Each repo has one job and a designed visibility, checked on every entry write:

| repo | visibility | holds |
| --- | --- | --- |
| Space MO-AI-Inv/icaif2026-holdout | private | the scorer page and its 2026 price file |
| dataset MO-AI-Inv/icaif2026-holdout-entries | private | the record of every submission |
| Space MO-AI-Inv/icaif2026-leaderboard | public | the board page, references, entry copies |

A deploy (`publish`) goes through whatever a Space's visibility is, by the owner's
decision of 2026-10-06: the owner sets visibility on HF and a deploy never changes it. It
says what it found, loudly when that differs from the design, so a scorer left public is
named on every deploy rather than discovered. Entry writes still require the designed
visibility: the dataset is the record and must not be public by accident.

The board is public so that viewing needs no login. It holds only what it shows: the
ranking code, results and notes. Prices stay in the private scorer, because they are
Alpaca's data and not ours to redistribute. The board also serves the only entry copy
the office network lets a page read: Netskope blocks authenticated HF downloads
(/resolve and /raw on private repos), and a Space's own host gets through.

A submit writes the dataset first, then the board. If the second write fails, the
entry is on record but not shown, and `sync` (off the office network) copies it across.
The two never disagree the other way.

No writer may undo another's work:
- a deploy (`publish`) mirrors a built page but never touches `entries/`;
- a submit adds one entry file and rewrites that repo's `entries/index.json`. A static
  page cannot list files, so it reads this index. The index is rebuilt from the repo's
  file list (an API call the proxy allows), never downloaded and edited. The commit
  pins its parent, so if someone else submitted in between, HF refuses it with 412
  and we retry on the new head. Two concurrent submits cannot drop each other's entry.

The scorer page submits the same way from the browser (space/index.html), using its
SUBMIT_TOKEN variable. This module is not shipped to either Space.
"""

import json
import sys
import time

REPO_ID = "MO-AI-Inv/icaif2026-holdout"
BOARD_ID = "MO-AI-Inv/icaif2026-leaderboard"
DATASET_ID = "MO-AI-Inv/icaif2026-holdout-entries"
ENTRIES = "entries/"
INDEX = ENTRIES + "index.json"


def _api():
    import truststore
    truststore.inject_into_ssl()   # the Netskope proxy re-signs TLS; macOS verifies
    from huggingface_hub import HfApi
    return HfApi()


def _require(api, repo_id: str, repo_type: str, visibility: str):
    """The repo must have the visibility it was designed with, or nothing is written."""
    info = api.repo_info(repo_id, repo_type=repo_type)
    if info.private != (visibility == "private"):
        sys.exit(f"{repo_type} {repo_id} is {'private' if info.private else 'PUBLIC'}, "
                 f"expected {visibility}; not writing to it")
    return info


def whoami() -> str:
    return _api().whoami()["name"]


def _visibility(api, repo_id: str, repo_type: str, designed: str) -> str:
    """The repo's actual visibility, with a warning when it is not the designed one."""
    actual = "private" if api.repo_info(repo_id, repo_type=repo_type).private else "public"
    if actual != designed:
        print(f"WARNING: {repo_type} {repo_id} is {actual.upper()}, designed {designed}; "
              "deploying anyway (visibility is set on HF, never by a deploy)")
    return actual


def publish(out, repo_id: str = REPO_ID, visibility: str = "private") -> None:
    """Mirror a built page into its Space, whatever the Space's visibility (see above).

    `visibility` is the designed one: used when the Space must be created, and compared
    against, never enforced.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from huggingface_hub.utils import RepositoryNotFoundError

    api = _api()
    try:
        _visibility(api, repo_id, "space", visibility)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="space", space_sdk="static",
                        private=visibility == "private")
    local = {p.relative_to(out).as_posix(): p for p in out.rglob("*") if p.is_file()}
    if any(k.startswith(ENTRIES) for k in local):
        sys.exit("the build must not contain entries/: a deploy would overwrite submissions")
    remote = api.list_repo_files(repo_id, repo_type="space")
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in local.items()]
    ops += [CommitOperationDelete(path_in_repo=k) for k in remote
            if k not in local and not k.startswith(ENTRIES) and k != ".gitattributes"]
    api.create_commit(repo_id, repo_type="space", operations=ops,
                      commit_message="Rebuild the page from icaif2026")
    actual = _visibility(api, repo_id, "space", visibility)
    print(f"deployed https://huggingface.co/spaces/{repo_id} ({actual}); "
          f"{sum(1 for k in remote if k.startswith(ENTRIES) and k != INDEX)} entries kept")


def _commit_entry(api, repo_id: str, repo_type: str, visibility: str, path: str,
                  body: bytes, title: str, attempts: int = 4) -> None:
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.utils import HfHubHTTPError

    for attempt in range(attempts):
        head = _require(api, repo_id, repo_type, visibility).sha
        files = api.list_repo_files(repo_id, repo_type=repo_type, revision=head)
        if path in files:
            return   # already there: a retry after a partial failure, or a sync
        listed = sorted({f for f in files if f.startswith(ENTRIES) and f != INDEX} | {path})
        ops = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=body),
               CommitOperationAdd(path_in_repo=INDEX,
                                  path_or_fileobj=json.dumps({"entries": listed}).encode())]
        try:
            api.create_commit(repo_id, repo_type=repo_type, operations=ops,
                              parent_commit=head, commit_message=title)
            return
        except HfHubHTTPError as err:
            # 412 (seen) or 409: someone committed since `head`. Rebuild on the new head.
            if getattr(err.response, "status_code", None) not in (409, 412) or attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)


def entry_path(entry: dict) -> str:
    from icaif.leaderboard import slug
    stamp = entry["submitted_at"].replace(":", "").replace("-", "")
    return f"{ENTRIES}{slug(entry['strategy'])}/{stamp}.json"


def submit(entry: dict, board_id: str = BOARD_ID, dataset_id: str = DATASET_ID) -> str:
    """Record the entry in the dataset, then copy it to the public board. Returns its path."""
    api = _api()
    path = entry_path(entry)
    body = json.dumps(entry, indent=1).encode()
    title = f"Submit {entry['strategy']} ({entry['author']})"
    _commit_entry(api, dataset_id, "dataset", "private", path, body, title)
    try:
        _commit_entry(api, board_id, "space", "public", path, body, title)
    except Exception as err:
        sys.exit(f"recorded in {dataset_id} but NOT on the board ({err}). Off the office "
                 "network, run: .venv/bin/python tools/build_holdout_space.py --sync")
    return path


def sync(board_id: str = BOARD_ID, dataset_id: str = DATASET_ID) -> list[str]:
    """Copy dataset entries missing from the board. Needs a network that allows downloads."""
    api = _api()
    have = set(api.list_repo_files(board_id, repo_type="space"))
    copied = []
    for path in api.list_repo_files(dataset_id, repo_type="dataset"):
        if not path.startswith(ENTRIES) or path == INDEX or path in have:
            continue
        local = api.hf_hub_download(dataset_id, path, repo_type="dataset")
        with open(local, "rb") as f:
            _commit_entry(api, board_id, "space", "public", path, f.read(),
                          f"Sync {path} from the dataset")
        copied.append(path)
    return copied
