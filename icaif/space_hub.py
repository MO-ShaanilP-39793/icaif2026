"""The private holdout Space and its entry dataset on HuggingFace.

Entries live in two places, and each place has one job:
- the dataset MO-AI-Inv/icaif2026-holdout-entries is the record of every submission;
- the Space holds a copy under `entries/`, because it is the only place the office
  network lets the page read. Netskope blocks authenticated downloads from HF
  (/resolve and /raw on private repos return its "Noncompliant action" page), and
  only the Space's own host gets through.

A submit writes the dataset first, then the Space. If the second write fails, the
entry is on record but not on the board, and `sync` (off the office network) copies
it across. The two never disagree the other way.

Each writer must not undo the others:
- a deploy (`publish`) mirrors the built page but never touches `entries/`;
- a submit adds one entry file and rewrites that repo's `entries/index.json`. A static
  page cannot list files, so it reads this index. The index is rebuilt from the repo's
  file list (an API call the proxy allows), never downloaded and edited. The commit
  pins its parent, so if someone else submitted in between, HF refuses it and we
  retry on the new head. Two concurrent submits cannot drop each other's entry.

The page submits the same way from the browser, with the viewer's own HF sign-in
(space/index.html). This module is not shipped to the Space: it needs a write token.
"""

import json
import sys
import time

REPO_ID = "MO-AI-Inv/icaif2026-holdout"
DATASET_ID = "MO-AI-Inv/icaif2026-holdout-entries"
ENTRIES = "entries/"
INDEX = ENTRIES + "index.json"


def _api():
    import truststore
    truststore.inject_into_ssl()   # the Netskope proxy re-signs TLS; macOS verifies
    from huggingface_hub import HfApi
    return HfApi()


def _require_private(api, repo_id):
    return _require_private_repo(api, repo_id, "space")


def whoami() -> str:
    return _api().whoami()["name"]


def publish(out, repo_id: str = REPO_ID) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from huggingface_hub.utils import RepositoryNotFoundError

    api = _api()
    try:
        _require_private(api, repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="space", space_sdk="static", private=True)
    local = {p.relative_to(out).as_posix(): p for p in out.rglob("*") if p.is_file()}
    if any(k.startswith(ENTRIES) for k in local):
        sys.exit("the build must not contain entries/: a deploy would overwrite submissions")
    remote = api.list_repo_files(repo_id, repo_type="space")
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in local.items()]
    ops += [CommitOperationDelete(path_in_repo=k) for k in remote
            if k not in local and not k.startswith(ENTRIES) and k != ".gitattributes"]
    api.create_commit(repo_id, repo_type="space", operations=ops,
                      commit_message="Rebuild the page from icaif2026")
    _require_private(api, repo_id)
    print(f"deployed https://huggingface.co/spaces/{repo_id} (private); "
          f"{sum(1 for k in remote if k.startswith(ENTRIES) and k != INDEX)} entries kept")


def _require_private_repo(api, repo_id, repo_type):
    info = api.repo_info(repo_id, repo_type=repo_type)
    if not info.private:
        sys.exit(f"{repo_type} {repo_id} is PUBLIC; refusing to write results to it")
    return info


def _commit_entry(api, repo_id: str, repo_type: str, path: str, body: bytes,
                  title: str, attempts: int = 4) -> None:
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.utils import HfHubHTTPError

    for attempt in range(attempts):
        head = _require_private_repo(api, repo_id, repo_type).sha
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
            # 409/412: someone committed since `head`. Rebuild the index on the new head.
            if getattr(err.response, "status_code", None) not in (409, 412) or attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)


def entry_path(entry: dict) -> str:
    from icaif.leaderboard import slug
    stamp = entry["submitted_at"].replace(":", "").replace("-", "")
    return f"{ENTRIES}{slug(entry['strategy'])}/{stamp}.json"


def submit(entry: dict, repo_id: str = REPO_ID, dataset_id: str = DATASET_ID) -> str:
    """Record the entry in the dataset, then copy it to the Space. Returns its path."""
    api = _api()
    path = entry_path(entry)
    body = json.dumps(entry, indent=1).encode()
    title = f"Submit {entry['strategy']} ({entry['author']})"
    _commit_entry(api, dataset_id, "dataset", path, body, title)
    try:
        _commit_entry(api, repo_id, "space", path, body, title)
    except Exception as err:
        sys.exit(f"recorded in {dataset_id} but NOT on the board ({err}). Off the office "
                 "network, run: .venv/bin/python tools/build_holdout_space.py --sync")
    return path


def sync(repo_id: str = REPO_ID, dataset_id: str = DATASET_ID) -> list[str]:
    """Copy dataset entries missing from the Space. Needs a network that allows downloads."""
    api = _api()
    have = set(api.list_repo_files(repo_id, repo_type="space"))
    copied = []
    for path in api.list_repo_files(dataset_id, repo_type="dataset"):
        if not path.startswith(ENTRIES) or path == INDEX or path in have:
            continue
        local = api.hf_hub_download(dataset_id, path, repo_type="dataset")
        with open(local, "rb") as f:
            _commit_entry(api, repo_id, "space", path, f.read(), f"Sync {path} from the dataset")
        copied.append(path)
    return copied
