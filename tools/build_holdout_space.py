"""Build the holdout harness as a static page, and optionally upload it to a private HF Space.

    .venv/bin/python tools/build_holdout_space.py                 # build output/space/
    .venv/bin/python tools/build_holdout_space.py --push          # build, then upload

The Space is an allowlist, not a copy of the repo. It ships the harness, the ledger,
the calendar and the organizers' validator and metric calculator, plus a 2026-only
price file. Nothing else goes: no models, no features, no strategy code, no organizer
panel. `icaif/` is mostly strategy code, and one stray import would carry it into the
upload. So the build imports the harness from the built folder alone and refuses to
finish if anything outside the allowlist loaded.

The page runs the harness in the browser through Pyodide (space/worker.js), so the
Space is static: HF hosts Gradio Spaces only on a paid plan. The build scores the
template natively through the page's own entry point (space/webapp.py) and requires
the CLI's numbers, so the only step not checked here is Pyodide itself.

--push refuses a Space that exists and is public. The upload mirrors the folder, so
files dropped from the allowlist are deleted remotely too.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))

import webapp  # noqa: E402
from icaif import calendar, data, holdout, markets  # noqa: E402

REPO_ID = "MO-AI-Inv/icaif2026-holdout"
# Static because a Gradio Space needs a paid plan: Team/Enterprise for the org, PRO for
# a personal account. Both refused with HF 402 on 2026-09-30.
ICAIF_MODULES = ["__init__", "calendar", "data", "holdout", "kit", "ranking", "sim", "windows"]
KIT_FILES = ["kit/__init__.py", "kit/config.py", "kit/contracts.py", "kit/evaluation.py",
             "universe.json"]
SPACE_FILES = ["index.html", "worker.js", "webapp.py", "README.md"]
# What the worker writes into Pyodide's filesystem; the page's own HTML/JS is not.
PY_FILES = ["webapp.py", *(f"icaif/{m}.py" for m in ICAIF_MODULES),
            *(f"starter-kit/{f}" for f in KIT_FILES),
            "data/exec_prices.csv", "data/closes.csv", "data/market.json"]
# From just before the holdout, so a Space scoring a start in early January has fills;
# the organizer panel (licensed to participants) is never an input here.
PRICES_FROM = "2025-12-01"


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    (out / "icaif").mkdir(parents=True)
    (out / "starter-kit" / "kit").mkdir(parents=True)
    (out / "data").mkdir()
    for m in ICAIF_MODULES:
        shutil.copy2(data.ROOT / "icaif" / f"{m}.py", out / "icaif" / f"{m}.py")
    for f in KIT_FILES:
        shutil.copy2(data.ROOT / "starter-kit" / f, out / "starter-kit" / f)
    for f in SPACE_FILES:
        shutil.copy2(data.ROOT / "space" / f, out / f)

    market = markets.research_market("alpaca")
    since = market.exec_prices.index >= PRICES_FROM
    webapp.write_frame(market.exec_prices[since], out / "data" / "exec_prices.csv")
    webapp.write_frame(market.closes[market.closes.index >= PRICES_FROM],
                       out / "data" / "closes.csv")
    snapshot = sorted((data.ROOT / "data" / "public").glob("alpaca_30m_2*.parquet"))[-1].name
    days = [d for d in market.days if str(d) >= PRICES_FROM]
    meta = {"snapshot": snapshot, "first_day": str(days[0]), "last_day": str(days[-1]),
            "degraded_days": [d for d in market.issues.get("degraded_days", [])
                              if d >= PRICES_FROM]}
    (out / "data" / "market.json").write_text(json.dumps(meta, indent=1))
    missing = [f for f in PY_FILES if not (out / f).exists()]
    if missing:
        sys.exit(f"manifest names files the build did not write: {missing}")
    (out / "manifest.json").write_text(json.dumps({"files": PY_FILES, "meta": meta}, indent=1))
    return meta


def check(out: Path) -> None:
    """Import the harness from the built folder only, and score a file with it.

    Run in a subprocess with the repo off sys.path: if a copied module imported one
    that was not copied, this is where it fails, not on the Space after upload.
    """
    probe = f"""
import sys
sys.path = [p for p in sys.path if p not in ("", {str(data.ROOT)!r}, {str(data.ROOT / "tools")!r})]
sys.path.insert(0, {str(out)!r})
import json, pandas as pd
import webapp
from icaif import holdout
loaded = sorted(m for m in sys.modules if m.startswith("icaif"))
outside = [m for m, mod in sys.modules.items() if m.startswith(("icaif", "kit"))
           and getattr(mod, "__file__", None) and not mod.__file__.startswith({str(out)!r})]
assert not outside, outside
allowed = {{"icaif"}} | {{"icaif." + m for m in {ICAIF_MODULES!r} if m != "__init__"}}
assert set(loaded) <= allowed, set(loaded) - allowed
print(json.dumps(loaded))
"""
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       cwd=out)
    if r.returncode:
        sys.exit(f"built Space does not import cleanly:\n{r.stderr}")
    # The probe's bytecode would otherwise be uploaded with the source.
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache)
    print(f"imports from the build only: {r.stdout.strip()}")


def parity(out: Path) -> None:
    """The page's entry point, on the shipped CSVs, must score a file as the CLI does.

    A file that rebalances only at rounds 1 and 4 exercises holds as well as trades.
    """
    full = markets.research_market("alpaca")
    webapp._MARKET = webapp.load_market(out)
    trimmed = webapp._MARKET[0]
    if not trimmed.exec_prices.equals(full.exec_prices.loc[trimmed.exec_prices.index]):
        sys.exit("shipped fill prices are not bit-identical to the full market's")
    w = {t: 1 / 40 for t in full.tickers}
    cash = 1 - sum(w.values())
    days = holdout.span_days(full, holdout.HOLDOUT_START, holdout.HOLDOUT_END)
    rows = [{"round_id": holdout.round_id(d, r["round"]), "cash": cash, "weights": w}
            for d in days for r in calendar.rounds_for(d) if r["round"] in (1, 4)]
    probe = out.parent / "space_parity.json"
    probe.write_text(json.dumps({"strategy": "parity", "decisions": rows}))
    dec = holdout.load_decisions(probe, full)
    a, _ = holdout.continuous(dec, full)
    a_roll, _ = holdout.rolling(dec, full)
    page = json.loads(webapp.score(str(probe), str(holdout.HOLDOUT_START),
                                   str(holdout.HOLDOUT_END), False, "pre_fee"))
    probe.unlink()
    b_roll = page["windows"]
    # Prices are bitwise equal; the last digits differ only because numpy's dot product
    # sums in an order that depends on memory layout. Anything above 1e-12 is data.
    close = lambda x, y: abs(x - y) <= 1e-12 * max(1.0, abs(x))  # noqa: E731
    diff = {k: (a[k], page["continuous"][k]) for k in holdout.METRICS
            if not close(a[k], page["continuous"][k])}
    if len(b_roll) != len(a_roll):
        diff["windows"] = (len(a_roll), len(b_roll))
    else:
        diff.update({f"window {w['window_start']} {k}": (row[k], w[k])
                     for (_, row), w in zip(a_roll.iterrows(), b_roll)
                     for k in holdout.METRICS if not close(row[k], w[k])})
    if diff:
        sys.exit(f"the page's entry point scores differently from the CLI: {diff}")
    print(f"page entry point matches the CLI: continuous run and {len(b_roll)} windows (to 1e-12)")


def push(out: Path, repo_id: str) -> None:
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.repo_info(repo_id, repo_type="space")
        if not info.private:
            sys.exit(f"{repo_id} exists and is PUBLIC; refusing to upload prices to it")
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="space", space_sdk="static", private=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="space",
                      delete_patterns="*", commit_message="Rebuild from icaif2026")
    if not api.repo_info(repo_id, repo_type="space").private:
        sys.exit(f"{repo_id} is public after upload; make it private now")
    print(f"uploaded to https://huggingface.co/spaces/{repo_id} (private)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=data.ROOT / "output" / "space")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id", default=REPO_ID)
    args = ap.parse_args()

    meta = build(args.out)
    print(f"built {args.out}: prices {meta['first_day']}..{meta['last_day']} "
          f"from {meta['snapshot']}")
    check(args.out)
    parity(args.out)
    if args.push:
        push(args.out, args.repo_id)


if __name__ == "__main__":
    main()
