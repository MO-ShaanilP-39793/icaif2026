"""A recorded agent run with code holding cash outside it: the overlay, applied after the fact.

    .venv/bin/python tools/overlay_replay.py v1_free_gemini --overlay 0.1 0.5 0.01 0.001

v3 asks one entry and holds it, so its book at another overlay is the same book bought at
another gross (`chain_replay.py stage1`). A desk that trades every morning (v1's free desk)
is not: its later decisions read its own book. Under the overlay as v3 defines it, the
desk never sees the cash code holds: it reads its own book, decides as it did, and code
buys `g` times every position it chose. So the decisions are the recorded ones, and no
model is asked again.

Two passes per window, both on the market the run was scored on:
1. **The record, replayed**: each morning's logged target submitted as it was. Its metrics
   must equal the run's own `windows.csv` row, or the log is not what was traded (a
   compile step, a fallback, a weight floored differently) and the scaled run would be
   scaled from a book that never existed. It refuses rather than guesses.
2. **Scaled**: at each logged rebalance, code submits `g * w * V / U`: the desk's weight
   times its NAV `V` (pass 1, at the same deadline) over the scaled book's NAV `U`. That
   buys `g` times the desk's shares, since a weight is a share of the book it is
   submitted to; `g * w` alone would size every position off the wrong NAV. Both NAVs are
   read at the deadline, as a live overlay would have to; execution moves them by the
   same factor to first order.

Writes `output/agent/<run>_ov<pct>_<window>/windows.csv` and `.out` for
`tools/submit_agentic.py`. The spend line reports no calls and no cost: each logged
decision stands in for a call, counted as a cache hit.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from icaif import baselines, data, markets, sim, windows  # noqa: E402
from icaif import weights as W  # noqa: E402
from entry_replay import HOLD, metrics_row  # noqa: E402
from icaif.replay_entry import DESK_PREFIXES  # noqa: E402

AGENT = data.ROOT / "output" / "agent"
METRICS = ("cumulative_return", "sharpe_ratio", "maximum_drawdown", "turnover")
TOL = 1e-9


def _nav(ctx) -> float:
    bars = ctx.recent_closes(1)
    px = bars.iloc[-1] if len(bars) else pd.Series(dtype=float)
    held = {t: s for t, s in ctx.shares.items() if abs(s) > 1e-12}
    return float(ctx.cash) + float(sum(s * px[t] for t, s in held.items()))


class Recorded:
    """Each logged morning's target, submitted as it was; NAV at each deadline recorded."""

    def __init__(self, targets: dict, tickers):
        self.targets, self.tickers, self.navs = targets, tickers, {}

    def __call__(self, ctx):
        key = (ctx.day, ctx.round)
        self.navs[key] = _nav(ctx)
        w = self.targets.get(key)
        return None if w is None else W.safe(w, self.tickers)


class Scaled:
    def __init__(self, targets: dict, tickers, g: float, navs: dict):
        self.targets, self.tickers, self.g, self.navs = targets, tickers, g, navs

    def __call__(self, ctx):
        key = (ctx.day, ctx.round)
        w = self.targets.get(key)
        if w is None:
            return None
        ratio = self.navs[key] / _nav(ctx)
        return W.safe({t: self.g * v * ratio for t, v in w.items()}, self.tickers)


def targets_of(log: pd.DataFrame, market, start) -> dict:
    """{(day, round): {ticker: weight}} for every logged rebalance; holds submit nothing."""
    days = market.days[market.days.index(start):market.days.index(start) + windows.WINDOW_DAYS]
    out = {}
    for r in log.itertuples():
        d = r.decision or {}
        if d.get("action") != "rebalance":
            continue
        w = {t: 0.0 for t in market.tickers}
        for x in d.get("weights") or []:
            if x["name"] not in w:
                raise SystemExit(f"{x['name']} is not a ticker: the log is anonymised, and a replay "
                                 "of it needs the run's code map")
            w[x["name"]] = float(x["weight"])
        out[(days[int(r.day) - 1], int(r.round))] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="the board entry's run prefix: output/agent/<run>_<window>")
    ap.add_argument("--overlay", type=float, nargs="+", required=True)
    args = ap.parse_args()
    market = markets.research_market()
    tickers = list(market.tickers)
    for path in sorted(AGENT.glob(f"{args.run}_20*")):
        if not path.is_dir():
            continue
        start = pd.Timestamp(re.search(r"(\d{4}-\d{2}-\d{2})$", path.name).group(1)).date()
        log = pd.read_json(path / "log.jsonl", lines=True)
        orig = pd.read_csv(path / "windows.csv")
        desk = [s for s in orig["strategy"].unique() if s.startswith(DESK_PREFIXES)]
        if len(desk) != 1:
            raise SystemExit(f"{path.name}: expected one desk row, found {desk}")
        want = orig[orig["strategy"] == desk[0]].iloc[0]
        tg = targets_of(log, market, start)
        rec = Recorded(tg, tickers)
        got = sim.run(rec, market, start, windows.WINDOW_DAYS).metrics()
        off = {m: (got[m], float(want[m])) for m in METRICS if abs(got[m] - float(want[m])) > TOL}
        if off:
            raise SystemExit(f"{path.name}: the logged targets do not reproduce the run: {off}")
        out_text = (path.parent / f"{path.name}.out").read_text()
        spend = re.search(r"^(\S+): \d+ calls", out_text, re.M).group(1)
        falls = int((log["source"] == "fallback").sum())
        hold = baselines.scaled(baselines.InverseVolHold, 0.75)
        for g in args.overlay:
            res = sim.run(Scaled(tg, tickers, g, rec.navs), market, start, windows.WINDOW_DAYS)
            pct = f"{g * 100:g}".replace(".", "p")
            tag = f"{args.run}_ov{pct}_{start}"
            name = f"{desk[0]}_ov{pct}"
            rows = [metrics_row(res, start, name), metrics_row(sim.run(hold(), market, start, windows.WINDOW_DAYS),
                                                               start, HOLD)]
            (AGENT / tag).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(AGENT / tag / "windows.csv", index=False)
            (AGENT / f"{tag}.out").write_text(
                f"{path.name} at overlay {g:g}: the logged decisions replayed, scaled by code\n"
                f"{spend}: 0 calls, $0.00; cache hits {len(log)}, misses 0\n"
                f"fallbacks: {'none' if not falls else f'logged {falls}'}\n")
            m = res.metrics()
            gross = [sum(v for v in w.values()) for w in tg.values()]
            print(f"{tag}: return {m['cumulative_return']:+.5f}, sharpe {m['sharpe_ratio']:.3f} "
                  f"(desk {got['sharpe_ratio']:.3f}), mdd {m['maximum_drawdown']:.5f}, turnover {m['turnover']:.5f} "
                  f"(desk {got['turnover']:.5f}); max gross target {g * max(gross or [0]):.4f}")
        print(f"{path.name}: the logged targets reproduce the run exactly ({len(tg)} rebalances)")


if __name__ == "__main__":
    main()
