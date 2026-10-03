"""What the shadow desk did with news, round by round: the evidence step 5's gate waits on.

    .venv/bin/python tools/news_shadow_report.py --phase validation
    .venv/bin/python tools/news_shadow_report.py --phase validation --prices   # + each move since

Headline judgement cannot be replayed: the model has read 2016-25, and a headline names
the company that a replay's codes hide. So it is judged on live rounds only, from
Validation on. The shadow desk decides on its own paper book with headlines and filings in
front of its Risk review and Event analyst, the rule's book is what goes in, and every
call's observation and answer is kept (output/live/<phase>/<round>/shadow_calls.json).

This reads them back: one row per name a review or an analyst call could act on with news
in front of it (a headline, or a new 8-K): what woke it, what it was shown, what it
answered (hold, trim, exit) and why. With --prices it adds the name's return from that
round's fill to the latest close: the move a trim or an exit kept or gave up.

A call that failed is listed as the rule's fallback, with its error, and a shadow run on
the rule brain is labelled so: either is the rule's decision, and counted as the model's
it would credit the model with the rule's record.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import calendar, live  # noqa: E402
from icaif.agents import untrusted  # noqa: E402


def _answer(role: str, name: str, ans) -> tuple[str, str]:
    if ans is None:
        return "fallback (rule)", ""
    if role == "event":
        call = next((c for c in ans["calls"] if c["name"] == name), None)
        if call is None:
            return "no call", ""
        return call["action"] + (f" {call['fraction']}" if call.get("fraction") else ""), call["reason"]
    trims = {t["name"]: t for t in ans.get("trim", [])}
    if name in ans.get("exit", []):
        return "exit", ans["rationale"]
    if name in trims:
        return f"trim {trims[name]['fraction']} ({trims[name]['cause']})", trims[name]["why"]
    return ans["action"], ans["rationale"]


def rows(phase_dir: Path) -> pd.DataFrame:
    out = []
    for rdir in sorted(p for p in phase_dir.iterdir() if (p / "shadow_calls.json").exists()):
        rec = json.loads((rdir / "round.json").read_text()) if (rdir / "round.json").exists() else {}
        brain = (rec.get("shadow") or {}).get("brain")
        for c in json.loads((rdir / "shadow_calls.json").read_text()):
            if c["role"] not in ("review", "event"):
                continue
            p, ans = c["payload"], c.get("answer")
            woke = {t["name"]: t for t in p.get("triggers", [])}
            for r in p["names"]:
                heads, t = r.get("headlines") or [], woke.get(r["name"])
                filed = (t or {}).get("new_8k") or []
                if (not heads and not filed) or (c["role"] == "event" and t is None):
                    continue
                answer, why = _answer(c["role"], r["name"], ans)
                out.append({
                    "round": rdir.name, "execution": rec.get("execution"), "brain": brain,
                    "role": c["role"], "name": r["name"], "woken_by": (t or {}).get("why"),
                    "headlines": len(heads), "naming": sum(h["names_the_company"] for h in heads),
                    "newest_seen_h": min((h["seen_hours_ago"] for h in heads), default=None),
                    "top_headline": heads[0][untrusted.FIELD]["title"] if heads else None,
                    "new_8k": "; ".join(", ".join(f["events"]) for f in filed) or None,
                    "filing_text": any(untrusted.FIELD in f for f in filed),
                    "answer": answer, "why": (why or "")[:300], "error": c.get("error")})
    return pd.DataFrame(out)


def with_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Each row's return from its round's fill (the 30m bar's open at the execution) to the
    latest completed 30m close. Yahoo keeps about 60 days of 30m bars; older rows get none."""
    bars, _ = live.fetch_intraday(pd.Timestamp.now(tz=calendar.TZ))
    fills = live.fills(bars)
    last = bars.sort_values("end").groupby("ticker")["close"].last()
    ret = []
    for r in df.itertuples():
        ex = pd.Timestamp(r.execution) if r.execution else None
        px = fills.at[ex, r.name] if ex is not None and ex in fills.index and r.name in fills else None
        ret.append(last.get(r.name) / px - 1 if px else None)
    return df.assign(return_since_fill=ret)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--out", default=None, help="the phase's directory (default output/live/<phase>)")
    ap.add_argument("--prices", action="store_true", help="add each name's return since its fill")
    args = ap.parse_args()
    phase_dir = Path(args.out) if args.out else live.LIVE_OUT / args.phase
    df = rows(phase_dir)
    if df.empty:
        print(f"no review or analyst call in {phase_dir} had news in front of it")
        return
    if args.prices:
        df = with_prices(df)
    df.to_csv(phase_dir / "news_shadow.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df.drop(columns=["execution", "why", "error"]).to_string(index=False))
    model = df[~df["answer"].str.startswith("fallback") & (df["brain"] != "rule")]
    print(f"\n{len(df)} name-calls with news; {len(model)} answered by the model, "
          f"{len(df) - len(model)} by the rule (a fallback, or a rule shadow)")
    print(model["answer"].str.split().str[0].value_counts().to_string())
    if args.prices and len(model):
        print("\nmean return since the fill, by answer (a handful of rounds: read as anecdotes):")
        print(model.groupby(model["answer"].str.split().str[0])["return_since_fill"]
              .agg(["count", "mean"]).to_string())
    print(f"\nwrote {phase_dir / 'news_shadow.csv'}")


if __name__ == "__main__":
    main()
