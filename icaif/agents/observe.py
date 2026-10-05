"""What an agent sees: a point-in-time, JSON-able snapshot built from daily closes.

The desk hands this module only closes a decision may see (`quant_strategies.
daily_closes` cuts them at the deadline), so nothing here can look ahead by itself;
the test that rewrites the future and requires the observation unchanged guards the
seam anyway.

**Anonymised mode** exists because the model has read 2016-25 market history. Shown
"NVDA, 2024-05-20", it can recall what happened next, and a replay that scores well
on memory is not evidence the agent can judge. So in replays each window gets its own
random name codes (S01-S30, sorted by code, so even the alphabetical order of real
tickers is gone), dates become "day k of 15", no price level appears, macro levels
become z-scores and changes (`macro.readings`), and headlines, which name companies,
are dropped. 8-K events stay as item labels: "director or officer change" names no
one, while the filing's own text would, so that is live only too. Only windows after the
model's training cutoff can be replayed with real names and still count. Our own
signals (HAR vols, the score's rank, sessions to earnings) are numbers about a code and
stay too.

**The universe ranking is context** (`universe_context`): the daily model's rank and
percentile for every name in its training universe that day, the 30 among them under
their own codes and flagged tradeable. In replays the other ~70 names get codes of their
own (`UniverseCodes`, U01-U99): drawn at random when a name is first seen in the window
and kept for the rest of it, so a code says nothing about the ticker, its alphabetical
place, or which names will join the universe later in the window. No sector, index
membership or entry date is shown for any of them: those would date the window.

**Headlines are for held names, in the roles that act on them** (the Risk review and the
Event analyst), and capped: a triggered name's newest few in full, any other held name's
titles that name the company. Every name's whole feed went into every role before:
55,000 of the first live entry observation's 65,000 characters, most of them stories
about other companies. External text is cleaned, capped and quoted in `source_text`
(`untrusted.py`).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from icaif import filings as F, quant, quant_strategies as qs
from icaif.agents import untrusted

ANNUAL = np.sqrt(252)
# External text where it is shown: a headline's title and summary, a filing's own words.
TITLE_CHARS, SUMMARY_CHARS, FILING_CHARS = 160, 240, 1200
# Headlines a triggered name shows in full, and titles naming the company for any other
# held name; first seen within HEADLINE_HOURS (a Monday review still sees Friday's).
HEADLINES_TRIGGERED, HEADLINES_HELD, HEADLINE_HOURS = 4, 2, 72


class Anonymizer:
    """A per-window bijection between tickers and codes; identity when disabled."""

    def __init__(self, tickers: list[str], seed: Optional[int]):
        self.enabled = seed is not None
        if self.enabled:
            order = np.random.default_rng(seed).permutation(len(tickers))
            codes = [f"S{i + 1:02d}" for i in range(len(tickers))]
            self.to_code = {tickers[j]: codes[i] for i, j in enumerate(order)}
        else:
            self.to_code = {t: t for t in tickers}
        self.to_ticker = {c: t for t, c in self.to_code.items()}

    @classmethod
    def from_mapping(cls, to_code: dict) -> "Anonymizer":
        """The same bijection back from `to_code`, for a desk restored between rounds."""
        anon = cls(list(to_code), None)
        anon.to_code = dict(to_code)
        anon.to_ticker = {c: t for t, c in anon.to_code.items()}
        anon.enabled = any(t != c for t, c in anon.to_code.items())
        return anon

    def code(self, ticker: str) -> str:
        return self.to_code[ticker]

    def ticker(self, code: str) -> str:
        """Raises KeyError on a code the agent invented, which the desk treats as an
        invalid answer: a guessed mapping would sell a name the agent never named."""
        return self.to_ticker[code]


class UniverseCodes:
    """Codes for the universe names outside the tradeable 30; real tickers when disabled.

    A name's code is drawn from a shuffled pool the first time the window shows it, so it
    is stable for the window and carries nothing from the ticker. Codes were not assigned
    up front for every name of the window: that would need the window's later universes,
    and the number of codes would say how many names join before it ends.
    """

    POOL = 99

    def __init__(self, seed: Optional[int], tradeable: list[str]):
        self.enabled = seed is not None
        self.tradeable = set(tradeable)
        self.pool = ([f"U{i:02d}" for i in np.random.default_rng([seed, 1]).permutation(
            np.arange(1, self.POOL + 1))] if self.enabled else [])
        self.to_code: dict = {}

    def code(self, ticker: str) -> str:
        if not self.enabled:
            return ticker
        if ticker not in self.to_code:
            n = len(self.to_code)
            # Past the pool (more than 99 names in one window; the most seen is 80), the
            # codes run on in order rather than reusing one.
            self.to_code[ticker] = self.pool[n] if n < len(self.pool) else f"U{n + 1}"
        return self.to_code[ticker]

    def is_universe_name(self, name: str) -> bool:
        """A name the universe block has shown that is not one of the 30."""
        return name in self.to_code.values() if self.enabled else name in self.to_code

    def state(self) -> dict:
        return {"enabled": self.enabled, "pool": self.pool, "to_code": dict(self.to_code)}

    @classmethod
    def from_state(cls, state: dict, tradeable: list[str]) -> "UniverseCodes":
        u = cls(None, tradeable)
        u.enabled, u.pool, u.to_code = bool(state["enabled"]), list(state["pool"]), dict(state["to_code"])
        return u


def universe_block(ranks, anon: "Anonymizer", ucodes: UniverseCodes) -> dict:
    """`UniverseScores.for_day` as a role reads it: one row per name, best first.

    Rows are lists under `columns` rather than one object each: about 100 rows a day, and
    keys repeated 100 times would double the block. Live, `ucodes` is disabled and keeps
    the real tickers it showed, so the desk can still refuse a lever naming one.
    """
    rows = []
    for t, r in ranks.iterrows():
        if r["tradeable"]:
            name = anon.code(t)
        else:
            name = ucodes.code(t)
            if not ucodes.enabled:
                ucodes.to_code[t] = t
        rows.append([name, int(r["rank"]), round(float(r["percentile"]), 3), bool(r["tradeable"])])
    return {"note": "Context only: the daily model's ranking of its whole training universe "
                    "today. Only the names in `names` can be traded.",
            "names_ranked": len(rows),
            "columns": ["name", "rank", "percentile", "tradeable"],
            "rows": rows}


@dataclass
class BookState:
    weights: pd.Series            # current weights, every ticker
    nav: list[float]              # NAV at each round 1 so far, first = 1,000,000
    traded_notional: float = 0.0  # to date, for the turnover the agent has spent
    entered: bool = False


@dataclass
class Readings:
    """The market-level numbers the rule and the agents both read (unrounded)."""

    returns: pd.DataFrame = field(repr=False)
    basket: pd.Series = field(repr=False)
    ewma_vol: float
    vol_ratio: float
    p_turbulent_next: float
    hmm: Optional[quant.HMM2]


def readings(closes: pd.DataFrame, hmm: Optional[quant.HMM2]) -> Readings:
    rets = np.log(closes).diff().iloc[1:]
    basket = qs._basket(rets)
    b = basket.dropna()
    ewma = np.sqrt((b ** 2).ewm(alpha=0.06).mean()) if len(b) else pd.Series(dtype=float)
    vol = float(ewma.iloc[-1]) if len(ewma) else np.nan
    ratio = float(vol / ewma.median()) if len(ewma) > 60 else np.nan
    p = quant.hmm_next_turbulent(b.to_numpy()[-250:], hmm) if hmm is not None and len(b) else np.nan
    return Readings(rets, basket, vol, ratio, p, hmm)


def fit_regime(closes: pd.DataFrame) -> Optional[quant.HMM2]:
    """The HMM fit a window uses throughout: on the history before its first round."""
    b = qs._basket(np.log(closes).diff().iloc[1:]).dropna().to_numpy()
    return quant.fit_hmm2(b) if len(b) >= 250 else None


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _hours(deadline, ts) -> float:
    return round((pd.Timestamp(deadline) - pd.Timestamp(ts)).total_seconds() / 3600, 1)


def headline_rows(rows: pd.DataFrame, deadline, full: bool) -> list[dict]:
    """`news.recent` rows as a role reads them: when we first had each (`seen_hours_ago`,
    the clock that counts), what the publisher dated it, whether it names the company,
    and its text, quoted. `full` adds the summary."""
    out = []
    for r in rows.itertuples():
        text = {"title": untrusted.clean(r.title, TITLE_CHARS)}
        if full:
            text["summary"] = untrusted.clean(r.summary, SUMMARY_CHARS)
        out.append({"seen_hours_ago": _hours(deadline, r.first_seen),
                    # Yahoo's pubDate can run after our first fetch; never "in the future".
                    "published_hours_ago": max(0.0, _hours(deadline, r.published)),
                    "names_the_company": bool(r.names_it), untrusted.FIELD: text})
    return out


def filing_rows(rows: pd.DataFrame, deadline, real: bool) -> list[dict]:
    """New 8-Ks as the Event analyst reads them: item labels (ours), hours since
    acceptance, and live only, the filing's own words, quoted and capped."""
    out = []
    for r in rows.sort_values("accepted", ascending=False).itertuples():
        row = {"hours_ago": _hours(deadline, r.accepted), "events": F.labels(r.items),
               "amendment": bool(r.amended)}
        text = getattr(r, "text", None)
        if real and isinstance(text, str) and text.strip():
            row[untrusted.FIELD] = untrusted.clean(text, FILING_CHARS)
        out.append(row)
    return out


def _signal(field_: str, value):
    """A rank is a count (1 = best), the rest are vols rounded like vol_ann_20d."""
    if field_.endswith("_rank"):
        return None if value is None or not np.isfinite(value) else int(value)
    return _r(value)


def observation(closes: pd.DataFrame, rd: Readings, book: BookState, anon: Anonymizer, *,
                day: int, window_days: int, round_no: int,
                signals: Optional[dict] = None,
                at_entry: Optional[dict] = None,
                previews: Optional[dict] = None,
                earnings: Optional[dict] = None,
                news: Optional[dict] = None,
                macro: Optional[dict] = None,
                filings: Optional[dict] = None,
                calendar_date: Optional[str] = None,
                positions: Optional[dict] = None,
                universe: Optional[dict] = None) -> dict:
    """`signals`: today's {"names": {ticker: {field: value}}, "market": {...}} from
    `Desk._signals`; `at_entry`: the same fields as they stood on the entry day, shown
    with an `_at_entry` suffix so a change since entry is a comparison the agent reads,
    not one it must remember; `previews`: {key: weights} shown as `weight_if_<key>`,
    the exact books a decision would buy; `positions`: the journal's fields for each
    held name (`Journal.name_fields`: entry day, gain since entry and its peak);
    `universe`: `universe_block`, the whole universe's ranking, shown as context;
    `news`: {ticker: `headline_rows`} for the names whose headlines this role reads
    (never shown anonymised); `filings`: `filings.recent` per name."""
    tickers = list(closes.columns)
    rets = rd.returns
    tail = rets.tail(qs.SHAPE_DAYS)
    shapes = {}
    for name in ("inverse_vol", "risk_parity"):
        s = qs.SHAPES[name](tail)
        shapes[name] = s.reindex(tickers) if s is not None else pd.Series(np.nan, index=tickers)
    cov = quant.shrunk_cov(tail)
    ou = quant.s_scores(tail) if len(tail) >= 20 else pd.DataFrame(index=tickers, columns=["s"])
    sig_names = (signals or {}).get("names", {})
    entry_names = (at_entry or {}).get("names", {})

    nav = np.array(book.nav, dtype=float)
    peak = nav.max() if len(nav) else np.nan
    names = []
    for t in tickers:
        r = rets[t]
        row = {
            "name": anon.code(t),
            "vol_ann_20d": _r(r.tail(20).std() * ANNUAL),
            "ret_1d": _r(r.iloc[-1] if len(r) else np.nan),
            "ret_5d": _r(r.tail(5).sum()),
            "ret_20d": _r(r.tail(20).sum()),
            "weight_now": _r(book.weights.get(t, 0.0)),
            "weight_if_inverse_vol": _r(shapes["inverse_vol"][t]),
            "weight_if_risk_parity": _r(shapes["risk_parity"][t]),
            "ou_s_score": _r(ou["s"].get(t, np.nan) if "s" in ou else np.nan, 2),
        }
        for key, w in (previews or {}).items():
            row[f"weight_if_{key}"] = _r(w.get(t, np.nan))
        for field_, value in sig_names.get(t, {}).items():
            row[field_] = _signal(field_, value)
        for field_, value in entry_names.get(t, {}).items():
            row[f"{field_}_at_entry"] = _signal(field_, value)
        if earnings is not None:
            row["earnings_in_sessions"] = earnings.get(t)
        if filings is not None:
            row["recent_8k_filings"] = filings.get(t, [])
        if news is not None and not anon.enabled and t in news:
            row["headlines"] = news[t]
        if positions and t in positions:
            row.update(positions[t])
        names.append(row)
    names.sort(key=lambda x: x["name"])

    avg_corr = None
    if cov is not None:
        sd = np.sqrt(np.diag(cov.to_numpy()))
        corr = cov.to_numpy() / np.outer(sd, sd)
        avg_corr = _r(corr[np.triu_indices_from(corr, 1)].mean())
    hmm = rd.hmm
    obs = {
        "clock": {"day": day, "of": window_days, "round": round_no,
                  "sessions_left_after_today": window_days - day},
        "book": {
            "return_to_date": _r(nav[-1] / nav[0] - 1 if len(nav) else 0.0),
            "drawdown_from_peak": _r(1 - nav[-1] / peak if len(nav) else 0.0),
            "gross": _r(float(book.weights.sum())),
            "entered": book.entered,
            "turnover_spent": _r(book.traded_notional / nav[0] if len(nav) else 0.0),
        },
        "market": {
            "basket_ret_1d": _r(rd.basket.iloc[-1] if len(rd.basket) else np.nan),
            "basket_ret_5d": _r(rd.basket.tail(5).sum()),
            "basket_ret_20d": _r(rd.basket.tail(20).sum()),
            "basket_vol_ann_ewma": _r(rd.ewma_vol * ANNUAL),
            "vol_vs_3y_median": _r(rd.vol_ratio, 3),
            "p_turbulent_next_session": _r(rd.p_turbulent_next, 6),
            "regime_persistence_days": (None if hmm is None else
                                        {"calm": _r(hmm.persistence[0], 1),
                                         "turbulent": _r(hmm.persistence[1], 1)}),
            "avg_pairwise_corr_60d": avg_corr,
            **{k: _signal(k, v) for k, v in (signals or {}).get("market", {}).items()},
            **{f"{k}_at_entry": _signal(k, v) for k, v in (at_entry or {}).get("market", {}).items()},
        },
        "names": names,
    }
    if macro is not None:
        obs["macro"] = macro
    if universe is not None:
        obs["universe_context"] = universe
    if calendar_date and not anon.enabled:
        obs["clock"]["date"] = calendar_date
    return obs
