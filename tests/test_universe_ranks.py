"""The daily model's ranking of its whole universe, shown as context beside the 30."""

import json

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, live, quant_strategies as qs, sim
from icaif.agents import signals
from icaif.agents.brains import RuleBrain
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.schemas import (EntryDecision, EventDecision, Exclusion, NameCall,
                                  ReviewDecision, Trim)
from tests.test_agents import Scripted
from tests.test_live import FakePredictor, NOW, world  # noqa: F401 - the fixture
from tests.test_signals import DAYS, I0, START, _world
from tests.test_sim import TICKERS, _market

# Ticker-shaped names outside the 30, so a leak would show as the real thing would.
OTHERS = [f"Q{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(40)]
JOINER = "ZJNR"   # enters the universe only on day 3 of the window


def _universe(seed=0, later_sign=1.0, joiner_from=None, cut=START) -> pd.Series:
    """(date, ticker) scores for the 30 and OTHERS every day; JOINER from `joiner_from`.
    `later_sign` flips every score dated after `cut`."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in DAYS:
        names = TICKERS + OTHERS + ([JOINER] if joiner_from and d >= joiner_from else [])
        vals = rng.normal(size=len(names)) * (later_sign if d > cut else 1.0)
        rows += [(pd.Timestamp(d), t, v) for t, v in zip(names, vals)]
    return pd.DataFrame(rows, columns=["date", "ticker", "pred"]).set_index(["date", "ticker"])["pred"]


def _kw(**universe_kw):
    m, kw = _world()
    kw["universe_scores"] = signals.UniverseScores(_universe(**universe_kw), TICKERS)
    return m, kw


def _payloads(cfg=None, n=3, brain=None, **universe_kw):
    m, kw = _kw(**universe_kw)
    b = brain or Scripted()
    d = Desk(b, cfg, **kw)
    sim.run(d, m, START, n)
    return d, b.seen


# ----------------------------------------------------------------------------- shown, as context

def test_every_desk_role_reads_the_whole_universe_ranked_with_only_the_30_marked_tradeable():
    """Present but missing names, or the 30 unmarked, would show the agent a ranking it
    could misread as a menu: every universe name ranked once, the 30 under the codes
    `names` uses, and the rank among the 30 still beside it as the primary signal."""
    d, seen = _payloads(DeskConfig(anonymize=True))
    for role, p in seen:
        u = p["universe_context"]
        assert u["columns"] == ["name", "rank", "percentile", "tradeable"], role
        rows = u["rows"]
        assert u["names_ranked"] == len(rows) == len(TICKERS) + len(OTHERS)
        assert [r[1] for r in rows] == list(range(1, len(rows) + 1))
        assert rows[0][2] == 1.0 and all(a[2] > b[2] for a, b in zip(rows, rows[1:]))
        tradeable = {r[0] for r in rows if r[3]}
        assert tradeable == {n["name"] for n in p["names"]}
        assert all(r[0].startswith("U") for r in rows if not r[3])
        assert sorted(n["model_score_rank"] for n in p["names"]) == list(range(1, 31))
    assert {r for r, _ in seen} >= {"entry", "review"}


def test_the_free_desk_arms_are_not_shown_the_universe():
    """The free desk compares what two prompts are given; a new input in both changes
    the experiment it was built to run."""
    m, kw = _kw()
    from icaif.agents.free import FreeDesk
    b = Scripted()
    sim.run(FreeDesk(b, "blank", **kw), m, START, 1)
    assert b.seen and "universe_context" not in b.seen[0][1]


def test_a_rule_desk_reading_the_universe_ranking_trades_exactly_as_the_backtested_candidate():
    """The ranking is context for an LLM. Had it reached the rule's path, every LLM-vs-rule
    comparison would measure that leak."""
    m, kw = _kw()
    ref = sim.run(qs.CANDIDATES["q_riskparity_entry_regime"](), m, START, 5)
    got = sim.run(Desk(RuleBrain(), None, **kw), m, START, 5)
    pd.testing.assert_frame_equal(got.ledger, ref.ledger)


# ----------------------------------------------------------------------------- point in time

@pytest.mark.parametrize("anonymize", [False, True])
def test_the_universe_ranks_through_day_2_are_unchanged_when_later_scores_and_members_are_rewritten(anonymize):
    """A row served a day late ranks the universe on prices the decision has not seen.
    And a code assigned up front for every name in the window would shift when a name
    joins later, so earlier rounds would know how many names were still to come."""
    cfg = DeskConfig(anonymize=anonymize)
    joiner_day = DAYS[I0 + 2]
    _, a = _payloads(cfg, later_sign=1.0)
    _, b = _payloads(cfg, later_sign=-1.0, joiner_from=joiner_day, cut=DAYS[I0 + 1])
    early = [(r, p) for r, p in a if p["clock"]["day"] <= 2]
    assert early
    assert [json.dumps(p, sort_keys=True) for _, p in early] == \
        [json.dumps(p, sort_keys=True) for _, p in b[:len(early)]]
    third = next(p for _, p in b if p["clock"]["day"] == 3)
    assert len(third["universe_context"]["rows"]) == len(TICKERS) + len(OTHERS) + 1


def test_a_universe_row_asked_for_on_another_day_raises_rather_than_serving_it():
    u = signals.UniverseScores(_universe(), TICKERS)
    with pytest.raises(Exception, match="requested at a deadline"):
        u.for_day(DAYS[I0 + 1], calendar.at(START, calendar.ROUNDS[1][0]))


# ----------------------------------------------------------------------------- anonymised

def test_replayed_universe_names_carry_no_ticker_and_keep_one_code_for_the_window():
    """A real ticker among the ~70 would let the model recall the window; a code that
    changed between days would make one name read as two, and its rank moves as noise."""
    d, seen = _payloads(DeskConfig(anonymize=True), n=4, joiner_from=DAYS[I0 + 2])
    text = json.dumps([p for _, p in seen])
    assert not [t for t in TICKERS + OTHERS + [JOINER] if f'"{t}"' in text]
    assert "2026-" not in text
    codes = d.ucodes.to_code
    assert len(set(codes.values())) == len(codes) == len(OTHERS) + 1
    scores = _universe(joiner_from=DAYS[I0 + 2])
    for _, p in seen:
        if p["clock"]["round"] != 1:
            continue
        day = DAYS[I0 + p["clock"]["day"] - 1]
        want = scores.xs(pd.Timestamp(day)).rank(ascending=False, method="min")
        got = {r[0]: r[1] for r in p["universe_context"]["rows"] if not r[3]}
        assert got == {codes[t]: int(want[t]) for t in codes if t in want.index}


def test_another_window_draws_other_codes_for_the_same_names():
    a, _ = _payloads(DeskConfig(anonymize=True), n=1)
    b, _ = _payloads(DeskConfig(anonymize=True, seed=1), n=1)
    assert a.ucodes.to_code.keys() == b.ucodes.to_code.keys()
    assert a.ucodes.to_code != b.ucodes.to_code


def test_an_anonymised_desk_restored_before_every_round_keeps_its_universe_codes():
    """Each live round is its own process; a desk that redrew its codes would show the
    same name under a new code each morning."""
    m, kw = _kw(joiner_from=DAYS[I0 + 2])
    cfg = DeskConfig(anonymize=True)
    whole, again = Scripted(), Scripted()
    sim.run(Desk(whole, cfg, **kw), m, START, 3)
    saved = {"state": None}

    def restarted(ctx):
        d = Desk(again, cfg, **kw)
        d.restore(json.loads(saved["state"]) if saved["state"] else None, ctx.market.tickers)
        out = d(ctx)
        saved["state"] = json.dumps(d.state())
        return out

    sim.run(restarted, m, START, 3)
    assert [json.dumps(p, sort_keys=True) for _, p in again.seen] == \
        [json.dumps(p, sort_keys=True) for _, p in whole.seen]


# ----------------------------------------------------------------------------- refused levers

def _avoid(name):
    return lambda p: EntryDecision(shape="risk_parity", views="none", exposure=0.6, rationale="x",
                                   avoid=[Exclusion(name=name, signal="model_score", why="x")])


def _exit(name):
    return lambda p: ReviewDecision(action="hold", exposure=None, reason=None, exit=[name],
                                    trim=[], rationale="x")


def _trim(name):
    return lambda p: ReviewDecision(action="hold", exposure=None, reason=None, exit=[],
                                    trim=[Trim(name=name, fraction="half", cause="news", why="x")],
                                    rationale="x")


def _universe_name(p, anonymize=True):
    return next(r[0] for r in p["universe_context"]["rows"] if not r[3])


@pytest.mark.parametrize("anonymize", [True, False])
@pytest.mark.parametrize("role, lever", [("entry", _avoid), ("review", _exit), ("review", _trim)])
def test_a_lever_naming_a_universe_name_is_refused_and_the_rule_trades(anonymize, role, lever):
    """The ~70 names are shown, never tradeable. An answer naming one falls back whole to
    the rule, so the book is the rule's trade for trade, and the log says why."""
    m, kw = _kw()
    b = Scripted(**{role: lambda p: lever(_universe_name(p, anonymize))(p)})
    d = Desk(b, DeskConfig(anonymize=anonymize), **kw)
    got = sim.run(d, m, START, 2)
    e = next(x for x in d.log if x["role"] == role)
    assert e["source"] == "fallback" and "not tradeable" in e["reason"]
    ref = sim.run(Desk(RuleBrain(), DeskConfig(anonymize=anonymize), **kw), m, START, 2)
    pd.testing.assert_frame_equal(got.ledger, ref.ledger)


def test_an_event_call_or_a_free_book_naming_a_universe_name_is_refused():
    """The one check covers every lever, including those added after it."""
    d, seen = _payloads(DeskConfig(anonymize=True), n=1)
    u = _universe_name(seen[0][1], True)
    call = EventDecision(calls=[NameCall(name=u, action="exit", fraction=None, cause=None, reason="x")])
    with pytest.raises(ValueError, match="not tradeable"):
        d._only_tradeable(call)
    from icaif.agents.schemas import FreeDecision, NameWeight
    with pytest.raises(ValueError, match="not tradeable"):
        d._only_tradeable(FreeDecision(action="rebalance", weights=[NameWeight(name=u, weight=0.1)],
                                       rationale="x"))
    with pytest.raises(ValueError, match="not one of the 30"):
        d._only_tradeable(call.model_copy(update={"calls": [call.calls[0].model_copy(update={"name": "S99"})]}))
    d._only_tradeable(EventDecision(calls=[NameCall(name=seen[0][1]["names"][0]["name"], action="hold",
                                                    fraction=None, cause=None, reason="x")]))


# ----------------------------------------------------------------------------- live

def test_the_live_universe_row_is_what_the_agent_reads_with_or_without_its_date(world, tmp_path):  # noqa: F811
    """The scorer hands the universe over as a file. Read under another date, the door
    serves the agent nothing and the block silently disappears."""
    scored = live.daily_scores(NOW, predictor=FakePredictor())
    live.archive_scores(scored, tmp_path)
    tickers = list(scored.ours.index)
    day = scored.decision_date.date()
    deadline = calendar.at(day, calendar.ROUNDS[1][0])
    got = live.universe_scores(tmp_path, tickers).for_day(day, deadline)
    assert set(got.index) == set(scored.universe.dropna().index)
    assert set(got.index[got["tradeable"]]) == set(scored.ours.dropna().index)
    old = pd.read_parquet(tmp_path / "scores_universe.parquet").drop(columns="date")
    old.to_parquet(tmp_path / "scores_universe.parquet", index=False)
    pd.testing.assert_frame_equal(live.universe_scores(tmp_path, tickers).for_day(day, deadline), got)


def test_a_universe_name_without_the_latest_bar_is_scored_nan_like_one_of_the_30(world, monkeypatch):  # noqa: F811
    """A name a day behind has features from older prices among today's ranks; shown,
    its rank would be a number from two dates."""
    from tests.test_live import OTHERS as LIVE_OTHERS
    behind = LIVE_OTHERS[0]
    fetch = live.yahoo_daily

    def lagging(symbols, start):
        bars, missing = fetch(symbols, start)
        last = bars["date"].max()
        return bars[~((bars["ticker"] == behind) & (bars["date"] == last))], missing

    monkeypatch.setattr(live, "yahoo_daily", lagging)
    scored = live.daily_scores(NOW, predictor=FakePredictor())
    assert behind in scored.universe.index and np.isnan(scored.universe[behind])
    assert any(behind in w for w in scored.warnings)
