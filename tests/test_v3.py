"""Desk v3, stage 1: the PM's entry, code's self-check, then a pure hold (icaif/agents/v3.py)."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from icaif import baselines, calendar, sim, suites
from icaif.agents import prompts_v3, selfcheck
from icaif.agents.brains import BrainError, CachedBrain
from icaif.agents.schemas import AnalystReport, Condition, EntryWeight, PMCheck, PMEntry
from icaif.agents.v3 import (STAGE1, STREAMS, V3Config, V3Desk, HoldBrain, condition_errors,
                             parse_gross, strip)
from tests.test_agents import DAYS, START, _mkt
from tests.test_quant import _bars
from tests.test_sim import TICKERS

HOLD = baselines.scaled(baselines.InverseVolHold, 0.75)
N = 4


class Script(HoldBrain):
    """The hold brain, with some roles answered by `answers[key](payload)` instead;
    records every role, prompt and payload it was asked."""

    def __init__(self, name="script", **answers):
        self.name, self.answers, self.seen = name, answers, []

    def decide(self, role, system, payload, schema, timeout):
        self.seen.append((role, system, payload))
        key = role.removeprefix("v3_")
        if key in self.answers:
            return self.answers[key](payload)
        return super().decide(role, system, payload, schema, timeout)


def _desk(quick, deep=None, cfg=None, **kw):
    return V3Desk({"quick": quick, "deep": deep or quick}, cfg or V3Config(), **kw)


def _ctx(m, day, rnd=1):
    r = calendar.rounds_for(day)[rnd - 1]
    return sim.RoundContext(day, rnd, r["deadline"], r["execution"], {}, sim.INITIAL_NAV, m)


def _entry(weights: dict, conditions=(), thesis="t") -> PMEntry:
    return PMEntry(weights=[EntryWeight(name=n, weight=w) for n, w in weights.items()],
                   thesis=thesis, conditions=list(conditions))


def _cond(**kw) -> Condition:
    base = dict(kind="drawdown_from_peak_pct", name=None, threshold=3.0, item=None,
                action="review", gross=None, why="w")
    return Condition(**dict(base, **kw))


def _confirm(_):
    return PMCheck(action="confirm", revised=None, rationale="ok")


def _asked(brain, role):
    return [p for r, _, p in brain.seen if r == f"v3_{role}"]


# ----------------------------------------------------------------------------- plumbing

def test_a_v3_desk_answered_by_code_trades_exactly_as_the_hold():
    """With no entry from the PM the desk must buy `inv_vol_hold_75` and hold it, or every
    PM-vs-hold comparison in stage 1 measures the desk's plumbing instead of the PM."""
    m = _mkt()
    d = _desk(HoldBrain())
    got, want = sim.run(d, m, START, N), sim.run(HOLD(), m, START, N)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    assert got.metrics() == want.metrics()
    assert d.fallbacks == {"pm:failed": 1, "entry:hold_book": 1}
    assert d.entry["source"] == "fallback" and d.plan is None
    assert {e["role"] for e in d.chain} == {"pm"}        # no draft, so no check


def test_a_sleeve_arm_falls_back_to_inverse_vol_at_its_own_gross_not_75():
    """At 75% a sleeve arm's fallback would carry the cash question into the arm built to
    hold it fixed."""
    m = _mkt()
    d = _desk(HoldBrain(), cfg=V3Config(gross=("fixed", 0.3)))
    got = sim.run(d, m, START, N)
    want = sim.run(baselines.scaled(baselines.InverseVolHold, 0.3)(), m, START, N)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)


def test_a_confirmed_entry_trades_exactly_the_weights_it_states_and_then_holds():
    m = _mkt()
    hold = HOLD()(_ctx(m, START))
    stated = {t: w for t, w in hold.items() if w > 0}
    b = Script(pm=lambda p: _entry(stated), pm_check=_confirm)
    d = _desk(b)
    got = sim.run(d, m, START, N)
    pd.testing.assert_frame_equal(got.ledger, sim.run(HOLD(), m, START, N).ledger)
    assert d.fallbacks == {} and d.entry["final"] == "draft"
    assert len(_asked(b, "pm")) == 1 and len(_asked(b, "pm_check")) == 1   # day 1 only


def test_the_pm_sees_its_drafts_numbers_beside_the_reference_books_at_the_same_gross():
    m = _mkt()
    stated = {TICKERS[0]: 0.3, TICKERS[1]: 0.2}
    b = Script(pm=lambda p: _entry(stated), pm_check=_confirm)
    sim.run(_desk(b), m, START, 1)
    shown = _asked(b, "pm_check")[0]
    assert shown["draft"]["weights"] == [{"name": n, "weight": w} for n, w in stated.items()]
    rep = shown["self_check"]
    assert rep["your_book"]["gross"] == 0.5 and rep["your_book"]["names_held"] == 2
    assert rep["your_book"]["effective_names"] == pytest.approx(1 / (0.6 ** 2 + 0.4 ** 2), abs=0.05)
    assert rep["inverse_vol_at_your_gross"]["gross"] == pytest.approx(0.5)
    assert rep["risk_parity_at_your_gross"]["gross"] == pytest.approx(0.5)
    assert rep["your_book"]["expected_vol_ann"] > rep["inverse_vol_at_your_gross"]["expected_vol_ann"]


def test_a_refused_draft_reaches_the_check_with_every_reason_and_a_valid_revision_trades():
    """The self-check is the refused draft's second chance; without it a typo in one
    weight would cost the window its PM and buy the fallback."""
    m = _mkt()
    bad = {TICKERS[0]: 0.45, TICKERS[1]: 0.2, "NOPE": 0.1}
    good = {TICKERS[0]: 0.25, TICKERS[1]: 0.2}
    b = Script(pm=lambda p: _entry(bad),
               pm_check=lambda p: PMCheck(action="revise", revised=_entry(good), rationale="fix"))
    d = _desk(b)
    got = sim.run(d, m, START, 1)
    refused = _asked(b, "pm_check")[0]["self_check"]["refused"]
    assert any("NOPE" in r for r in refused)
    assert d.entry["final"] == "revised" and d.fallbacks == {}
    assert got.ledger.iloc[0][TICKERS[0]] > 0 and d.entry["weights"] == pytest.approx(good, abs=1e-9)
    assert "revised_self_check" in d.entry


def test_a_refused_revision_buys_the_fallback_and_is_counted():
    m = _mkt()
    bad = {TICKERS[0]: 0.45}
    b = Script(pm=lambda p: _entry(bad),
               pm_check=lambda p: PMCheck(action="revise", revised=_entry(bad), rationale="no"))
    d = _desk(b)
    got = sim.run(d, m, START, N)
    pd.testing.assert_frame_equal(got.ledger, sim.run(HOLD(), m, START, N).ledger)
    assert d.fallbacks == {"pm_check:refused": 1, "entry:hold_book": 1}
    assert "over the 30% cap" in d.entry["reason"]


def test_without_the_self_check_the_draft_is_final_and_nothing_else_is_asked():
    m = _mkt()
    b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2}))
    d = _desk(b, cfg=V3Config(self_check=False))
    sim.run(d, m, START, 1)
    assert _asked(b, "pm_check") == [] and d.entry["final"] == "draft"


def test_an_all_cash_entry_is_entered_and_never_asked_again():
    """Cash ranks respectably; a desk that read an empty book as "not entered yet" would
    ask the PM every morning and buy whatever it said on day 2."""
    m = _mkt()
    b = Script(pm=lambda p: _entry({}), pm_check=_confirm)
    d = _desk(b)
    got = sim.run(d, m, START, N)
    assert got.metrics()["turnover"] == 0 and len(_asked(b, "pm")) == 1
    assert d.entry["final"] == "draft" and d.fallbacks == {}


# ----------------------------------------------------------------------------- the gross rule

@pytest.mark.parametrize("gross, weights, ok", [
    (("band", 0.25, 0.5), {TICKERS[0]: 0.3, TICKERS[1]: 0.3}, False),     # 0.6 above the band
    (("band", 0.25, 0.5), {TICKERS[0]: 0.2, TICKERS[1]: 0.2}, True),
    (("free",), {TICKERS[0]: 0.3, TICKERS[1]: 0.3, TICKERS[2]: 0.3, TICKERS[3]: 0.2}, False),
    (("free",), {TICKERS[0]: 0.1234567}, False),                           # 7 decimals
    (("fixed", 0.5), {TICKERS[0]: 0.5, TICKERS[1]: 0.5}, True),            # 0.25 each as bought
    (("fixed", 0.5), {TICKERS[0]: 0.7, TICKERS[1]: 0.3}, False),           # 0.35 as bought
    (("fixed", 0.5), {TICKERS[0]: 0.5, TICKERS[1]: 0.4}, False),           # shares sum to 0.9
])
def test_the_gross_rule_is_checked_on_the_book_as_bought_and_refused_never_rescaled(gross, weights, ok):
    m = _mkt()
    b = Script(pm=lambda p: _entry(weights))
    d = _desk(b, cfg=V3Config(gross=gross, self_check=False))
    got = sim.run(d, m, START, 1)
    assert (d.entry["source"] == "brain") is ok
    if ok and gross[0] == "fixed":
        assert sum(got.ledger.iloc[0][TICKERS]) * 100 / sim.INITIAL_NAV == pytest.approx(0.5, abs=1e-5)


def test_the_gross_rule_reaches_the_pm_in_its_payload():
    for gross, want in ((("free",), "free"), (("band", 0.2, 0.6), "band"), (("fixed", 0.3), "sleeve")):
        b = Script()
        sim.run(_desk(b, cfg=V3Config(gross=gross)), _mkt(), START, 1)
        assert _asked(b, "pm")[0]["gross_rule"]["rule"] == want


def test_parse_gross_reads_each_form_and_refuses_anything_else():
    assert parse_gross("free") == ("free",)
    assert parse_gross("band:0.25:0.75") == ("band", 0.25, 0.75)
    assert parse_gross("fixed:0.3") == ("fixed", 0.3)
    with pytest.raises(ValueError):
        parse_gross("fixed")
    with pytest.raises(ValueError):
        V3Config(gross=("band", 0.8, 0.2))


# ----------------------------------------------------------------------------- conditions

@pytest.mark.parametrize("cond, problem", [
    (dict(kind="move_from_entry_sigma", name=TICKERS[5], threshold=-2.5, action="exit"), "not in the book"),
    (dict(kind="move_from_entry_sigma", name=None, threshold=-2.5, action="exit"), "names no name"),
    (dict(kind="drawdown_from_peak_pct", name=TICKERS[0], threshold=3.0), "names no name"),
    (dict(kind="drawdown_from_peak_pct", action="exit"), "acts by"),
    (dict(kind="drawdown_from_peak_pct", action="set_gross"), "needs a gross"),
    (dict(kind="new_8k_item", name=TICKERS[0], threshold=None, action="review"), "no 8-K item"),
    (dict(kind="vol_ratio_above", threshold=None), "no threshold"),
])
def test_a_condition_code_cannot_evaluate_refuses_the_whole_entry(cond, problem):
    """A fired condition binds (it reaches the PM with its action as the proposal), so one
    code cannot evaluate would sit in the plan never firing, while the PM believed it
    was guarded."""
    c = _cond(**cond)
    errs = condition_errors(c, {TICKERS[0]})
    assert any(problem in e for e in errs), errs
    b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2}, [c]))
    d = _desk(b, cfg=V3Config(self_check=False))
    sim.run(d, _mkt(), START, 1)
    assert d.entry["source"] == "fallback" and d.plan is None


def test_a_valid_plan_is_kept_with_the_entry_and_survives_a_restart():
    m = _mkt()
    conds = [_cond(kind="move_from_entry_sigma", name=TICKERS[0], threshold=-2.5, action="exit"),
             _cond(kind="new_8k_item", name=TICKERS[0], threshold=None, item="2.05", action="review"),
             _cond(kind="vol_ratio_above", threshold=1.8, action="set_gross", gross=0.4)]
    b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2}, conds), pm_check=_confirm)
    d = _desk(b)
    sim.run(d, m, START, 2)
    assert [c["kind"] for c in d.plan["conditions"]] == [c.kind for c in conds]
    again = _desk(HoldBrain())
    again.restore(json.loads(json.dumps(d.state())), TICKERS)
    assert again.plan == d.plan and again.entry == d.entry


# ----------------------------------------------------------------------------- streams

class _Full(V3Desk):
    """A v3 desk whose observation carries every stream's fields, as a live desk's does;
    the synthetic market alone has no scores, HAR, macro, news or universe."""

    def _payload(self, *a, **k):
        obs = super()._payload(*a, **k)
        obs["macro"] = {"vix_z": 1.0}
        obs["universe_context"] = {"rows": [["U01", 1, 0.99, False]]}
        obs["market"].update(vol_ann_har_1d=0.2, vol_ann_har_3d=0.21)
        for i, r in enumerate(obs["names"]):
            r.update(headlines=[{"title": "x"}], recent_8k_filings=[{"items": "2.02"}],
                     model_score_rank=i + 1, vol_ann_har_1d=0.3, vol_ann_har_3d=0.31)
        return obs

    def _new_filings(self, ctx):
        return {self.anon.code(TICKERS[0]): [{"items": "8.01"}]}


FIELDS_OF = {"regime": ("p_turbulent_next_session", "regime_persistence_days"),
             "headlines": ("headlines",), "filings": ("recent_8k_filings", "new_filings"),
             "macro": ("macro",), "model_rank": ("model_score_rank",),
             "universe": ("universe_context",), "har_vol": ("vol_ann_har_1d", "vol_ann_har_3d")}


def _keys(x) -> set:
    if isinstance(x, dict):
        return set(x) | set().union(*(_keys(v) for v in x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*(_keys(v) for v in x)) if x else set()
    return set()


@pytest.mark.parametrize("dropped", STREAMS)
def test_a_dropped_stream_reaches_no_role_and_every_other_stream_still_does(dropped):
    """A leave-one-out run that still carried the stream somewhere (an analyst's slice, the
    PM's check) would measure nothing and read as "this stream doesn't matter"."""
    assert set(FIELDS_OF) == set(STREAMS)
    b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2}), pm_check=_confirm)
    keep = tuple(s for s in STREAMS if s != dropped)
    d = _Full({"quick": b, "deep": b}, V3Config(analysts="reports_raw", streams=keep))
    sim.run(d, _mkt(), START, 1)
    every = set().union(*(_keys(p) for _, _, p in b.seen))
    for s, fields in FIELDS_OF.items():
        present = set(fields) & every
        if s == dropped:
            assert not present, (s, present)
        else:
            assert present, (s, "missing although kept")
    for _, _, p in b.seen:
        assert not (set(FIELDS_OF[dropped]) & _keys(p))


def test_strip_removes_entry_copies_and_leaves_the_original_untouched():
    obs = {"market": {"vol_ann_har_1d": 1, "vol_ann_har_1d_at_entry": 1, "basket_ret_1d": 0},
           "names": [{"name": "A", "model_score_rank": 1, "model_score_rank_at_entry": 2, "ret_1d": 0}],
           "macro": {}, "universe_context": {}}
    out = strip(obs, ("regime", "headlines", "filings"))
    assert out == {"market": {"basket_ret_1d": 0}, "names": [{"name": "A", "ret_1d": 0}]}
    assert "macro" in obs and "model_score_rank" in obs["names"][0]


# ----------------------------------------------------------------------------- analysts

def test_each_entry_analyst_reads_only_its_own_slice_and_the_pm_reads_their_reports():
    b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2}), pm_check=_confirm)
    d = _Full({"quick": b, "deep": b}, V3Config(analysts="reports_raw"))
    sim.run(d, _mkt(), START, 1)
    view = {r.removeprefix("v3_"): p for r, _, p in b.seen}
    assert "names" not in view["market"] and "macro" in view["market"]
    assert "headlines" not in _keys(view["quant"]) and "universe_context" in view["quant"]
    assert "model_score_rank" not in _keys(view["news"]) and "headlines" in _keys(view["news"])
    assert set(view["pm"]["reports"]) == {"market", "earnings", "news", "quant"}
    assert "headlines" in _keys(view["pm"])                      # raw beside the reports
    assert view["pm"]["reports"]["earnings"]["unavailable"].startswith("no name reports")


def test_a_pm_reading_reports_only_sees_no_raw_signal_beside_them():
    b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2}), pm_check=_confirm)
    d = _Full({"quick": b, "deep": b}, V3Config(analysts="reports_only"))
    sim.run(d, _mkt(), START, 1)
    pm = _asked(b, "pm")[0]
    raw = set().union(*FIELDS_OF.values()) | {"ou_s_score"}
    assert not (raw & (_keys({k: v for k, v in pm.items() if k != "reports"})))
    assert set(pm["reports"]) == {"market", "earnings", "news", "quant"}


def test_an_analyst_naming_a_name_outside_the_30_is_skipped_not_believed():
    from icaif.agents.schemas import NameView

    b = Script(quant=lambda p: AnalystReport(summary="s", names=[NameView(name="ZZZ", lean="buy", note="n")]),
               pm=lambda p: _entry({TICKERS[0]: 0.2}), pm_check=_confirm)
    d = _desk(b, cfg=V3Config(analysts="reports_raw"))
    sim.run(d, _mkt(), START, 1)
    assert d.fallbacks["quant:failed"] == 1
    assert "unavailable" in _asked(b, "pm")[0]["reports"]["quant"]


# ----------------------------------------------------------------------------- prompts, cache

def test_only_the_evidence_arm_tells_the_pm_our_backtests_and_no_analyst_ever_hears_them():
    for evidence in (False, True):
        b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2}), pm_check=_confirm)
        sim.run(_desk(b, cfg=V3Config(analysts="reports_raw", evidence=evidence)), _mkt(), START, 1)
        for role, system, _ in b.seen:
            told = "Buying once and holding wins" in system
            assert told == (evidence and role in ("v3_pm", "v3_pm_check")), role


def test_v3_roles_reach_the_cache_under_their_own_names_and_a_repeat_asks_again(tmp_path):
    """A v3 PM sharing v2's "v2_pm" key would be handed a v2 answer whenever the payloads
    matched; a noise check reusing repeat 0's cache would compare each answer with itself."""
    payload, system = {"a": 1}, "s"
    k0 = CachedBrain.key("m", "v3_pm", system, payload, PMEntry)
    assert k0 == CachedBrain.key("m", "v3_pm", system, payload, PMEntry, 0)
    assert k0 != CachedBrain.key("m", "v3_pm", system, payload, PMEntry, 1)
    assert k0 != CachedBrain.key("m", "v2_pm", system, payload, PMEntry)
    b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2}), pm_check=_confirm)
    cached = CachedBrain(b, tmp_path)
    sim.run(_desk(cached), _mkt(), START, 1)
    sim.run(_desk(cached), _mkt(), START, 1)
    assert cached.misses == 2 and cached.hits == 2
    again = CachedBrain(b, tmp_path, repeat=1)
    sim.run(_desk(again), _mkt(), START, 1)
    assert again.misses == 2 and again.hits == 0


# ----------------------------------------------------------------------------- point in time

def test_no_entry_payload_or_self_check_changes_when_every_later_bar_is_rewritten():
    bars = _bars(DAYS, 8)
    cut = calendar.rounds_for(START)[0]["deadline"]
    late = bars.copy()
    late.loc[late["end"] > cut, ["open", "high", "low", "close"]] *= 3.0
    seen = []
    for b_ in (bars, late):
        rec = Script(pm=lambda p: _entry({TICKERS[0]: 0.2, TICKERS[1]: 0.1}), pm_check=_confirm)
        sim.run(_desk(rec, cfg=V3Config(analysts="reports_raw")), _mkt(b_), START, 1)
        seen.append({r: p for r, _, p in rec.seen})
    assert set(seen[0]) == {"v3_market", "v3_quant", "v3_pm", "v3_pm_check"} and seen[0] == seen[1]


def test_the_self_check_reads_only_closes_before_the_deadline():
    m = _mkt()
    ctx = _ctx(m, START)
    from icaif import quant_strategies as qs

    closes = qs.daily_closes(ctx, qs.HISTORY_DAYS + 1)
    assert closes.index[-1].date() < START
    book = pd.Series(0.0, index=TICKERS)
    book[TICKERS[:3]] = 0.2
    rep = selfcheck.report(book, closes, {TICKERS[0]: 3}, lambda t: t)
    assert rep["your_book"]["weight_reporting_within_10_sessions"] == pytest.approx(0.2)
    assert np.isfinite(rep["your_book"]["expected_vol_ann"])


class _Restarted:
    def __init__(self, brain):
        self.brain, self.saved, self.desk = brain, None, None

    def __call__(self, ctx):
        d = _desk(self.brain)
        d.restore(json.loads(self.saved) if self.saved else None, ctx.market.tickers)
        out = d(ctx)
        self.saved = json.dumps(d.state())
        self.desk = d
        return out


def test_a_v3_desk_restored_before_every_round_trades_as_one_that_never_stopped():
    """Live, each round runs in its own process: a desk that forgot it had entered would
    ask the PM again on day 2 and buy a second book."""
    m = _mkt()
    mk = lambda: Script(pm=lambda p: _entry({TICKERS[0]: 0.2, TICKERS[3]: 0.15}), pm_check=_confirm)  # noqa: E731
    whole, again = _desk(mk()), _Restarted(mk())
    a, b = sim.run(whole, m, START, N), sim.run(again, m, START, N)
    pd.testing.assert_frame_equal(a.ledger, b.ledger)
    assert again.desk.entry == whole.entry


# ----------------------------------------------------------------------------- windows

def test_stage1_windows_follow_the_cutoff_never_overlap_and_split_selection_before_confirmation():
    """Confirmation must be windows the choice never saw, and official4 is stage 2's."""
    o4 = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in suites.get("official4").windows]
    allw = [(date.fromisoformat(a), date.fromisoformat(b)) for split in ("select", "confirm")
            for a, b in STAGE1[split]]
    assert allw[0][0] >= date(2025, 2, 1)
    for (a0, a1), (b0, _) in zip(allw, allw[1:]):
        assert a0 <= a1 < b0
    for a, b in allw:
        assert all(b < s or a > e for s, e in o4), (a, b)
    assert max(b for _, b in (map(date.fromisoformat, w) for w in STAGE1["select"])) < \
        min(a for a, _ in (map(date.fromisoformat, w) for w in STAGE1["confirm"]))
    assert len(STAGE1["select"]) == 13 and len(STAGE1["confirm"]) == 9


def test_a_bought_book_at_its_own_gross_repeats_the_desk_trade_for_trade():
    """The rescale scores the PM's names at other grosses; at its own it must be the
    desk's run exactly, or every rescaled score carries a plumbing difference."""
    from icaif.agents.v3 import BoughtBook

    m = _mkt()
    b = Script(pm=lambda p: _entry({TICKERS[0]: 0.2, TICKERS[3]: 0.123457}), pm_check=_confirm)
    seen = []
    d = _desk(b)
    got = sim.run(lambda ctx: seen.append(d(ctx)) or seen[-1], m, START, N)
    bought = next(w for w in seen if w is not None)
    again = sim.run(BoughtBook(bought), m, START, N)
    pd.testing.assert_frame_equal(got.ledger, again.ledger)
    lower = sim.run(BoughtBook(bought, 0.25), m, START, N)
    assert sum(lower.ledger.iloc[0][TICKERS]) * 100 / sim.INITIAL_NAV == pytest.approx(0.25, abs=1e-5)
    with pytest.raises(ValueError, match="over the 30% cap"):
        BoughtBook(bought, 0.5)       # AAPL would be 0.31: a gross this book cannot reach


def test_the_self_check_shows_a_drafts_recent_drawdown_but_never_its_trailing_return():
    """A trailing return beside a draft argues for whatever just rose: in the first paid
    window the PM confirmed a concentrated book citing it as "backtested performance"."""
    from icaif import quant_strategies as qs

    m = _mkt()
    closes = qs.daily_closes(_ctx(m, START), qs.HISTORY_DAYS + 1)
    book = pd.Series(0.0, index=TICKERS)
    book[TICKERS[:3]] = 0.2
    rep = selfcheck.report(book, closes, None, lambda t: t)
    for k in ("your_book", "inverse_vol_at_your_gross", "risk_parity_at_your_gross"):
        recent = rep[k]["last_15_sessions_if_held"]
        assert set(recent) == {"max_drawdown"} and recent["max_drawdown"] >= 0
    assert "return" not in json.dumps(rep)


# ----------------------------------------------------------------------------- the experiment

def test_the_streams_factorial_is_balanced_orthogonal_and_keeps_mains_clear_of_pairs():
    """A stream's estimated effect is only its own if every stream is in for half the runs,
    the columns are orthogonal, and no main effect is aliased with a pair of others."""
    import itertools

    from icaif.agents.v3 import FACTORIAL_ORDER, STREAMS16

    assert set(FACTORIAL_ORDER) == set(STREAMS) and len(STREAMS16) == 16
    X = np.array([[1 if s in run else -1 for s in FACTORIAL_ORDER] for run in STREAMS16])
    assert (X.sum(axis=0) == 0).all()
    assert (X.T @ X == 16 * np.eye(7)).all()
    for i, (j, k) in itertools.product(range(7), itertools.combinations(range(7), 2)):
        if i not in (j, k):
            assert abs((X[:, i] * X[:, j] * X[:, k]).sum()) < 16, (i, j, k)
    assert STREAMS16[0] == () and set(STREAMS16[-1]) == set(STREAMS)
    assert len(set(STREAMS16)) == 16


def test_a_capped_book_lowers_only_a_book_above_the_floors_gross_and_keeps_its_shape():
    from icaif.agents.v3 import capped_book, floor_gross

    book = {"A": 0.3, "B": 0.2, "C": 0.1}
    assert capped_book(book, floor_gross(0.0)) == book
    assert capped_book(book, floor_gross(0.4)) == book                 # 0.6 fits under 0.6
    low = capped_book(book, floor_gross(0.75))
    assert sum(low.values()) == pytest.approx(0.25)
    assert low["A"] / low["B"] == pytest.approx(1.5)
