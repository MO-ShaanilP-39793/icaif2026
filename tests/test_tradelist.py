"""The v2 turnover budget and trade-list compiler (icaif/agents/budget.py, tradelist.py)."""

import json
import pandas as pd
import pytest

from icaif import calendar, kit, sim
from icaif import weights as W
from icaif.agents import brains
from icaif.agents.budget import BudgetExceeded, TurnoverBudget, window_rounds
from icaif.agents.journal import Journal
from icaif.agents.observe import Anonymizer
from icaif.agents.schemas import AddLine, CutLine, TradeList, Trim
from icaif.agents.tradelist import MIN_TRADE, TradeListError, compile_trades
from tests.test_agents import DAYS, START, _mkt
from tests.test_sim import TICKERS

HELD10 = TICKERS[:10]
CURRENT = pd.Series(0.0, index=TICKERS)
CURRENT[HELD10] = 0.075                      # the 75% book, ten names
ID = {t: t for t in TICKERS}
A, B, C, D = HELD10[0], HELD10[1], TICKERS[20], TICKERS[21]


def _tl(adds=(), cuts=(), trims=(), exposure=None) -> TradeList:
    return TradeList(adds=[AddLine(name=n, weight=w, why="x") for n, w in adds],
                     cuts=[CutLine(name=n, why="x") for n in cuts],
                     trims=[Trim(name=n, fraction=f, cause="news", why="x") for n, f in trims],
                     target_exposure=exposure, rationale="r")


def _go(tl, current=CURRENT, left=10.0, **kw):
    return compile_trades(tl, current, ID, budget_left=left, **kw)


def _refused(tl, *needles, **kw):
    with pytest.raises(TradeListError) as err:
        _go(tl, **kw)
    for n in needles:
        assert any(n in e for e in err.value.errors), (n, err.value.errors)
    return err.value.errors


# ----------------------------------------------------------------------------- the compiler

def test_an_empty_list_is_a_hold_and_trades_nothing():
    assert _go(_tl()) is None


def test_lines_are_exact_and_every_name_no_line_touches_keeps_its_weight():
    got = _go(_tl(adds=[(C, 0.1)], cuts=[A], trims=[(B, "half")]))
    assert got.target[A] == 0 and got.target[B] == pytest.approx(0.0375) and got.target[C] == 0.1
    rest = [t for t in TICKERS if t not in (A, B, C)]
    pd.testing.assert_series_equal(got.target[rest], CURRENT[rest])
    assert got.turnover == pytest.approx(0.075 + 0.0375 + 0.1)
    kit.validate_weights(got.weights)
    assert all(abs(w / W.GRID - round(w / W.GRID)) < 1e-6 for w in got.weights.values())


def test_a_target_exposure_scales_only_the_names_no_line_touches():
    """A stated add of 10% must stay 10% when the PM also cuts exposure; scaled with the
    rest, the book would hold a weight no line asked for."""
    got = _go(_tl(adds=[(C, 0.1)], exposure=0.7))
    assert got.target[C] == pytest.approx(0.1)
    assert got.target.sum() == pytest.approx(0.7)
    assert got.target[A] == pytest.approx(0.075 * 0.6 / 0.75)
    alone = _go(_tl(exposure=0.5))
    assert alone.target.sum() == pytest.approx(0.5) and alone.target[A] == pytest.approx(0.05)


def test_a_list_with_one_bad_line_is_refused_whole_with_every_reason():
    """Dropping the bad line and trading the rest would run a book nobody proposed, and
    the rationale logged with it would describe another."""
    errors = _refused(_tl(adds=[(C, 0.1), ("ZZZZ", 0.05)], cuts=[D]),
                      "ZZZZ is not one of the 30")
    assert len(errors) == 1                               # names are checked before any book
    errors = _refused(_tl(adds=[(C, 0.1), (B, 0.07)], cuts=[D]),
                      f"{D}: a cut of a name not held", f"{B}: an add to 0.0700")
    assert len(errors) == 2


@pytest.mark.parametrize("tl,needle,kw", [
    (_tl(cuts=["U07"]), "universe_context only", {"universe_names": {"U07"}}),
    (_tl(cuts=[A], trims=[(A, "half")]), "more than one line", {}),
    (_tl(trims=[(C, "half")]), "a trim of a name not held", {}),
    (_tl(adds=[(A, 0.05)]), "at or below its weight", {}),
    (_tl(adds=[(C, 0.1234567)]), "more than 6 decimals", {}),
    (_tl(exposure=0.7000001), "more than 6 decimals", {}),
    (_tl(adds=[(C, 0.003)]), "minimum trade", {}),
    (_tl(exposure=0.752), "minimum trade", {}),
    (_tl(adds=[(C, 0.3), (D, 0.3)]), "over the 1 allowed", {}),
    (_tl(adds=[(C, 0.1)]), "over the 0.8 allowed", {"exposure_cap": 0.8}),
    (_tl(adds=[(C, 0.3), (D, 0.3)], exposure=0.5), "above the target exposure", {}),
    (_tl(cuts=list(HELD10), exposure=0.2), "no name outside the lines", {}),
    (_tl(cuts=[A, B]), "budget has 0.1000 left", {"left": 0.1}),
])
def test_each_rule_refuses_the_list(tl, needle, kw):
    _refused(tl, needle, **kw)


def test_scaling_a_name_past_the_cap_is_refused_not_clipped():
    """Clipped at 30%, the book would land under the exposure the PM chose, saying nothing."""
    cur = CURRENT.copy()
    cur[A] = 0.29
    _refused(_tl(exposure=1.0), "over the 30% cap", current=cur)


def test_a_quarter_trim_of_a_small_position_is_refused_under_the_minimum_trade():
    cur = CURRENT.copy()
    cur[A] = 0.015
    _refused(_tl(trims=[(A, "quarter")]), "minimum trade", current=cur)
    assert _go(_tl(trims=[(A, "quarter")])).target[A] == pytest.approx(0.075 * 0.75)
    assert 0.075 * 0.25 >= MIN_TRADE


def test_codes_are_read_through_the_windows_mapping_never_as_tickers():
    anon = Anonymizer(TICKERS, seed=3)
    code = anon.code(C)
    got = compile_trades(_tl(adds=[(code, 0.05)]), CURRENT, anon.to_ticker, budget_left=1.0)
    assert got.target[C] == 0.05 and got.lines[0]["name"] == code
    with pytest.raises(TradeListError, match="not one of the 30"):
        compile_trades(_tl(adds=[(C, 0.05)]), CURRENT, anon.to_ticker, budget_left=1.0)


def test_the_trade_list_sent_to_grok_carries_no_numeric_bound_and_pydantic_keeps_them():
    """Bedrock snapped bounded numbers to a bound (every weight 0.30); the v2 schema goes
    through `bedrock_schema` like the rest."""
    text = json.dumps(brains.bedrock_schema(TradeList))
    assert not any(f'"{k}"' in text for k in brains.BOUNDS)
    assert "Must be >= 0.0 and <= 0.3." in text
    with pytest.raises(ValueError):
        AddLine(name=C, weight=0.31, why="x")


# ----------------------------------------------------------------------------- the budget

class Trader:
    """Trades between two books on a schedule, through a journal, as a desk does."""

    def __init__(self, plan):
        self.plan, self.journal, self.day_no, self._day = plan, Journal(), 0, None
        self.seen: list[float] = []

    def __call__(self, ctx):
        if ctx.day != self._day:
            self._day, self.day_no = ctx.day, self.day_no + 1
        self.journal.open_round(ctx, self.day_no)
        self.seen.append(TurnoverBudget(9.0, 1).spent(self.journal))
        px = ctx.market.exec_prices.loc[ctx.execution]
        sh = pd.Series(ctx.shares, dtype=float).reindex(TICKERS).fillna(0.0)
        nav = ctx.cash + float((sh * px).sum())
        current = sh * px / nav
        w = self.plan.get((self.day_no, ctx.round))
        self.journal.close_round([], w, current)
        return w


def _plan():
    a = {t: (0.075 if t in HELD10 else 0.0) for t in TICKERS}
    b = {t: (0.06 if t in TICKERS[5:20] else 0.0) for t in TICKERS}
    return {(1, 1): a, (1, 4): b, (2, 3): a, (3, 7): b}


def test_a_spent_budget_is_the_boards_turnover_times_the_rounds():
    """Spend counted in any other unit (weights, not notional over NAV before) would
    tell the PM it has room the board will charge for."""
    m = _mkt(seed=4)
    days = [d for d in m.days if d >= START][:5]
    tr = Trader(_plan())
    res = sim.run(tr, m, START, 5)
    bud = TurnoverBudget(total=3.0, rounds=window_rounds(days))
    kit_turnover = res.metrics()["turnover"]
    assert bud.filled(tr.journal) == pytest.approx(kit_turnover * bud.rounds, rel=1e-9)
    assert bud.pending(tr.journal) == 0.0
    assert bud.view(tr.journal)["board_turnover_if_no_more_trades"] == pytest.approx(kit_turnover, abs=1e-5)


def test_an_order_not_yet_filled_is_charged_so_two_rounds_cannot_spend_the_same_headroom():
    """Live, an order fills after the next round has started. Counted only once filled,
    the next round would be shown the headroom the order is about to spend."""
    m = _mkt(seed=4)
    plan = _plan()
    tr = Trader({(1, 1): plan[(1, 1)], (1, 7): plan[(1, 4)]})
    res = sim.run(tr, m, START, 1)       # round 7's order: placed, never seen filled
    bud = TurnoverBudget(total=3.0, rounds=7)
    last = res.periods[-1]
    assert bud.filled(tr.journal) == pytest.approx(0.75)
    assert bud.pending(tr.journal) == pytest.approx(last["traded_notional"] / last["nav_before"])
    assert bud.spent(tr.journal) == pytest.approx(res.metrics()["turnover"] * 7)
    bud.check(tr.journal, bud.left(tr.journal))
    with pytest.raises(BudgetExceeded):
        bud.check(tr.journal, bud.left(tr.journal) + 0.01)


def test_a_restarted_desk_reads_the_same_budget_from_its_journal():
    """Live, each round runs in a fresh process; a budget kept anywhere but the persisted
    journal would reset to full on every restart."""
    m = _mkt(seed=4)
    tr = Trader(_plan())
    sim.run(tr, m, START, 4)
    bud = TurnoverBudget(total=3.0, rounds=105)
    assert bud.view(Journal.from_json(json.loads(json.dumps(tr.journal.to_json())))) == bud.view(tr.journal)


def test_spend_seen_at_a_round_counts_only_fills_that_round_could_see():
    """Day 1's round-4 trade fills at its execution; the budget may count it from round 5,
    never in round 4's own decision."""
    m = _mkt(seed=4)
    tr = Trader(_plan())
    sim.run(tr, m, START, 2)
    r = {k: i for i, k in enumerate([(d, rr["round"]) for d in [START, DAYS[DAYS.index(START) + 1]]
                                     for rr in calendar.rounds_for(d)])}
    assert tr.seen[r[(START, 1)]] == 0.0
    assert tr.seen[r[(START, 2)]] == pytest.approx(0.75, rel=1e-3)
    assert tr.seen[r[(START, 4)]] == tr.seen[r[(START, 2)]]
    assert tr.seen[r[(START, 5)]] > tr.seen[r[(START, 4)]] + 0.5


def test_a_half_day_counts_four_rounds_toward_the_boards_divisor():
    half = next(iter(sorted(calendar.EARLY_CLOSES)))
    assert window_rounds([half]) == 4 and window_rounds([START]) == 7
    with pytest.raises(ValueError):
        TurnoverBudget(total=0.0, rounds=105)
