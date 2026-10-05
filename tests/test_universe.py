"""The training universe's membership: renames, late index changes, and staleness."""

import pandas as pd
import pytest

from icaif import live, universe
from tests.test_live import NOW, FakePredictor, world  # noqa: F401 - the fixture


def _snapshot():
    try:
        return universe.latest("sp500_ticker_start_end_*.csv")
    except FileNotFoundError:
        pytest.skip("no membership snapshot in data/external")


def test_each_rename_is_one_company_whose_old_spell_ends_the_day_its_new_one_starts():
    """A takeover mapped as a rename would price the acquirer under the target's spell:
    the wrong company's returns in the universe. Each old ticker must be a member up to
    the day its new ticker became one, and the new ticker still a member."""
    m = pd.read_csv(_snapshot(), parse_dates=["start_date", "end_date"])
    for old, new in universe.RENAMES.items():
        o = m[m["ticker"] == old].sort_values("start_date").iloc[-1]
        n = m[(m["ticker"] == new) & (m["start_date"] == o["end_date"])]
        assert len(n) == 1, f"{old} -> {new}: no {new} spell starting {o['end_date']:%Y-%m-%d}"
        assert n["end_date"].isna().all(), f"{new} has left the index since"


def test_a_renamed_company_prices_as_one_symbol_through_both_spells():
    """Under its old ticker the company had no prices for that spell, and the universe
    took the next name by dollar volume in its place."""
    m = universe.load_membership(_snapshot())
    assert set(m.loc[m["ticker"].isin(universe.RENAMES), "symbol"]) == set(universe.RENAMES.values())
    days = pd.DatetimeIndex(["2024-06-03", "2025-06-02", "2026-09-28"])
    mask = universe.member_mask(m, days)
    for sym in ("BNY", "FISV", "MRSH"):
        assert mask[sym].all(), sym


def test_late_changes_apply_once_and_give_way_to_a_snapshot_that_has_them():
    spells = pd.DataFrame({"ticker": ["OLD", "KEEP"], "start": pd.to_datetime(["2000-01-03"] * 2),
                           "end": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]")})
    changes = [("NEW", "2026-09-21", "add"), ("OLD", "2026-09-21", "drop")]
    once = universe.with_late_changes(spells, changes)
    assert once.loc[once["ticker"] == "OLD", "end"].item() == pd.Timestamp("2026-09-21")
    assert once.loc[once["ticker"] == "NEW", "end"].isna().item()
    pd.testing.assert_frame_equal(universe.with_late_changes(once, changes), once)
    # A snapshot that dropped OLD earlier keeps its own date.
    early = spells.assign(end=pd.to_datetime(["2026-09-01", None]))
    assert universe.with_late_changes(early, changes).loc[0, "end"] == pd.Timestamp("2026-09-01")


def test_the_september_2026_rebalance_is_in_the_universe_the_live_desk_ranks():
    """The snapshot's source last updated on 2026-09-07; without the late changes BE
    (about 4x the top-100 cut in dollar volume) was never priced live."""
    m = universe.load_membership(_snapshot())
    day = pd.Timestamp("2026-10-05")
    current = set(m[(m["start"] <= day) & (m["end"].isna() | (m["end"] > day))]["symbol"])
    assert {"BE", "ILMN", "P"} <= current
    assert not {"TAP", "TTD", "BLDR"} & current
    assert universe.as_of(m) >= universe.last_rebalance(day)


@pytest.mark.parametrize("day, want", [("2026-10-05", "2026-09-21"), ("2026-09-20", "2026-06-22"),
                                       ("2026-09-21", "2026-09-21"), ("2026-01-10", "2025-12-22")])
def test_the_last_rebalance_is_the_monday_after_the_third_friday_of_the_quarter(day, want):
    assert universe.last_rebalance(day) == pd.Timestamp(want)


def test_a_stale_membership_is_named_in_the_warnings_and_a_gone_name_is_not_a_missing_member(
        world, monkeypatch):  # noqa: F811 - the fixture
    """Recorded, not raised: the scores still mean something, but a ranking over last
    quarter's index should say so. And a takeover Yahoo no longer serves is expected,
    kept apart from a current member gone unpriced, the list that costs the universe a name."""
    membership, fetch = live.load_membership(), live.yahoo_daily
    gone = pd.DataFrame({"ticker": ["GONE"], "symbol": ["GONE"], "start": [pd.Timestamp("2010-01-04")],
                         "end": [pd.Timestamp("2026-01-05")]})
    monkeypatch.setattr(live, "load_membership", lambda: pd.concat([membership, gone], ignore_index=True))

    def no_gone(symbols, start):
        bars, missing = fetch([s for s in symbols if s != "GONE"], start)
        return bars, missing + (["GONE"] if "GONE" in symbols else [])

    monkeypatch.setattr(live, "yahoo_daily", no_gone)
    scored = live.daily_scores(NOW, predictor=FakePredictor())
    assert any("2026-09-21 rebalance" in w for w in scored.warnings)
    meta = scored.inputs.meta
    assert meta["membership_as_of"] == "2026-01-05"
    assert meta["ended_spells_unpriced"] == ["GONE"] and meta["current_members_unpriced"] == []
