"""Day-1 sizing from the HAR forecast (Roadmap step 3): weights and exposure read once at entry."""

import numpy as np
import pandas as pd
import pytest

from icaif import baselines, calendar, compiler, har_sizing as hs, quant_strategies as qs, sim, vol
from icaif.agents.signals import VolForecasts
from tests.test_quant import _bars, _random_cov
from tests.test_signals import DAYS, START
from tests.test_sim import TICKERS, _market

# The fixture's forecasts start at the Apr 2026 refit, a few sessions before START, so
# only a short typical window has history enough to lean against.
SHORT = 5


def _world(bars=None):
    bars = _bars(DAYS, 8) if bars is None else bars
    return _market(DAYS, info_bars=bars), VolForecasts.from_bars(bars, TICKERS)


def _ctx(m, day, round_no=1):
    r = calendar.rounds_for(day)[round_no - 1]
    return sim.RoundContext(day, round_no, r["deadline"], r["execution"], {}, sim.INITIAL_NAV, m)


def _har(mkt_var: list, h: int = 3, days=None) -> VolForecasts:
    """Every name at 1e-4 a day; the basket's forecasts as given, one per session."""
    days = DAYS[:len(mkt_var)] if days is None else days
    wide = pd.DataFrame(1e-4, index=pd.Index(days, name="session"), columns=TICKERS)
    wide[vol.MARKET] = mkt_var
    frame = wide.stack().rename(f"har_h{h}").to_frame()
    frame.index.names = ["session", "ticker"]
    return VolForecasts(frame, TICKERS)


def _deadline(day):
    return calendar.rounds_for(day)[0]["deadline"]


VARIANTS = {
    "weights": lambda har: hs.inverse_vol_hold(har, weights_h=3),
    "exposure": lambda har: hs.inverse_vol_hold(har, exposure=hs.HarExposure(3, SHORT, 0.25, 0.95)),
    "both": lambda har: hs.inverse_vol_hold(har, 3, hs.HarExposure(1, SHORT, 0.25, 0.95)),
    "rp_sizing": lambda har: hs.risk_parity_entry(har, vol_h=3),
    "rp_instead": lambda har: hs.risk_parity_entry(har, None, "har", hs.HarExposure(3, SHORT, 0.25, 0.95)),
    "rp_mean": lambda har: hs.risk_parity_entry(har, 3, "mean", hs.HarExposure(3, SHORT, 0.25, 0.95)),
    "rp_min": lambda har: hs.risk_parity_entry(har, 1, "min", hs.HarExposure(1, SHORT, 0.25, 0.95)),
}


# ----------------------------------------------------------------------------- the references

def test_with_har_switched_off_each_book_is_its_reference_trade_for_trade():
    """A HAR book may differ from its reference only where it reads HAR. If the wrapper
    moved anything itself (another snap to the grid, a different entry round, an HMM
    asked on other days), every score difference would be measuring the wrapper."""
    m, har = _world()
    pairs = ((hs.inverse_vol_hold(har), baselines.scaled(baselines.InverseVolHold, hs.E_REF)),
             (hs.risk_parity_entry(har), qs.CANDIDATES["q_riskparity_entry_regime"]))
    for ours, ref in pairs:
        got, want = sim.run(ours(), m, START, 5), sim.run(ref(), m, START, 5)
        pd.testing.assert_frame_equal(got.ledger, want.ledger)
        assert got.metrics() == want.metrics()


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_har_book_trades_once_at_entry_and_never_again(name):
    """The point of sizing at entry is that it costs no turnover the hold does not pay.
    A book that re-read the forecast later and traded on it would be the daily vol
    target again, which already lost to the hold."""
    m, har = _world()
    res = sim.run(VARIANTS[name](har)(), m, START, 5)
    traded = [p for p in res.periods if p["traded_notional"] > 0]
    assert len(traded) == 1 and traded[0]["execution"] == calendar.rounds_for(START)[0]["execution"]
    assert not res.invalid_rounds


# ----------------------------------------------------------------------------- weights

def test_har_weights_are_inverse_to_each_names_forecast_vol_at_the_hold_gross():
    """The substitution is the vol and nothing else: same normalisation, same gross."""
    m, har = _world()
    ctx = _ctx(m, START)
    b = hs.HarInverseVolHold(har, weights_h=3)
    w = pd.Series(b(ctx))[TICKERS]
    inv = 1 / hs.har_vols(har, ctx, 3)[TICKERS]
    np.testing.assert_allclose(w, inv / inv.sum() * hs.E_REF, atol=2e-6)
    ref = pd.Series(baselines.scaled(baselines.InverseVolHold, hs.E_REF)()(ctx))[TICKERS]
    assert (w - ref).abs().max() > 1e-4  # it really is a different book
    assert b.fallbacks == []


def test_har_covariance_keeps_the_sample_correlations_and_takes_the_forecast_vols():
    """HAR forecasts each name's variance, not co-movement. Swapping the whole matrix
    for a diagonal one would turn risk parity into inverse-vol under another name."""
    cov = _random_cov(6, 0)
    hv = pd.Series(np.linspace(0.01, 0.03, 6), index=cov.index)
    out = hs.har_covariance(cov, hv)
    np.testing.assert_allclose(np.sqrt(np.diag(out)), hv)
    sd = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(out / np.outer(hv, hv), cov / np.outer(sd, sd))


def test_a_name_without_a_forecast_falls_back_to_the_reference_shape_and_says_so():
    """Dropping the name would score a 29-name book as HAR sizing; blocking the entry
    would sit in cash. Either reads as a result. The book buys what the reference would
    have bought, and the fallback is on record."""
    bars = _bars(DAYS, 8)
    m = _market(DAYS, info_bars=bars)
    frame = vol.walk_forward(vol.realised_variance(bars), first_test="2026-04-01")
    frame.loc[(START, TICKERS[0]), "har_h3"] = np.nan
    har = VolForecasts(frame, TICKERS)
    ctx = _ctx(m, START)
    for ours, ref in ((hs.HarInverseVolHold(har, weights_h=3),
                       baselines.scaled(baselines.InverseVolHold, hs.E_REF)()),
                      (hs.HarRiskParity(har, vol_h=3), qs.CANDIDATES["q_riskparity_entry_regime"]())):
        assert ours(ctx) == ref(ctx)
        assert len(ours.fallbacks) == 1 and TICKERS[0] in ours.fallbacks[0]


# ----------------------------------------------------------------------------- exposure

@pytest.mark.parametrize("today, want", [(4e-4, 0.375), (1e-4, 0.75), (0.25e-4, 0.95), (25e-4, 0.25)])
def test_exposure_leans_against_the_forecast_and_stays_inside_its_clip(today, want):
    """e = 0.75 x typical / forecast in vol: a basket forecast at 4x its typical variance
    (2x the vol) halves the hold's gross, a calm one raises it, and neither leaves the
    clip. Read in variance instead of vol, the 4x day would cut to 0.19."""
    mkt = [1e-4] * 39 + [today]
    har = _har(mkt)
    day = DAYS[39]
    e = hs.HarExposure(3, 40, 0.25, 0.95)(har, day, _deadline(day))
    assert e == pytest.approx(want)


def test_too_short_a_history_for_a_typical_level_takes_the_hold_gross_and_says_so():
    """A median of a few sessions is not a typical level; leaning against it would be a
    second, noisier forecast. The book holds e_ref, the reference's own gross, and the
    log says why."""
    m, har = _world()
    ctx = _ctx(m, START)
    b = hs.HarInverseVolHold(har, exposure=hs.HarExposure(3, 250, 0.25, 0.95))
    assert b(ctx) == baselines.scaled(baselines.InverseVolHold, hs.E_REF)()(ctx)
    assert b.entry["exposure"] == hs.E_REF and "e_ref" in b.fallbacks[0]


def test_the_typical_level_reads_no_forecast_dated_after_the_entry_day():
    """The median is over history, where one row too many hides: tomorrow's forecast in
    today's median nudges the exposure in every window, and no single lookup shows it.
    The door serves rows up to the decision's own day, and raises for any other day."""
    day = DAYS[39]
    base = [1e-4] * 30 + [2e-4] * 10
    e = hs.HarExposure(3, 60, 0.25, 0.95)
    calm = _har(base + [1e-6] * 20, days=DAYS[:60])
    wild = _har(base + [1e-1] * 20, days=DAYS[:60])
    assert e(calm, day, _deadline(day)) == e(wild, day, _deadline(day)) == e(_har(base), day, _deadline(day))
    with pytest.raises(compiler.LookAheadError):
        calm.trailing(day, _deadline(DAYS[38]), 60, 3)


@pytest.mark.parametrize("mode", ["har", "mean", "min"])
def test_the_rule_desks_har_modes_combine_the_hmm_and_har_exposures_as_named(mode):
    """"min" must never buy more than the HMM asked for, "mean" sits between the two,
    and "har" ignores the HMM. A mode wired to the wrong number would still trade a
    plausible book."""
    m, har = _world()
    b = hs.HarRiskParity(har, None, mode, hs.HarExposure(3, SHORT, 0.25, 0.95))
    w = b(_ctx(m, START))
    rule, h = b.entry["regime"], b.entry["har"]
    assert h is not None and abs(h - rule) > 0.01 and not b.fallbacks
    want = {"har": h, "mean": (rule + h) / 2, "min": min(rule, h)}[mode]
    assert b.entry["exposure"] == pytest.approx(want)
    assert sum(w.values()) == pytest.approx(want, abs=1e-4)


# ----------------------------------------------------------------------------- no look-ahead

@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_the_entry_book_is_unchanged_when_every_later_bar_is_rewritten(name):
    """Entry reads the forecasts for its own session, made before it opened, and a median
    of the sessions before. The forecast for the next session already holds this one's
    variance, so an off-by-one would size on the day being traded. Rewriting every bar
    from the entry session on (prices x3 from its open) must leave the entry as it was."""
    got, logs = [], []
    for shock in (None, START):
        m, har = _world(_bars(DAYS, 8, shock_from=shock))
        b = VARIANTS[name](har)()
        got.append(b(_ctx(m, START)))
        logs.append(b.fallbacks)
    assert got[0] is not None and got[0] == got[1]
    assert logs == [[], []]  # every HAR input was really read, none fell back
