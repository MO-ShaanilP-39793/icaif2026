from datetime import date

import numpy as np
import pandas as pd

from icaif import memprobe as mp


def _set(actual, guess):
    return pd.DataFrame({"id": [f"T{i} 2026-01-2{i % 9}" for i in range(len(actual))],
                         "actual": actual, "guess": guess})


def test_a_model_that_calls_every_move_up_in_a_rising_window_is_not_taken_to_remember():
    """In a window where most big moves were up, a model that says "up" to everything
    agrees on most signs and knows nothing. Tested against even odds it would be flagged,
    and a clean window dropped; chance comes from the two sides' up/down mixes."""
    actual = [3.1, 4.2, 2.5, 5.0, 3.3, 2.8, 6.1, 4.4, 3.9, -3.0, 2.2, 3.7]
    s = mp.score(_set(actual, [1.5] * len(actual)))
    assert s["sign_right"] == 11 and s["remembered"] is False


def test_a_model_that_recalls_the_moves_is_flagged():
    """The failure the probe exists for: guesses that track the real moves, signs and
    sizes. A replay of such a window scores memory as skill."""
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 4, 18).round(2)
    s = mp.score(_set(actual, actual * 0.8 + rng.normal(0, 0.8, 18)))
    assert s["remembered"] is True and s["corr"] > 0.8 and s["within_1pp"] > 5


def test_a_model_with_no_memory_is_flagged_at_about_the_rate_the_docstring_states():
    """The screen's false-flag rate decides how many clean windows it throws away. Two
    tests at 5% one-sided flag a guesser about 7% of the time a set (the sign test is
    discrete); much more, and the cost the docstring states is a fiction."""
    rng = np.random.default_rng(1)
    flags = [mp.score(_set(rng.normal(0, 4, 20), rng.normal(0, 2, 20)))["remembered"]
             for _ in range(400)]
    assert 0.03 < np.mean(flags) < 0.13


def test_a_line_the_model_skipped_is_left_out_never_filled():
    """A skipped line filled with zero or a mean is a guess the model never made, and it
    moves both the sign count and the correlation."""
    moves = pd.DataFrame({"id": ["AAPL 2026-01-30", "MSFT 2026-01-29"], "actual": [-3.0, 4.0]})
    got = mp.attach(moves, mp.Guesses(guesses=[mp.Guess(id=" AAPL 2026-01-30 ", pct=-2.0),
                                               mp.Guess(id="AAPL 2026-01-30", pct=9.0),
                                               mp.Guess(id="NOPE", pct=1.0)]))
    assert got["guess"].iloc[0] == -2.0 and np.isnan(got["guess"].iloc[1])


def test_a_probe_the_model_did_not_answer_never_clears_a_window():
    """A failed call (an expired AWS login, a refusal) returns no guesses. Read as "no
    edge", it would clear every window without asking the model anything."""
    unanswered = mp.score(_set([1.0] * 8, [np.nan] * 8))
    assert unanswered["remembered"] is None
    assert mp.verdict([unanswered, {"remembered": False}]) == "unanswered"
    assert mp.verdict([{"remembered": False}, {"remembered": True}]) == "remembered"


def test_a_window_with_almost_no_results_is_judged_on_its_largest_moves_alone():
    """Two results reactions can never beat chance. Marked unanswered, a window outside
    earnings season could never be cleared; asked, it would cost a call that tests
    nothing. And a window where nothing was asked is not clean."""
    few = mp.untested(pd.DataFrame({"id": ["A 2026-03-11", "B 2026-03-12"], "actual": [1.0, -2.0]}))
    assert few["asked"] is False
    assert mp.verdict([few, {"remembered": False}]) == "clean"
    assert mp.verdict([few, few]) == "untested"


def test_each_name_reacts_once_on_its_reaction_session_and_previews_fold_into_their_release():
    """A preview filed under item 2.02 a few weeks before results (Tesla's deliveries)
    would ask about a quiet day as if it were results, and two lines for one name count
    its memory twice."""
    ny = "America/New_York"
    table = pd.DataFrame({
        "ticker": ["TSLA", "TSLA", "AAPL", "MSFT"],
        "accepted": pd.to_datetime(["2026-01-02 08:00", "2026-01-28 16:05", "2026-01-29 16:30",
                                    "2026-03-05 16:05"]).tz_localize(ny),
        "session": [date(2026, 1, 2), date(2026, 1, 29), date(2026, 1, 30), date(2026, 3, 6)],
        "pct": [0.01, -0.05, 0.04, 0.02]})
    days = [date(2026, 1, 2), date(2026, 1, 29), date(2026, 1, 30)]
    got = mp.earnings_moves(table, days)
    assert got["id"].tolist() == ["TSLA 2026-01-29", "AAPL 2026-01-30"]
    assert got["actual"].tolist() == [-5.0, 4.0]
    assert "16:05 ET" in got["note"].iloc[0]


def test_a_missing_close_is_never_a_move():
    """A move across a hole, measured from the last close before it, is two days' move
    asked about as one; and a hole read as zero is a quiet day that never happened."""
    idx = pd.to_datetime(["2026-01-02 16:00", "2026-01-05 16:00", "2026-01-06 16:00"]).tz_localize(
        "America/New_York")
    daily = pd.DataFrame({"A": [100.0, np.nan, 120.0], "B": [50.0, 51.0, 49.98]}, index=idx)
    got = mp.largest_moves(daily, [date(2026, 1, 5), date(2026, 1, 6)], n=5)
    assert set(got["ticker"]) == {"B"} and len(got) == 2
