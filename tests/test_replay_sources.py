"""Headlines and 8-K texts for real-names replays (icaif/alpaca_news.py, icaif/filings.py)."""

import fnmatch
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from icaif import alpaca_news, calendar, filings, news

ET = calendar.TZ


def _story(i, created, symbols, updated=None, headline="Apple beats"):
    return {"id": i, "created_at": created, "updated_at": updated or created, "headline": headline,
            "summary": "sum", "url": f"https://x/{i}", "symbols": symbols}


def test_a_story_counts_from_its_last_edit_so_edited_text_is_never_read_before_it_existed():
    """Alpaca returns each story as edited; stamped by its first publication, a story
    rewritten two days later would show the rewrite to rounds that ran before it."""
    rows = alpaca_news.to_rows([_story(1, "2026-01-27T10:00:00Z", ["AAPL"], "2026-01-29T10:00:00Z")],
                               ["AAPL"])
    r = rows.iloc[0]
    assert r["published"] == pd.Timestamp("2026-01-27T10:00:00Z").tz_convert(ET)
    assert r["fetched_at"] == pd.Timestamp("2026-01-29T10:00:00Z").tz_convert(ET)
    assert list(rows.columns) == news.COLUMNS


def test_a_story_tagged_with_several_names_is_one_row_per_competition_name_only():
    """A story in several of our names' feeds reaches each, as a Yahoo story does; a tag
    for a name outside the 30 (SFTBY) would add a ticker no desk can hold."""
    rows = alpaca_news.to_rows([_story(1, "2026-01-27T10:00:00Z", ["MSFT", "SFTBY", "AAPL"]),
                                _story(1, "2026-01-27T10:00:00Z", ["MSFT"]),     # a repeat page
                                _story(2, "2026-01-27T11:00:00Z", ["AAPL"], headline="  ")],
                               ["AAPL", "MSFT"])
    assert sorted(rows["ticker"]) == ["AAPL", "MSFT"] and set(rows["guid"]) == {"alpaca:1"}


def test_an_alpaca_archive_shows_a_round_only_the_stories_known_by_its_deadline(tmp_path):
    """Read through the same `news.known_at` as the Yahoo archive: a story known at
    09:30 is invisible to the 09:10 round and there for the 10:10 one."""
    rows = alpaca_news.to_rows([_story(1, "2026-01-27T11:00:00Z", ["AAPL"]),       # 06:00 ET
                                _story(2, "2026-01-27T14:30:00Z", ["AAPL"])],      # 09:30 ET
                               ["AAPL"])
    alpaca_news.save(rows, pd.Timestamp("2026-01-26", tz=ET), pd.Timestamp("2026-01-28", tz=ET),
                     ["AAPL"], tmp_path)
    at = lambda hhmm: pd.Timestamp(f"2026-01-27 {hhmm}", tz=ET)  # noqa: E731
    assert set(news.known_at(at("09:10"), tmp_path)["guid"]) == {"alpaca:1"}
    assert set(news.known_at(at("10:10"), tmp_path)["guid"]) == {"alpaca:1", "alpaca:2"}
    assert news.known_at(at("05:00"), tmp_path).empty


def test_a_window_outside_every_fetched_span_is_named_not_replayed_as_no_news(tmp_path):
    """No rows for a span we never fetched looks exactly like a quiet market."""
    lo, hi = pd.Timestamp("2026-01-18", tz=ET), pd.Timestamp("2026-02-10 16:00", tz=ET)
    alpaca_news.save(alpaca_news.to_rows([], ["AAPL"]), lo, hi, ["AAPL", "MSFT"], tmp_path)
    assert alpaca_news.uncovered(lo, hi, ["AAPL", "MSFT"], tmp_path) is None
    assert "no Alpaca news" in alpaca_news.uncovered(lo, hi + pd.Timedelta(days=1), ["AAPL"], tmp_path)
    assert "no Alpaca news" in alpaca_news.uncovered(lo, hi, ["AAPL", "TSLA"], tmp_path)
    assert "nothing" in alpaca_news.uncovered(lo, hi, ["AAPL"], tmp_path / "empty")
    with pytest.raises(FileExistsError):
        alpaca_news.save(alpaca_news.to_rows([], ["AAPL"]), lo, hi, ["AAPL"], tmp_path)


INDEX = """<table><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/789019/000119312526027198/msft-20260128.htm">msft-20260128.htm</a></td><td>8-K</td><td>62244</td></tr>
<tr><td>2</td><td>EX-99.1</td><td><a href="/Archives/edgar/data/789019/000119312526027198/msft-ex99_1.htm">msft-ex99_1.htm</a></td><td>EX-99.1</td><td>903665</td></tr>
</table>"""


class FakeEdgar:
    """Stands in for the httpx client: serves a filing index and two documents."""

    def __init__(self, index=INDEX):
        self.index, self.urls = index, []

    def get(self, url):
        self.urls.append(url)
        if url.endswith("-index.htm"):
            if self.index is None:
                raise RuntimeError("index unreadable")
            body = self.index
        elif url.endswith("ex99_1.htm"):
            body = "<html><p>EX-99.1 2 msft-ex99_1.htm EX-99.1 Exhibit 99.1 Revenue was $81.3 billion</p></html>"
        else:
            body = "<html><p>Item 2.02. A press release is furnished as Exhibit 99.1.</p></html>"
        return SimpleNamespace(raise_for_status=lambda: SimpleNamespace(text=body))


def test_an_earnings_8k_is_read_from_its_press_release_not_its_cover_note():
    """Item 2.02's own document only points at Exhibit 99.1: read alone, every results
    day gave the analyst a boilerplate sentence instead of the numbers."""
    got = filings.filing_text(FakeEdgar(), 789019, "0001193125-26-027198", "msft-20260128.htm",
                              "2.02", sleep=0)
    assert got == "Exhibit 99.1 Revenue was $81.3 billion"
    assert filings.exhibits(INDEX) == {"8-K": "msft-20260128.htm", "EX-99.1": "msft-ex99_1.htm"}


def test_a_press_release_labelled_plain_ex99_is_still_the_one_read():
    """GE, NextEra and Pfizer file theirs as EX-99; matched on 99.1 alone, their results
    days showed the analyst the cover note."""
    assert filings.press_release({"8-K": "a.htm", "EX-99": "ge-release.htm", "EX-101.SCH": "x"}) == "ge-release.htm"
    assert filings.press_release({"EX-99.2": "b.htm", "EX-99.1": "a.htm", "EX-99": "c.htm"}) == "a.htm"
    assert filings.press_release({"EX-99.3": "c.htm", "EX-99.2": "b.htm"}) == "b.htm"
    assert filings.press_release({"8-K": "a.htm", "EX-10.1": "contract.htm"}) is None


def test_other_8ks_and_an_unreadable_index_fall_back_to_the_main_document():
    """A departure (5.02) says what happened in its own document; and an index that
    cannot be read still leaves the filing's own words, rather than none."""
    for client, items in ((FakeEdgar(), "5.02"), (FakeEdgar(index=None), "2.02"),
                          (FakeEdgar(index="<table></table>"), "2.02,9.01")):
        got = filings.filing_text(client, 1, "0000000000-26-000001", "main.htm", items, sleep=0)
        assert got.startswith("Item 2.02. A press release")
    c = FakeEdgar()
    filings.filing_text(c, 1, "0000000000-26-000001", "main.htm", "5.02", sleep=0)
    assert not any(u.endswith("-index.htm") for u in c.urls)


def test_replay_texts_never_match_the_8k_snapshot_pattern_and_a_missing_span_raises(tmp_path):
    """The snapshot loaders take the last `edgar_8k_*.parquet` by name; a texts file
    caught by that glob would replace the 8-K history with a few weeks of filings."""
    lo, hi = pd.Timestamp("2026-01-14 09:10", tz=ET), pd.Timestamp("2026-02-10 15:25", tz=ET)
    p = filings.texts_path(lo, hi, tmp_path)
    assert not fnmatch.fnmatch(p.name, "edgar_8k_*.parquet")
    assert filings.TEXTS_DIR.name != "external"
    pd.DataFrame({"ticker": ["MSFT"], "text": ["x"]}).to_parquet(p)
    assert filings.load_texts(lo + pd.Timedelta(days=7), hi, tmp_path)["text"].tolist() == ["x"]
    with pytest.raises(FileNotFoundError):
        filings.load_texts(lo - pd.Timedelta(minutes=1), hi, tmp_path)


def test_a_real_names_replay_stops_on_a_window_whose_news_was_never_fetched(tmp_path, monkeypatch):
    """Otherwise it scores a desk shown no headlines and no filing text, which reads as
    an LLM that ignored the news."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import agent_replay

    days = [d.date() for d in pd.bdate_range("2026-01-21", periods=15)]
    market = SimpleNamespace(days=days, tickers=["AAPL"])
    monkeypatch.setattr(alpaca_news, "ARCHIVE", tmp_path)
    monkeypatch.setattr(alpaca_news, "uncovered",
                        lambda lo, hi, t, directory=tmp_path: "no Alpaca news fetched")
    with pytest.raises(SystemExit, match="no Alpaca news"):
        agent_replay.real_name_sources(market, [days[0]], pd.DataFrame())
    monkeypatch.setattr(alpaca_news, "uncovered", lambda *a, **k: None)
    monkeypatch.setattr(filings, "TEXTS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no 8-K texts"):
        agent_replay.real_name_sources(market, [days[0]], pd.DataFrame())
