"""The alert watcher (tools/alert_watch.py): what it tells the owner, and when it keeps quiet."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import alert_watch as A  # noqa: E402

R1 = "official-2026-10-12-r1"
OPENS = datetime(2026, 10, 12, 13, 10, tzinfo=timezone.utc)   # round 1's deadline


def _phase(tmp_path, rounds=None, recs=(), entry=None, beat=True):
    d = tmp_path / "official"
    d.mkdir()
    sched = {"rounds": [{"id": R1, "phase": "official", "deadline": "2026-10-12T13:10:00+00:00"},
                        {"id": "official-2026-10-30-r7", "phase": "official",
                         "deadline": "2026-10-30T19:25:00+00:00"}]}
    (d / "schedule.json").write_text(json.dumps(sched))
    (d / "state.json").write_text(json.dumps({"entry": entry, "rounds": rounds or {}}))
    (d / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    if beat:
        (d / "runner.log").write_text("x\n")
    t = (OPENS - timedelta(minutes=12)).timestamp()   # the heartbeat, on the tests' clock
    for f in d.iterdir():
        os.utime(f, (t, t))
    return d


def _watch(d, tmp_path, now=OPENS - timedelta(minutes=10), alive=True, post=None):
    sent = []
    keys = A.watch("official", d, tmp_path / "seen.json", now, alive, post or sent.append)
    return keys, sent


def test_a_refused_entry_alerts_once_not_every_minute(tmp_path):
    """A refused round 1 leaves the book in cash until the next day's round 1; the owner
    must hear at once, and a phone buzzing every minute after would be muted."""
    d = _phase(tmp_path, rounds={R1: {"outcome": "trade (agent) -> upload refused", "upload": "refused"}},
               recs=[{"round_id": R1, "submitted": {"upload": {"status": "refused", "why": "slot closed"}}}])
    keys, sent = _watch(d, tmp_path)
    assert keys == [f"upload:{R1}"] and sent[0]["priority"] == "urgent"
    assert "slot closed" in sent[0]["body"]
    assert _watch(d, tmp_path)[0] == []


def test_a_pm_that_fell_back_is_reported_though_the_round_looks_healthy(tmp_path):
    """On 2026-10-11 the VM's PM got HTTP 429 on every call; round.json showed no error or
    warning and the outcome read "trade (agent)". Only agent.roles said "fallback"."""
    rec = {"round_id": R1, "errors": [], "submitted": {"upload": {"status": "uploaded", "submission_id": 9}},
           "agent": {"gross": 0.1, "names": 30, "roles": [{"role": "pm", "source": "fallback"},
                                                          {"role": "news", "source": "model"}]}}
    d = _phase(tmp_path, rounds={R1: {"outcome": "trade (agent) -> upload uploaded", "upload": "uploaded"}},
               recs=[rec])
    keys, sent = _watch(d, tmp_path)
    assert f"fallback:{R1}" in keys and f"entry:{R1}" in keys
    assert "pm" in next(m for m in sent if m["key"] == f"fallback:{R1}")["title"]
    assert "submission 9, gross 0.100, 30 names" in next(m for m in sent if m["key"] == f"entry:{R1}")["body"]


def test_a_dead_runner_alerts_during_the_phase_and_not_before_it(tmp_path):
    """A dead runner writes no file, so nothing else would ever say it is gone: every
    later round would be missed in silence."""
    d = _phase(tmp_path)
    assert _watch(d, tmp_path, now=OPENS - timedelta(days=1), alive=False)[0] == []
    keys, sent = _watch(d, tmp_path, alive=False)
    assert keys == ["silent"] and "no runner process" in sent[0]["title"]


def test_a_hung_runner_alerts_though_its_process_is_alive(tmp_path):
    d = _phase(tmp_path)
    old = (OPENS - timedelta(hours=1)).timestamp()
    for f in ("runner.log", "schedule.json", "state.json"):
        os.utime(d / f, (old, old))
    keys, sent = _watch(d, tmp_path, alive=True)
    assert keys == ["silent"] and "silent since" in sent[0]["title"]


def test_a_second_outage_alerts_again_after_the_runner_recovers(tmp_path):
    d = _phase(tmp_path)
    assert _watch(d, tmp_path, alive=False)[0] == ["silent"]
    keys, sent = _watch(d, tmp_path, alive=True)
    assert keys == [] and sent[0]["key"] == "back"
    assert _watch(d, tmp_path, alive=False)[0] == ["silent"]


def test_a_failed_send_is_retried_next_minute_not_lost(tmp_path):
    """ntfy unreachable for a minute must not mark the alert as delivered."""
    d = _phase(tmp_path, rounds={R1: {"outcome": "missed: the worker died", "upload": None}})

    def down(msg):
        raise OSError("unreachable")

    assert _watch(d, tmp_path, post=down)[0] == []
    assert _watch(d, tmp_path)[0] == [f"outcome:{R1}"]


def test_holds_and_a_healthy_runner_send_nothing(tmp_path):
    d = _phase(tmp_path, rounds={"official-2026-10-12-r2": {"outcome": "hold: the desk returned no trade",
                                                            "upload": "hold"}})
    assert _watch(d, tmp_path)[0] == []


def test_no_message_carries_the_decision_files_contents(tmp_path):
    """decision.json carries the team token; the watcher reads only state, index and log."""
    d = _phase(tmp_path, rounds={R1: {"outcome": "no submission: guard", "upload": None}})
    (d / R1).mkdir()
    (d / R1 / "decision.json").write_text(json.dumps({"team_token": "SECRET-TOKEN"}))
    _, sent = _watch(d, tmp_path, alive=False)
    assert sent and "SECRET-TOKEN" not in json.dumps(sent)
