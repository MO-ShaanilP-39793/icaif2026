"""The unanchored levered desk and the free desk's news (icaif/agents/desk.py, free.py)."""

import json

import pandas as pd

from icaif import sim
from icaif.agents import prompts, untrusted
from icaif.agents.brains import BrainError, RuleBrain
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.free import FreeDesk
from icaif.agents.schemas import FreeDecision
from tests.test_agents import START, _mkt
from tests.test_news_events import APPLE, DAY2, _archive, _at, _deadline, _filings


class Recorder:
    """Records what each role was shown; answers with `answer`, or fails."""

    name = "recorder"

    def __init__(self, answer=None):
        self.answer, self.seen = answer, []

    def decide(self, role, system, payload, schema, timeout):
        self.seen.append((role, system, payload))
        if self.answer is None:
            raise BrainError("no answer")
        return self.answer


def test_an_unanchored_desk_is_never_shown_the_rules_answer_and_still_falls_back_on_it():
    """Shown `rule_proposal` with "adopt it unless", a replay measures the rule with an
    LLM allowed to object; the unanchored desk must see neither, in any role. A failed
    answer must still become the rule's, or one bad reply would leave the book in cash."""
    m = _mkt()
    b = Recorder()
    d = Desk(b, DeskConfig(anchored=False))
    got = sim.run(d, m, START, 4)
    roles = {r for r, _, _ in b.seen}
    assert {"entry", "review"} <= roles
    for role, system, payload in b.seen:
        assert "rule_proposal" not in payload and "rule_proposal" not in system
        assert system == prompts.UNANCHORED[role]
    assert all(e["source"] == "fallback" for e in d.log)
    want = sim.run(Desk(RuleBrain(), DeskConfig()), m, START, 4)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)


def test_the_anchored_desk_still_reads_the_rule_and_its_prompt_is_unchanged():
    b = Recorder()
    sim.run(Desk(b, DeskConfig()), _mkt(), START, 2)
    for role, system, payload in b.seen:
        assert "rule_proposal" in payload and system == prompts.SYSTEM[role]


def test_the_unanchored_prompts_differ_from_the_anchored_only_in_the_rule_paragraph():
    """Everything else the roles are told (the game, the evidence, the levers) is the
    same, so a difference between the two runs is the anchor's and nothing else."""
    for role in ("entry", "review", "event"):
        a, u = prompts.SYSTEM[role], prompts.UNANCHORED[role]
        assert a.replace(prompts.ANCHOR, prompts.UNANCHOR).replace(prompts.TRIMS_NOTE, "") == u
        assert prompts.system(role, anchored=False) == u and prompts.system(role) == a


def test_a_desk_without_evidence_is_told_no_backtest_finding_and_nothing_else_changes():
    """Told that holding wins, the unanchored desk held through all 86 questions of its
    window; a run without the evidence must differ from it in the evidence alone, or the
    comparison blames the wrong thing."""
    import pytest

    b = Recorder()
    sim.run(Desk(b, DeskConfig(anchored=False, evidence=False)), _mkt(), START, 3)
    for role, system, payload in b.seen:
        assert system == prompts.NO_EVIDENCE[role] and "rule_proposal" not in payload
        assert not any(w in system for w in ("backtest", "measured", "out of sample", "wins"))
    for role in ("entry", "review", "event"):
        rebuilt = prompts.UNANCHORED[role]
        for old, new in prompts.NO_EVIDENCE_EDITS:
            rebuilt = rebuilt.replace(old, new)
        assert rebuilt == prompts.NO_EVIDENCE[role]
        assert "The ranking is NOT return" in prompts.NO_EVIDENCE[role]   # the game stays
    assert "20 bps" in prompts.NO_EVIDENCE["review"]                       # and its costs
    with pytest.raises(ValueError):
        prompts.system("entry", anchored=True, evidence=False)


HOLD = FreeDecision(action="hold", weights=[], rationale="hold")


def _free_payloads(tmp_path, *, anonymize=False, ev=None, n=3):
    _archive(tmp_path)
    b = Recorder(HOLD)
    d = FreeDesk(b, "blank", DeskConfig(anonymize=anonymize), news_dir=tmp_path,
                 filings=ev if ev is not None else _filings())
    sim.run(d, _mkt(), START, n)
    return {p["clock"]["day"]: p for _, _, p in b.seen}


def test_the_free_desk_reads_headlines_for_names_it_does_not_hold(tmp_path):
    """It can buy any of the 30; shown only held names' news, it would choose a book
    blind to the news on everything it did not already own (here: all of it)."""
    seen = _free_payloads(tmp_path)
    names = {r["name"]: r for r in seen[2]["names"]}
    assert [h[untrusted.FIELD]["title"] for h in names[APPLE]["headlines"]] == [
        "Apple names a new finance chief", "Apple supplier warns"]
    assert [h[untrusted.FIELD]["title"] for h in names["MSFT"]["headlines"]] == [
        "Microsoft and OpenAI talk"]
    anon = _free_payloads(tmp_path / "anon", anonymize=True)
    assert "headlines" not in json.dumps(anon) and "new_filings" not in json.dumps(anon)


def test_the_free_desk_reads_each_new_8k_once_with_its_text_and_none_before_acceptance(tmp_path):
    """A filing accepted after a morning's deadline is the next morning's news; read on
    the morning it was accepted after, it would be a look into the future."""
    ev = _filings((APPLE, _at(DAY2, "07:00"), "2.02"),
                  (APPLE, _deadline(DAY2, 1) + pd.Timedelta(minutes=5), "5.02"))
    ev["text"] = ["Apple revenue rose 8%", "Apple's CFO resigns"]
    seen = _free_payloads(tmp_path, ev=ev)
    day2 = seen[2]["new_filings"][APPLE]
    assert [f[untrusted.FIELD] for f in day2] == ["Apple revenue rose 8%"]
    day3 = seen[3]["new_filings"][APPLE]
    assert [f[untrusted.FIELD] for f in day3] == ["Apple's CFO resigns"]
    assert "new_filings" not in seen[1] or APPLE not in seen[1]["new_filings"]


def test_the_free_desk_never_changes_the_config_it_was_given():
    """One config is shared by every window's desk in a replay; a free desk adding its
    role to it would hand the next levered desk a role list it never asked for."""
    cfg = DeskConfig()
    before = (dict(cfg.timeouts), tuple(cfg.headline_roles))
    FreeDesk(Recorder(HOLD), "blank", cfg)
    assert (dict(cfg.timeouts), tuple(cfg.headline_roles)) == before


def test_without_the_regime_model_no_role_sees_its_read_and_every_raw_reading_stays():
    """The free desk trusted a "calm" label through a 3% sell-off. Hidden, the label must
    be gone from every role's view, while the readings an LLM could judge for itself
    (basket returns, vol against its median, correlation) stay, or the run tests a
    blinder desk rather than one without the label."""
    from icaif.agents import observe

    seen = {}
    for regime in (True, False):
        b = Recorder()
        sim.run(Desk(b, DeskConfig(anchored=False, regime=regime)), _mkt(), START, 3)
        f = Recorder(HOLD)
        sim.run(FreeDesk(f, "blank", DeskConfig(regime=regime)), _mkt(), START, 2)
        seen[regime] = [p["market"] for _, _, p in b.seen + f.seen]
    assert all(set(observe.REGIME_FIELDS) <= set(m) for m in seen[True])
    for with_, without in zip(seen[True], seen[False]):
        assert set(with_) - set(without) == set(observe.REGIME_FIELDS)
        assert {k: v for k, v in with_.items() if k not in observe.REGIME_FIELDS} == without
