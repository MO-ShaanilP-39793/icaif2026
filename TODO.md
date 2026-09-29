# TODO

Dates are ET. Register before **Oct 8 00:00** to get Validation (Oct 8–9). Official
rounds run Oct 12–30. Final materials are due Nov 3.

Where we stand (2026-09-29): the entry today is the inverse-vol hold at 75% gross. Over
109 rolling 15-day windows in H1 2026 it finished top 3 in 94% of them, but its mean
score (3.21) trails cash (2.99), because cash wins every falling window. The model's
tilt loses to the hold after fees, and the GNN failed its stage-1 gate.

## Owner's decisions and actions

- [ ] Register the team on Codabench before Oct 8 00:00 ET. The team token goes in
      `.icaif/`; it is one-time and never reset.
- [ ] Choose the machine that runs live rounds, Oct 8–30. It needs 7 rounds a day, about
      19:00–01:30 IST, and must not sleep. It must not be alphaBT infrastructure.
- [ ] Decide on an LLM or no LLM, after the first paid replay below.
- [ ] If there is an LLM: apply for Nodexi API credits when the window opens.
- [ ] Set `SEC_USER_AGENT="<name> <email>"` in the runner's environment.
- [ ] Confirm the employer is fine with a public entry and published final materials.
- [ ] Decide whether GNN stage 2 is dropped. Recommended, since stage 1's blend lowered
      IC by 0.006.

## Research (ask before each run)

- [ ] **Correction (2026-09-29):** the ~0.1 score edge of risk parity (and of the
      rule desk, "-0.100, SE 0.039") over `inv_vol_hold_75` is an artefact. The default
      field holds `inv_vol_hold` at 100%, a near-clone of the reference; holding cash
      shaves the clone's Sharpe ~0.4%, so it loses that near-tie in 89% of windows.
      Without the clone every shape is within +-0.02 (2016-22) and +0.01..+0.06 worse
      (2023-26). Re-score quant_report and agent_replay on a no-clone field too.

- [x] Exposure-timing race (`tools/quant_report.py`, `icaif/quant.py`): vol target,
      Grossman-Zhou drawdown control, a 2-state HMM, min variance, risk parity, an OU
      residual tilt. Every rule that trades after entry loses to the hold; risk parity
      decided at entry is the one consistent (small) gain.
- [ ] "Always avoid reporters" as a no-LLM rule, the bar for the earnings analyst.
- [x] Rank-playing controller (`icaif/rankplay.py`, `tools/rankplay_report.py`): each
      morning, the exposure with the best expected final rank against a simulated
      field, over bootstrapped rest-of-window paths. Planned and scored on both fields.
- [ ] Exotic queue, in order: signature features (for the write-up; the organizers
      are the signatures group), Black-Litterman with the model scores as views on the
      risk-parity prior, rough-volatility sizing. Overnight-vs-intraday is ruled out by
      arithmetic: a 1.6 bps/day gap against 20 bps x fraction moved.
- [ ] Refit the 2026 folds after the VIX-holiday context fix. About 15 min.
- [ ] Move the period and rolling-window reports into `tools/period_report.py`.

## Live runner (before Validation, Oct 8)

- [ ] Commit the Yahoo earnings calendar (`icaif/earnings_calendar.py`, its tool and
      tests).
- [ ] Guard that a real round can never upload a "hold" decision by accident.
- [ ] Watchdog on scoring, for the LightGBM/torch OpenMP deadlock that would silently
      miss a round.
- [ ] Intraday bars live, so exits can fire in rounds 2–7.
- [ ] Read the portfolio in the kit's format once registration shows it.
- [ ] A scheduler for the 7 rounds, with the fallback chain: agent, then compiler
      default, then no submission.
- [ ] A rehearsal: a full day of dry-run rounds on the chosen machine.

## Agent (`icaif/agents/`, `tools/agent_replay.py`)

- [x] Harness: point-in-time observation, a structured schema per role, validation,
      a round budget with per-role timeouts, and a fallback to the rule on any failure.
- [x] Rule brain: the desk reproduces `q_riskparity_entry_regime` in all 167 windows.
- [x] Claude brain (allowed models only, no server-side model fallback), a response
      cache and offline replays, and anonymised observations for replays.
- [x] News archiver (`tools/news_archive.py`, Yahoo RSS; yfinance's news endpoint
      answers 500 here and returns []). First snapshot 2026-09-29: 551 headlines, 30 names.
- [x] News archiver scheduled on the owner's Mac (launchd, `tools/install_news_launchd.py`):
      5 minutes before each round's deadline and at 08:00 ET, every day.
- [ ] Rerun `tools/install_news_launchd.py` after 2026-11-01 (US clocks change; launchd
      runs in local time). Keep the Mac awake and online through US market hours.
- [ ] Back up `data/external/news/`: it is the only copy and not in git.
- [x] Macro in the observation (`icaif/macro.py`): SPY, VIX, yields, curve, sectors as
      of the prior close; levels become z-scores in replays. FOMC decisions 2021-27
      from the Fed's page (`tools/macro_calendar.py`); next: **2026-10-28**, inside
      the Official phase, statement at 14:00 ET.
- [ ] Fetch 8-K events (`tools/filings_events.py`): needs `SEC_USER_AGENT`. Then pass
      `filings=` to the desk in `tools/agent_replay.py`.
- [ ] CPI and jobs-report dates (BLS schedules), and FOMC before 2021 (the Fed's
      per-year archive pages, a different layout).
- [ ] First paid replay (owner approves the spend): entry-only (`--no-review`), 2025
      windows, anonymised. It shows whether the Strategist beats its rule at all.
- [ ] Live adapter: the desk on a live round (portfolio and journal from disk),
      writing decision.json through `live.check_decision`, never uploading.
- [ ] Shadow the agent through Validation (Oct 8–9): submit the rule's book, and log
      what the agent would have done.

## Final materials (Nov 3)

- [ ] `disclosures.md`: LLM use, external data (Yahoo, Alpaca, EDGAR), software.
- [ ] Write-up, built from this repo alone, with no alphaBT or client data.
- [ ] Update the design doc's Results tab with the H1 2026 rolling result and the GNN
      gate outcome.
