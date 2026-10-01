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
- [x] Set `SEC_USER_AGENT="<name> <email>"` on the owner's Mac (2026-10-01, in ~/.bash_profile and ~/.zshrc). The live
      runner's machine needs it too, in its launchd/cron job's own environment.
- [ ] Confirm the employer is fine with a public entry and published final materials.
- [ ] Decide whether GNN stage 2 is dropped. Recommended, since stage 1's blend lowered
      IC by 0.006.
- [ ] After registering, before Oct 8 09:10 ET (the live runner; README "Live runner"):
      put `CODABENCH_TOKEN` and `ICAIF_PROFILE` in `starter-kit/.env`; run
      `tools/live_runner.py portfolio --phase validation` and send its printed shape
      (values blanked) so `portfolio.parse` can be checked; then `arm --phase
      validation` at a terminal and start `run --phase validation --live`.
- [ ] Approve the shadow's spend (Opus 5, capped at $10 a phase; about 3 calls in
      Validation) and set `ANTHROPIC_API_KEY` in the runner's environment, or run it with
      `--shadow rule` and shadow nothing but the rule.

## Research (ask before each run)

- [ ] **Correction (2026-09-29):** the ~0.1 score edge of risk parity (and of the
      rule desk, "-0.100, SE 0.039") over `inv_vol_hold_75` is an artefact. The default
      field holds `inv_vol_hold` at 100%, a near-clone of the reference; holding cash
      shaves the clone's Sharpe ~0.4%, so it loses that near-tie in 89% of windows.
      Without the clone every shape is within +-0.02 (2016-22) and +0.01..+0.06 worse
      (2023-26). Re-score quant_report and agent_replay on a no-clone field too.
      The agent prompts no longer claim the edge (2026-10-01).

- [x] Exposure-timing race (`tools/quant_report.py`, `icaif/quant.py`): vol target,
      Grossman-Zhou drawdown control, a 2-state HMM, min variance, risk parity, an OU
      residual tilt. Every rule that trades after entry loses to the hold; risk parity
      decided at entry is the one consistent (small) gain.
- [ ] "Always avoid reporters" as a no-LLM rule, the bar for the earnings analyst.
- [x] Rank-playing controller (`icaif/rankplay.py`, `tools/rankplay_report.py`): each
      morning, the exposure with the best expected final rank against a simulated
      field, over bootstrapped rest-of-window paths. Planned and scored on both fields.
- [ ] Exotic queue, in order: signature features (for the write-up; the organizers
      are the signatures group), rough-volatility sizing. Black-Litterman is built, as
      the Strategist's `views` lever (Roadmap step 2), but not yet scored as a rule.
      Overnight-vs-intraday is ruled out by arithmetic: a 1.6 bps/day gap against
      20 bps x fraction moved.
- [x] Score a rule desk that always takes `views: light` (and `strong`) against the
      plain one on the 61 windows with scores (`tools/views_report.py`): neither pays.
      light -0.020 (SE 0.059) on the default field, -0.016 (0.049) without the clone;
      strong +0.008 (0.113) and -0.008 (0.088).
- [ ] Refit the 2026 folds after the VIX-holiday context fix. About 15 min.
- [ ] Move the period and rolling-window reports into `tools/period_report.py`.

## Live runner (before Validation, Oct 8)

- [x] Commit the Yahoo earnings calendar (`icaif/earnings_calendar.py`, its tool and
      tests): in a5a0f54.
- [x] The dry run submits the rule desk's book (risk parity at the regime-blended
      exposure at round 1 from cash, then hold), not the compiler's top-10 default.
      Matches the research desk on 7 past entry days (gross within 0.0011).
- [x] Guard that a real round can never upload a "hold" decision by accident: a hold
      is `hold.json`, which the kit cannot upload; `runner.guard` checks every trade.
- [x] Watchdog on scoring, for the LightGBM/torch OpenMP deadlock that would silently
      miss a round: scoring and each round run in killable child processes.
- [x] Intraday bars live, so exits can fire in rounds 2–7 (Yahoo 30m, paired into the
      60m grid; the shadow's Event analyst wakes on a 3-sigma move).
- [ ] Read the portfolio in the kit's format once registration shows it. The strict
      reader (`icaif/portfolio.py`) and the `portfolio` command exist; the shape is a
      guess until the server shows it, and `arm` refuses until it parses.
- [x] A scheduler for the 7 rounds, with the fallback chain: agent, then rule, then no
      submission (`tools/live_runner.py run`).
- [x] A fast rehearsal: 2026-09-30's 7 rounds as worker processes, in 36 s.
- [ ] A rehearsal: a full day of dry-run rounds at real times on the chosen machine
      (`tools/live_runner.py rehearse`, 08:58-15:25 ET).
- [ ] Snapshot the earnings calendar and 8-K filings daily on the runner's machine:
      the shadow reads the latest snapshot (the calendar's is 2026-09-28).
- [ ] A standby host (the Deployment tab's hh:23 check) is not built.
- [x] Restarts (Roadmap step 4): every runner file is written whole (temp, fsync,
      rename), a round commits once (state.json, with both books and both journals),
      and a dead worker is retried in a fast rehearsal too. `tools/restart_drill.py`
      replays Sep 25-30 with 5 workers SIGKILLed mid-write and the scheduler stopped
      twice: the same end state as an unbroken run, every journal agreeing with its book.

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
- [x] Fetch 8-K events (`tools/filings_events.py`): 10,411 for the 30 names on 2026-10-01.
- [ ] Pass `filings=` to the desk in `tools/agent_replay.py`.
- [ ] CPI and jobs-report dates (BLS schedules), and FOMC before 2021 (the Fed's
      per-year archive pages, a different layout).
- [ ] First paid replay (owner approves the spend): entry-only (`--no-review`), 2025
      windows, anonymised. It shows whether the Strategist beats its rule at all.
- [x] Live adapter: the desk on a live round, its state carried between rounds as
      JSON (`Desk.state`/`restore`), writing decision.json through `live.check_decision`,
      uploading only when armed (`icaif/runner.py`).
- [ ] Shadow the agent through Validation (Oct 8–9): submit the rule's book, and log
      what the agent would have done.
- [x] Our signals in every role's observation (Roadmap step 2, `icaif/agents/signals.py`):
      HAR 1- and 3-day vol per name and for the basket, the score's rank among the 30,
      sessions to earnings, and their values at entry, each with a no-look-ahead test.
      The levers: exclusions with their cause, Black-Litterman views on risk parity,
      and a rebalance with a reason (at most 2 a window). The rule desk reading every
      input equals its candidate in all 167 windows (`agent_replay.py --ledgers-only`).
- [ ] What the LLM does with the signals is step 6's paid replay: count views,
      exclusions by signal, and rebalances by reason in its log.
- [x] Portfolio memory (Roadmap step 4, `icaif/agents/journal.py`): a journal per desk,
      so per book (the submitted one and the shadow's paper one), reconciled against the
      book every round, server first. Every role reads it as `memory` (the latest 7
      rounds in full, earlier days a line each, under 6,000 characters) and per held name
      `entry_day`, `gain_since_entry`, `peak_gain_since_entry`, which step 5's trim reads.
      The rule desk with it still equals its candidate in all 167 windows, and its
      journal agrees with its ledger in all 167 (README "Portfolio memory").
- [ ] Whether the LLM's stated reasons stay consistent with its memory is step 6's paid
      replay: the journal makes it checkable (`journal.verify` checks levers, not prose).
- [x] Day-1 HAR sizing (Roadmap step 3, `tools/har_sizing_report.py`): HAR weights,
      HAR entry exposure, both, and each inside the rule desk. Chosen on 2016-25 and
      committed before one look at Jan-Jun 2026. Nothing wins. HAR exposure lost to a
      fixed 75% at every setting, and HAR weights went from -0.029 (1.4 SE) to +0.018
      on the holdout. The desk is unchanged (README "Day-1 HAR sizing").

## Final materials (Nov 3)

- [ ] `disclosures.md`: LLM use, external data (Yahoo, Alpaca, EDGAR), software.
- [ ] Write-up, built from this repo alone, with no alphaBT or client data.
- [ ] Update the design doc's Results tab with the H1 2026 rolling result and the GNN
      gate outcome.
