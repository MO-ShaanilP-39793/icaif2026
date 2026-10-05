# ICAIF 2026 Trading Agent Competition

Entry for [Codabench competition 99](https://hackathon2.deepintomlf.ai/competitions/99/):
hourly long-only target weights for 30 US large caps, $1M, 0.1% cost on traded
notional, ≤30% per name, ≤100% gross. Its own private GitHub repo, cloned into an
alphaBT checkout as `icaif2026/` so alphaBT is at hand for context. It imports nothing
from `src/`, and alphaBT never tracks it (see Setup), so CodeCommit never sees it.

`starter-kit/` is the organizers' kit vendored verbatim from
[DeepIntoStreams/2026ICAIF_Trading_Agent_Competition](https://github.com/DeepIntoStreams/2026ICAIF_Trading_Agent_Competition)
at `ad237b0`. Do not edit it; write our code beside it, so an upstream refresh is a
plain directory replace.

## Dates (ET; IST = ET + 9:30 in October)

| | ET | IST |
|---|---|---|
| Register to enter Validation | before Oct 8 00:00 | Oct 8 09:30 |
| Live Validation (14 rounds, unscored) | Oct 8–9 | |
| Official registration cutoff | Oct 12 00:00 | Oct 12 09:30 |
| Official (105 rounds) | Oct 12–30 | |
| Final materials | Oct 30 16:00 → Nov 3 23:59 | |

Round deadlines run 09:10 → 15:25 ET, i.e. **18:40 → 00:55 IST**, every trading day.
Submission has to be automated (`tools/auto_submit.py watch`), and the rules forbid
hand-editing agent weights anyway.

## Scoring shapes the strategy

Final rank = mean of four per-metric ranks: cumulative return ↑, Sharpe ↑ (on the
105 round returns, √1764 annualised), max drawdown ↓, **turnover ↓**. Two of four
reward doing little: an all-cash book ties for first on MDD and turnover. Round 7's
return spans the overnight gap, so it dominates round-return variance.

## Traps in the data

- **Historical bars are on a different grid from live execution.** The parquet's
  bars start 09:30, 10:00, 11:00 … 15:00 (the first is a half hour). Live decisions
  execute at the opens of 09:30, **10:30, 11:30 … 15:30**. A backtest filling at
  "next bar open" trades 30 minutes off the live schedule — plausible Sharpe, wrong
  fills. Only the 09:30 execution is exactly observable; mid-day fills must be
  approximated from the bar that contains them, and the approximation should be
  stated wherever a backtest number is shown.
- Prices are **split-adjusted but not spin-off-adjusted**. T 2022-04-11 and GE
  2023-01-04 / 2024-04-02 each gap down ~20% on a distribution, not a loss.
  `data.CORPORATE_ACTIONS` back-adjusts them; a scan of 2021–25 against Yahoo found
  no other permanent step.
- **Early-close days carry extended-hours bars** (13:00–15:00, plus odd 14:30/15:30
  prints) on all 10 half-days. The loader keeps the regular session only; the 16:00
  "close" of a half-day was otherwise a thin post-market trade.
- 85 ticker-days across 5 days (2021-04-19, 2021-10-25, 2022-01-24, 2022-01-26,
  2022-03-08) are missing bars; recorded in `DataIssues`, not filled.
- **A float weight can fail the cap.** The backend checks `Decimal(str(w)) <= 0.30`,
  so a computed `0.1 + 0.2` (`0.30000000000000004`) makes the whole decision invalid
  and the round silently holds. `sim.run` validates through the kit's own contract,
  so this shows in backtests as `invalid_rounds`.
- **Yahoo has no 12:30–13:00 bar on half-days**, and gaps on 2026-01-30 / 02-02.
  `sim.market_from_public_60m` stands in the last close and lists the day in
  `issues["degraded_days"]`; windows touching one should be skipped, not scored.
- **Sizing against the fee is unconfirmed.** `sim.run(sizing="pre_fee")` (default)
  sizes targets on pre-fee NAV, leaving cash −0.1% × notional when fully invested;
  `post_fee` covers the fee. Validation receipts will show which the backend does.
- The kit ships a **metrics calculator, not a backtester**. `kit/evaluation.py` is
  the official formula (self-check passes); the simulator that produces its inputs
  is ours to write.

## Baselines vs the field (`tools/baselines_report.py --exposure-scan`)

170 non-overlapping 15-day windows, Jan 2016 – Sep 2026. Fills are Alpaca's :30 opens
(the organizer's vendor), and windows touching one of 11 degraded days are skipped.
Ranked by the official rules; lower mean overall score is better. SE is 0.04–0.11,
about half what the earlier 40-window Yahoo run allowed.

| Strategy | Mean score (SE) | Mean return | Mean MDD | Mean turnover |
| --- | --- | --- | --- | --- |
| inv_vol_hold at 75% gross | **2.71** (0.05) | 0.68% | 2.56% | 0.72% |
| cash | 2.84 (0.09) | 0 | 0 | 0 |
| inv_vol_hold | 3.03 (0.06) | 0.91% | 3.40% | 0.96% |
| ew_hold | 3.39 (0.05) | 1.02% | 3.73% | 0.96% |
| kit_momentum_hourly | 7.41 (0.07) | −4.39% | 7.87% | 61.0% |

The 40-window Yahoo run (Nov 2023 on) had cash first at 2.69 and 75% gross at 2.74,
too close to call. With 4x the windows, **75% gross beats cash**. Cash still wins
drawdown and turnover outright in every window, but a partly invested hold now takes
enough return and Sharpe rank to overtake it. Scaling gross exposure leaves Sharpe
flat (1.82–1.86) while return, MDD and turnover scale linearly. The field is our
guess at the rivals; every conclusion here is conditional on it.
`--fills yahoo` reproduces the earlier setup.

## Holdout harness (`tools/holdout_eval.py`)

This scores any agent's decisions on Jan 2 – Jun 30 2026 (`holdout.HOLDOUT_START/END`)
in each of the 109 rolling 15-day windows, which amount to about 8 independent samples.
The contest starts every entrant from $1M in cash, so the file holds **one run per
window**: the agent run from cash at that window's first round, for its 15 trading days.

```json
{"strategy": "my_agent",
 "windows": {"2026-01-02": [{"round_id": "holdout-2026-01-02-r1", "cash": 0.25,
                             "weights": {"AAPL": 0.03, "...": "all 30 symbols"}}, ...],
             "2026-01-05": [...], ...}}
```

`tools/holdout_template.py` writes an equal-weight file with every window key. With
`--rebalance once` (the default), each window buys 1/30 each at its first round and
holds, and scores exactly as the board's `ew_hold` reference. `--rebalance every` names
every round: 11,445 decisions, 6.6 MB. Half-days have only rounds 1–4. There is no
six-month continuous run: the contest never scores one.

- **These reject the file:** a missing window or a key that starts none, a round that
  doesn't exist or lies outside its window, a duplicate round, a wrong symbol set, or a
  cash weight more than 1e-9 away from 1 − Σw. So does the old one-run `"decisions"`
  file, by name.
- **These hold, as the backend would:** a missing round, or a weight that breaks a rule,
  such as float dust over the 0.30 cap. Both are listed. `--strict` makes them fatal.
- **Why one run per window.** The old format held one six-month run and replayed it
  into every window from cash. An equal-weight hold written that way bought, in a March
  window, the weights that had drifted since 2 Jan, and an agent that decides from its
  own book saw a book it never had. Both scored as strategies that never existed.
- **Held out only from here on.** The baseline field's 170 windows run to Sep 2026, so
  choices made from that report (e.g. 75% gross) have already seen this span.

**The same harness as a private web page (the scorer):** https://huggingface.co/spaces/MO-AI-Inv/icaif2026-holdout.
- **How it runs.** It's a static Space, because Gradio Spaces need a paid HF plan. It runs
  in the browser on Pyodide 0.29.5. A full file (every round of every window) scores
  natively in ~2 s; expect several times that in the browser.
- **What it ships.** `tools/build_holdout_space.py [--push]` rebuilds it from a fixed list
  of files: the harness modules and 2026 fill prices only. It checks that the page's
  entry point matches the CLI before uploading.
- **Parity.** In-browser results agree with native to within 1e-13.
- **Pyodide trap.** Pyodide must load `tzdata` as well. Without it, every `tz_localize`
  retries a failed import, and scoring is 30× slower.

**Leaderboard:** public, and viewable with no login: https://huggingface.co/spaces/MO-AI-Inv/icaif2026-leaderboard.
It ranks prospective strategies against each other the way the contest does. Every entry
is ranked in each of the 109 windows, and the board is ordered by mean Overall Rank Score,
with the SE computed on ~8 independent windows. Each row shows each metric's distribution
across windows as a histogram. Bins are shared down a column, so shapes compare row to row.
- **References.** Three strategies are always on the board: cash, ew_hold and
  inv_vol_hold_75. They are scored natively at build time, each run fresh in every window.
- **Submitting needs no sign-in.** Score a file on the private scorer, then use *Submit to
  leaderboard* with a name and a note. The page writes with the scorer Space's
  `SUBMIT_TOKEN` variable, a fine-grained token limited to the entry dataset and the
  board. Or run `tools/holdout_eval.py --decisions F --submit --note "..."`. Only the
  board's span and `pre_fee` sizing are accepted.
- **In-repo strategies** are submitted with `tools/submit_strategy.py NAME [--dry]`, which
  runs them fresh in every window as the references are: the same thing a decisions
  file now writes down. First entry: `model_tilt_0.5` ranked 3rd of 4 (2.74 ± 0.14),
  behind cash and inv_vol_hold_75, despite the best six-month Sharpe (1.67).
- **Versions.** Only the newest version of a name ranks. Older ones are listed, so the
  number of looks at the holdout stays visible. Entries scored under the old one-run
  format (schema 1) are listed as "old format" and never ranked; resubmit them.
- **Three repos, fixed visibility** (`icaif/space_hub.py`):
  - the scorer Space is private, because it carries Alpaca prices and the kit;
  - the entry dataset `MO-AI-Inv/icaif2026-holdout-entries` is private and is the record;
  - the board Space is public and ships an exact allowlist: ranking code, references,
    entries. A price or kit file in its build stops the deploy.
- **Office network.** The board's copy of each entry is the only one a page can read
  there: Netskope blocks authenticated HF downloads. If that copy fails, run
  `tools/build_holdout_space.py --sync` off that network.

Equal weight bought at each window's first round and held (`ew_hold`, the sanity
baseline) scores a mean window return of 0.67% and a median of −0.11%, with a median
window drawdown of 2.40% and turnover of 0.95% (one full buy from cash) in every window.

## Features and labels (`tools/feature_report.py`)

`features.build`: 21 per-ticker features (returns over 1/2/5/10/20 sessions, raw and
sector-relative; vol 5d/20d, vol ratio, Parkinson, prior gap, overnight variance
share; volume vs 20d and vs the same clock time; distance from the 20d high/low and
the 5d mean in σ). They are cross-sectionally ranked to [−0.5, 0.5] and joined with 7
raw `ctx_*` market features. All are defined in sessions or clock time, because the
information grid changes on 2026-01-01. `labels.build` builds two targets:
- alphabt-features' composite (reward-to-risk 0.4, terminal 0.3, path Sharpe 0.3), over 7, 21 and 35 rounds;
- alphaBT's Target 2 rescaled ("upside on fills": the mean of the top-k fills against entry), over 21 and 35 rounds.

Each comes as a [0, 1] percentile, the regression target, and as a top-40% binary for
ablation. Label fills are exact Yahoo opens from Nov 2023, with nothing standing in for a
missing bar.

**The upside target is mostly a volatility bet.** Trailing volatility ranks it with IC
0.13–0.18 (t 15), ten times any other feature. Upside-only targets reward names that
move, whichever way. That is why alphaBT divides the probability by trailing
volatility, and why that division happens after the model rather than inside it.

Univariate IC, rounds 1 and 4, 2021–2026 (label-shuffle canary: max |IC| 0.004):

| Feature | IC vs 1-day label | IC vs 3-day label | Reading |
| --- | --- | --- | --- |
| dist_low_20d | +0.009 | **+0.017** | names far above their 20d low keep going |
| volume_today_ratio | **−0.015** (t −2.7) | −0.007 | heavy volume so far today → reversal |
| ret_1s | −0.012 | −0.011 | short-term reversal, as expected in mega-caps |

Every single feature is weak (|IC| ≤ 0.017), about 2–4× the canary. That is the
expected size for mega-cap intraday signal, and it leaves the question for step 4:
does the ensemble combine them into the 0.02–0.05 we guessed? The composite label is
not clearly easier to rank than plain forward return (unlike alphaBT's quarterly
upside result). Session features agree across the two grids (r ≥ 0.97); same-day ones
less so (ret_1s 0.96, volume_today 0.94 at round 4), because a :30-grid bar is 30
minutes staler at a given deadline.

## Daily data for the broad model (`tools/enrich_data.py`)

Snapshots are dated in `data/external/`. The training universe on each date is the
top 100 S&P 500 members by trailing dollar volume, plus the 30 competition names
(`universe.build`, about 104 names a day). Membership is point-in-time, from
[fja05680/sp500](https://github.com/fja05680/sp500) (MIT). Context series are VIX,
SPY, 11 sector ETFs and Cboe Treasury yield indices, all from Yahoo, since 1999.

**Survivorship is large before about 2015.** Yahoo keeps only symbols still trading,
so delisted members have no prices. Share of S&P 500 members priced, by year:

| 1999 | 2005 | 2010 | 2015 | 2020 | 2023 | 2026 |
| --- | --- | --- | --- | --- | --- | --- |
| 45% | 53% | 66% | 74% | 87% | 95% | 99% |

The top 100 by dollar volume is likely covered better than the whole index, since
delistings skew small. But the missing names can't be ranked, so we can't measure
that. The daily model therefore treats its training start year as a setting to test,
judged only on 2023+ test years, where coverage is 95% or more. Old data that teaches
survivor behaviour will show up there as worse test IC.

Also known: GOOG and GOOGL are both in the universe (near-duplicate rows in a
cross-section), and a reused symbol would carry the later company's prices.

**Renames and late index changes** (`universe.RENAMES`, `universe.LATE_CHANGES`). The
membership file uses the ticker in force on each date, and Yahoo keys a renamed
company's whole history under its new ticker. So a spell under the old ticker priced
nothing, and the company dropped out of the universe for that spell (Fiserv as FI from
2023-06 to 2025-11). BK→BNY, FI→FISV, MMC→MRSH and SATS→ECHO now map to the current
symbol; a test ties each to the file (the old spell ends the day the new one starts).
The file's source last updated on 2026-09-07, so the September rebalance (effective
2026-09-21: BE, ILMN, P in; TAP, TTD, BLDR out) is added by hand until a refreshed
snapshot has it. BE trades about $3.7bn a day against a top-100 cut near $0.9bn. Without
it, NXPI held its place in the live universe on 2026-10-05, and 8 of the 30's within-30
ranks moved, by up to 2 places. A live score now warns when the membership records no
change since the last quarterly rebalance. The walk-forward predictions and the frozen
model were built before both fixes; they take effect at the next retrain.

The live fetch always lists names Yahoo no longer serves: `live_symbols` also asks for
spells that ended in the last two years. On 2026-10-05 these were 12 takeovers and
take-privates, all gone from Yahoo under any symbol: ANSS, CTLT, CTRA, DAY, DFS, HES,
HOLX, IPG, JNPR, K, MRO and WBA. Before the renames, BK, FI, MMC and SATS were among them.
`scores_meta.json` now keeps them as `ended_spells_unpriced`, apart from
`current_members_unpriced`, the list that can cost the universe a name.

Earnings dates come from EDGAR 8-K item 2.02 acceptance times (`earnings.py`). This
needs `SEC_USER_AGENT="<name> <email>"` in the environment.

## Daily-model features (`tools/daily_feature_report.py`)

`daily_features.build` gives one row per (date, name) in the day's universe: 718k rows,
1999–2026, a median of 104 names a day. It holds 20 per-name features ranked within
the universe, three raw earnings-timing columns, 13 raw `ctx_*` columns and the
`is_competition` flag: 37 in all. Every value
is as of the close of d−1, since the decision is before d's open.

IC against the 5-day composite (label shuffled within each day: max |IC| 0.007):

| Feature | 2001–10 | 2011–19 | 2020–22 | 2023+ |
| --- | --- | --- | --- | --- |
| e_sessions_to_next (earnings within 10 sessions) | +0.021 | +0.030 | +0.017 | **+0.062** |
| mom_12_1 | +0.016 | +0.026 | +0.006 | +0.030 |
| ret_5d (short-term reversal) | −0.027 | −0.006 | −0.017 | −0.015 |
| vol_20d | +0.004 | −0.018 | −0.021 | +0.023 |
| e_last_reaction (post-earnings drift) | +0.018 | +0.019 | −0.004 | 0.000 |

- **The earnings row is measured on a subset, so its size is overstated.**
  `e_sessions_to_next` is NaN unless a release is within 10 sessions, so its IC counts
  only the ~5 names a day with one coming. Among those, a nearer release scores worse
  on the drawdown-aware composite and better on upside (IC −0.066 there): the release
  adds a jump to the path. Across the full cross-section, with "no release" treated as
  a value, its IC is about 0 (tools/intraday_diagnosis.py). It is a risk flag, not a
  ranking signal. Live, it needs an earnings calendar, since EDGAR only records
  releases after they happen.
- **Volatility changes sign by era.** Low volatility won in 2011–22 and high volatility
  since 2023. A model trained on all eras will average that away, which is one more
  reason the training start year is a tested setting.
- **Post-earnings drift has faded** since 2020, as the literature says it has.
- **The upside target is again mostly volatility** (Parkinson IC 0.16–0.20 in every era).

## Live runner (`tools/live_runner.py`, `icaif/runner.py`)

Each round submits the rule desk's book, and the LLM desk shadows it. The rule desk is
the backtested `q_riskparity_entry_regime`. At the phase's first round 1 from cash it
buys risk parity at the regime-blended exposure, then holds. The LLM desk decides on
its own paper book, and its decision is logged beside the submitted one. The fallback
chain is the agent's book, then the rule's, then no submission. `--submit rule` (the
default, and Validation's) puts in the rule's book. `--submit agent` waits for the
Roadmap's step-6 gate.

```bash
.venv/bin/python tools/live_dry_run.py --round 1            # one round's book, now, nothing uploaded
.venv/bin/python tools/live_runner.py rehearse              # today's 7 rounds at real times, dry
.venv/bin/python tools/live_runner.py rehearse --date 2026-09-30 --fast   # a past day in ~40 s
.venv/bin/python tools/live_runner.py portfolio --phase validation        # after registration
.venv/bin/python tools/live_runner.py arm --phase validation              # owner only, at a terminal
.venv/bin/python tools/live_runner.py run --phase validation --live
.venv/bin/python tools/live_runner.py status --phase validation
.venv/bin/python tools/live_runner.py journal --phase validation          # what each desk's roles are shown
```

- **Nothing uploads unless armed.** An upload needs `--live` and
  `starter-kit/.icaif/ARMED.json`. That file names one phase and one submit mode, and
  it expires an hour after the phase's last close. Only `arm` writes it: at a terminal,
  after the server's portfolio parses, with the phase name typed back. A dry run's
  file carries the kit's placeholders and a `dryrun-` round id. The kit refuses it
  locally.
- **A hold is never a file the kit can upload.** A hold is written as `hold.json`, and
  the kit uploads only a file named `decision.json`. A trade must pass `runner.guard`
  first. The rule trades once a phase, at round 1 from cash. A trade below 0.5%
  summed turnover is drift. Re-submitting the book as weights would pay the fee on
  every name's drift.
- **Entered means entered.** Any held share, or an entry the kit couldn't confirm
  (`ambiguous`), blocks a second entry. An INVALID receipt frees the next round 1.
  A receipt that was pending when the upload returned is re-read at the next round 1.
- **The watchdog.** Each round runs in its own process, and so does the daily model's
  scoring. Both are killed at a deadline, together with their process group. The
  LightGBM/torch deadlock was a hang at 0% CPU with no error, which a thread timeout
  can't interrupt. Scoring runs after the upload, and only the shadow reads its
  scores, so a hang there never touches the submitted book.
- **The clock.** It runs on the server's schedule and clock, waking 12 minutes before
  each deadline. The schedule is re-read every 10 minutes while waiting, so a
  cancellation or a moved deadline is caught. No upload starts within 45 s of a
  deadline. On macOS the runner holds `caffeinate -ims` while it runs, but a closed
  lid on battery still sleeps.
- **What a round sees.** It reads Yahoo daily closes for the 30 names, about 1,200
  days, and every name must have the latest session's bar. It also reads Yahoo 30m
  bars, pairing today's into the backtest's 60m grid for the event trigger. Paper
  books fill at the 30m bar's :30 open through `sim.rebalance`, the backtest's own
  rule. A session still trading is never read as a daily close. For the shadow's HAR
  forecast it reads the `data/public` archive and 60 days of Yahoo 30m bars, once a
  day (`output/live/<phase>/vol/<day>/`). Without the archive the forecast is left out
  rather than refit on two months.
- **Parity with the backtest.** On 7 past entry days (Oct 2025 to Sep 2026) the live
  rule on Yahoo closes and the research desk on Alpaca bars agree closely. Gross
  differs by at most 0.0011, the largest single-name gap is 0.0016, and the summed
  difference is at most 0.015. The turbulent 2025-10-13 (p = 0.78, gross 0.42) agreed
  as well.
- **Timing.** A dry round 1 takes 2.7 s without scoring. With the daily model scored it
  took 21 s on the stale earnings snapshot, and 89 s live in the Oct 1 rehearsal, 76 s
  of it the scorer asking EDGAR for recent filings (its limit is 240 s). Rounds 2-7
  reuse the day's scores and took 1.4-3.1 s. A fast rehearsal of 2026-09-30 ran 7
  worker processes in 36 s.
- **Restarts.** A round commits once, at its end: one atomic write of state.json holds
  the entry, both paper books and both desks' journals. Every file is written whole
  (temp, fsync, rename), so a worker killed mid-write leaves the last good file and its
  round runs again from the last commit. After a crash, start the same command again
  ("Portfolio memory" has the drill).

The runner's environment needs `CODABENCH_TOKEN` and `ICAIF_PROFILE` (in
`starter-kit/.env`). It also needs `SEC_USER_AGENT` for fresh EDGAR events in the
scores and for this week's 8-Ks (without it the shadow reads the dated snapshot, and the
new-8-K trigger never fires), and `ANTHROPIC_API_KEY` for the shadow. Without the key, every role falls back
to the rule, and the record says so. Shadow spend is capped at $10 a phase. Each
round's evidence is kept under `output/live/<phase>/`: the inputs, both desks'
answers, every LLM call's observation and answer, and the file it wrote.

**Before arming.** The portfolio's format is unpublished. `portfolio.parse` accepts
one declared shape and raises on anything else, rather than reading an unknown book as
all cash and buying the entry again. After registration, run `live_runner.py
portfolio --phase validation`, and fix `parse` if the shape it prints is different.

## Agent signals (Roadmap step 2; `icaif/agents/signals.py`)

Every role's observation carries our own signals, each served for the decision's day
only (`compiler.DailyPanel`'s door raises on any other day):

| Field | What | Replays | Live |
| --- | --- | --- | --- |
| `vol_ann_har_1d`, `_3d` | HAR forecast per name, and the basket's in `market` | `vol.walk_forward` from 2016-07 | `vol.forecast_next`, once a day |
| `model_score_rank` | daily model's rank among the 30 (1 = best) | walk-forward predictions, 2023 on | frozen 2026 model |
| `earnings_in_sessions` | sessions to the open that reflects the next release | EDGAR releases within 10 sessions | Yahoo calendar |
| `*_at_entry` | the rank and 3-day vol on the entry day, after entry | | |

The levers built on them, all checked by code rather than asked for in the prompt:

- **Exclusions.** The Strategist's `avoid` lists names, each with its signal
  (`model_score`, `earnings`, `volatility`, `filing`, `other`) and a reason, so a
  replay can score exclusions by cause.
- **Black-Litterman views.** `views` is `none`, `light` or `strong`, and applies to
  risk parity only. The risk-parity book is the prior and the score ranks are views
  sized by Grinold's IC x vol x z, at IC 0.03 (the frozen model's last two years among
  the 30). On the 61 entry days with scores, `light` moves a median 10% of the book
  and `strong` 27% (`tools/bl_calibration.py`, 24 s). The Strategist is shown both
  books exactly; `none` is the rule's own path.
- **Rebalance.** The Risk review may `rebalance`, with a reason (`score_change` or
  `vol_change`): the entry's recipe (shape, views, exclusions plus every exit since)
  rebuilt on today's inputs. It is shown the book, its turnover and fee first.
  Rebalances are capped at 2 a window, and one under 2% turnover is a hold. The book is
  built at the gross it trades to, so a book of a few names under the cap lands on its
  exposure. Once every name is out, no rebalance is offered: cash is `set_exposure` 0.

**The views do not pay as a rule** (`tools/views_report.py`, 49 s). A rule desk that
always takes them, over the 61 windows with scores (2023-01 to 2026-08), against the
plain book (difference in score, negative better):

| Views | Default field | No-clone field | Better / worse (default) |
| --- | --- | --- | --- |
| `light` | -0.020 (SE 0.059) | -0.016 (SE 0.049) | 25% / 25% |
| `strong` | +0.008 (SE 0.113) | -0.008 (SE 0.088) | 34% / 46% |

`strong` buys a better return rank (-0.30) with a worse drawdown rank (+0.43), and they
cancel. So a Strategist choosing views needs a reason the rule does not have.

The rule desk reading every input equals `q_riskparity_entry_regime` trade for trade
in all 167 windows (`tools/agent_replay.py --ledgers-only`, 142 s with step 4's journal
check). Each input has a test that rewrites the future and requires the entry's
observation unchanged (`tests/test_signals.py`).

## Day-1 HAR sizing (Roadmap step 3; `tools/har_sizing_report.py`)

**The HAR forecast does not improve the entry, so the desk is unchanged.** Each variant
is bought once and held, so it pays no turnover its reference doesn't
(`icaif/har_sizing.py`). With HAR off, each equals its reference trade for trade.
Settings were chosen on the 146 non-overlapping windows from 2016-10 to 2025, by mean
score on the no-clone field. The choice was committed (`reports/har_sizing_choice.json`,
832878c) and then scored once on the 109 rolling Jan–Jun 2026 windows (8 independent).
Paired score difference on the no-clone field, with its SE (negative is better):

| Variant, chosen settings | 2016–25 vs hold | vs rule | Jan–Jun 2026 vs hold | vs rule |
| --- | --- | --- | --- | --- |
| 1. Weights on HAR 15-session vol | −0.029 (0.021) | −0.038 (0.034) | +0.018 (0.080) | −0.083 (0.241) |
| 2. Exposure 0.75 × typical / forecast: h15, median of 250 sessions, clip [0.6, 0.9] | +0.027 (0.011) | +0.019 (0.032) | 0.000 (0.034) | −0.101 (0.225) |
| 3. Both | −0.022 (0.022) | −0.031 (0.033) | +0.023 (0.078) | −0.078 (0.237) |
| 4a. Rule desk, HAR vols in risk parity | −0.007 (0.030) | −0.015 (0.021) | +0.085 (0.214) | −0.016 (0.057) |
| 4b. Rule desk, HAR exposure instead of the HMM | +0.002 (0.030) | −0.007 (0.013) | +0.110 (0.230) | +0.009 (0.036) |
| 4c. Both inside the rule desk | +0.005 (0.030) | −0.003 (0.024) | +0.092 (0.210) | −0.009 (0.051) |

The references score 2.719 (hold) and 2.728 (rule) on 2016–25, and 2.842 and 2.943 on
the holdout. Their top-3 shares are 92% and 89%, then 94% and 80%. The variants' top-3
shares run 89–94% and 79–95%.

- **The win rule was fixed before scoring** (96f9e40): negative against both references
  on both fields in both splits, and more than 2 SE below zero on the no-clone 2016–25
  windows. Nothing reached 2 SE in selection, so nothing could win.
- **HAR exposure at entry loses.** All 18 settings did worse than a fixed 75%, in both
  eras and on both fields. The more a setting may lean, the more it loses: +0.03 to
  +0.04 at [0.6, 0.9], and +0.09 to +0.10 at [0.5, 0.95] or [0.25, 0.95]. The chosen
  setting loses return rank (+0.06), Sharpe rank (+0.02) and drawdown rank (+0.03). A
  clip up to 0.95 also costs +0.23 of turnover rank. On the holdout the chosen setting changed 10 windows' scores,
  5 for the better and 5 for the worse.
- **HAR weights are the one consistent sign in selection**: −0.018 in 2016–22 and
  −0.051 in 2023–25, but only 1.4 SE overall. On the holdout they are +0.018: 17
  windows better, 18 worse and 74 the same. A few percent of reweighting rarely changes
  a rank. The 15- and 3-session horizons tied exactly, and the tie-break by name took
  h15.
- **The default field flatters every reweighted variant** by about −0.13 against the
  hold, because none of them is a copy of `inv_vol_hold` any more. That is the
  near-clone artefact, not HAR, which is why the choice reads the no-clone field.
- **`--holdout` refuses to run until the choice file is committed.** It records every
  look in `reports/har_sizing_holdout.json` (1 so far). Selection takes 134 s and the
  holdout 80 s.
- **Fallbacks are logged, not hidden.** The 2022-01-31 window lacks forecasts for 6
  names, so variants 1, 3, 4a and 4c bought the reference's shape there. A typical level
  needs 150 forecasts, so the 10 windows before 2017-06 took the reference's exposure.
  No holdout window fell back.

## Portfolio memory (Roadmap step 4; `icaif/agents/journal.py`)

**Each desk keeps a journal of its own book, and every role reads it back.** A round's
entry holds its decisions as the desk logged them (role, brain or rule fallback,
levers, stated reason), what was submitted, its fill once a later round sees it, and
the return since. Each held name has its entry day and price, its weight, its gain since
entry and the best that gain has been. Step 5's trim reads those last two.

- **One journal per desk, so one per book.** In Validation the rule desk's journal is
  the submitted book's, and the shadow agent's is its paper book's. Both live in
  state.json, committed with the paper books they describe. A copy of each sits in
  `journal/<desk>.json` for reading, with each round's P&L since in dollars, and
  `live_runner.py journal --phase P` prints what each desk's roles are shown.
- **The book is the source of truth.** Each round the journal checks the book the round
  starts from (the server's portfolio live) against what it expects: last round's book
  plus each open order's fill, sized by `sim.rebalance` at the 30m opens the paper books
  fill at. A match records the fill. A gap of more than 5 bps of NAV, in one name or in
  cash, is an issue naming the names, and the journal then adopts the book. The desks
  trade from the server's numbers either way. Vendor noise, Yahoo's opens against
  Alpaca's (p99 21 bps of a 3% name, about 0.6 bps of NAV), sits far below that line.
- **What was submitted is on record.** The runner tells each journal what became of its
  decision: dry-run, uploaded, on paper, held by the guard, not armed. So a guard hold
  never reads as a fill that failed, and an upload never reads as a decision nobody made.
- **Point in time.** Fills come through `Market.fill_prices`, which raises for an
  execution at or after the deadline, and marks through `recent_closes`. A round's own
  order has no fill until a later round sees it in the book. A test rewrites every
  later bar and fill and requires the journal and the memory unchanged.
- **Anonymised in replays.** Codes, day numbers and returns only: no ticker, date,
  price level or NAV.
- **Bounded.** `memory` shows the latest 7 rounds in full, with quiet holds folded,
  earlier days a line each, names sold and issues. It is held under 6,000 characters,
  and that is a hard bound: the oldest detail goes first, and a note says what went.

**Measured** (`tools/journal_report.py`, 234 s; `reports/journal_budget.json`). The
memory a role would read at every one of the 105 rounds, in each of the 167 windows:

| Desk | Max chars | Median | p95 | Journal vs ledger |
| --- | --- | --- | --- | --- |
| Rule desk (what Validation submits) | 3,776 | 1,604 | 2,667 | 167 of 167 agree |
| Busy scripted desk (a lever at every chance, 1,000-character reasons) | 5,859 | 4,371 | 5,757 | 167 of 167 agree |
| Worst case (every round the longest answer the schemas allow, and a trade; 12 windows) | 5,873 | 5,568 | 5,859 | no desk to check |

A whole observation runs about 16,600 characters at the median for the rule desk
(18,100 for the busy one, 20,100 at most). The memory is 9.5% of it at the median (20% at
most), or 22% for the busy desk (33% at most). The per-name journal fields add about
2,300 characters when all 30 names are held. The busy desk's memory was trimmed to fit
in 243 of the 3,588 rounds a role was asked. "Agree" means every fill to the cent, every
held name's entry, cost and peak rebuilt from the ledger alone, every stated lever (an
avoided name not bought, an exit sold out, an exposure bought at its level), and no
issue on record. The busy desk's 3,530 fills and 3,359 names sold raised
none.

**Restarts.** The journals are committed with everything else in one atomic write of
state.json, so a round commits all of it or none of it. Every runner file is now
written whole: a temp file, fsync, then a rename. Before, a worker killed while writing
decision.json left a truncated file. Its retry re-uses an existing decision.json by
design, so the kit can match an upload by its bytes, and the kit refuses a truncated
one: the entry would have gone unsubmitted and the book sat a day in cash. A fast
rehearsal now retries a dead worker once, as the live scheduler does.
`tools/restart_drill.py` replays Sep 25-30 (28 rounds) twice through the real
scheduler, each round in its own worker process, on one Yahoo snapshot. One run is
unbroken. In the other, five workers are SIGKILLed halfway through a write (the entry's
decision.json, two state.json commits, a journal copy, rounds.jsonl) and the scheduler
is stopped twice between rounds. Both end in the same state, every round is indexed,
and each of the four journals agrees with its paper ledger, with no issue
(151 s; `reports/restart_drill.json`).

The rule desk reading its journal still equals `q_riskparity_entry_regime` trade for
trade in all 167 windows, and its journal agrees with its ledger in all 167
(`agent_replay.py --ledgers-only`, 142 s). What these checks cannot show is whether an
LLM's own reasons stay consistent with its memory. `journal.verify` checks the levers
a reason came with, not its prose, so that is step 6's paid replay.

## News and profit booking (Roadmap step 5; `icaif/news.py`, `icaif/filings.py`, `icaif/trim.py`)

**What the roles read.** The Risk review and the Event analyst see each held name's
headlines (live only), every role sees each name's 8-Ks of the last 7 days, and a new 8-K
for a held name wakes the Event analyst beside earnings and the 3-sigma move.

| Input | Who reads it | Replays | Live | Point in time by |
| --- | --- | --- | --- | --- |
| Headlines (`headlines`) | review, analyst; held names only | never (they name the company) | the archive, plus a fetch at the round | first fetch that carried it |
| 8-K item labels (`recent_8k_filings`) | every role, every name | dated snapshot, codes only | snapshot + EDGAR's newest | EDGAR acceptance |
| A new 8-K (`triggers[].new_8k`) | analyst, held names | as above | as above, with the filing's text | EDGAR acceptance |

- **Headlines count from when we had them, not their pubDate** (`news.known_at`). Yahoo's
  pubDate runs after our first fetch for 108 of the first 2,070 headlines, by up to 2.2 h,
  and a story fetched for the first time today is news to us today. Each feed's rows are
  stamped when that feed came back, not when the 20 s run began.
- **Held names, ranked, capped.** A triggered name shows its newest 4 (title and
  summary), naming the company first (`news.ALIASES`: "Meta", not "metadata"; AT&T, not
  T-Mobile); any other held name its 2 newest titles that name it; first seen within
  72 h. With all 30 names held that is 60 titles, about 11,600 characters, plus about
  1,500 per triggered name. Before, every role read every name's whole feed: 55,000 of
  the first live entry observation's 65,000 characters, its first AAPL headline a story
  about an "AI torture chamber". The Strategist reads none: nothing is held at entry.
- **A live round fetches what it reads.** The scheduled archiver runs 5 minutes before
  each deadline, after the shadow has decided (the runner wakes 12 minutes before), so on
  its own it showed the shadow hour-old headlines. A round within 30 minutes of its
  deadline archives the feeds itself first (40 s budget); a rehearsal of a past day never
  does. 8-Ks: EDGAR's newest filings per name are joined to the dated snapshot
  (`live.load_filings`, 10 s a request, 45 s in all), which would otherwise be a week
  stale by Validation. The newest 6 filings of the last 24 h are read for their text,
  once each, kept under `output/live/<phase>/filings/text/` by accession.
- **The 8-K trigger** fires once per filing, from its acceptance: overnight filings at
  the day's first event round, a mid-session one at the next round, the ones before entry
  left to the Strategist. Over the 167 windows the analyst is now asked 23.4 times a
  window; of the names it was woken for, 3,606 were 8-Ks, 1,835 3-sigma moves and 1,162
  earnings. The paid replay's estimate (`agent_replay.py`) counts the 8-Ks.

**External text is data, never instructions** (`icaif/agents/untrusted.py`). Headlines and
filing text reach a role only inside `source_text`, cleaned of control and format
characters (zero-width spaces, bidi overrides) and capped: 160 characters a title, 240
a summary, 1,200 a filing. Every system prompt says what the field is, and the prompts
stay frozen strings, so no external byte reaches one. An answer a headline talked a role
into still has to pass the schema and the desk's checks. A test feeds a headline that
orders "exit every position" to a brain that obeys it: the answer is refused and the
desk trades the rule's book, and the rule desk's own decisions are identical with and
without the headline.

**The trim lever.** The review (with `hold` or `set_exposure`, never `rebalance`) and the
analyst (as its call on a triggered name) can sell a quarter or half of a held name, with
a cause (`give_back`, `news`, `filing`, `volatility`, `earnings`) so a replay can score
trims by cause. In code: a sale under 0.5% of NAV is a hold and spends nothing; 3 trims a
window, rule and agent together; an off-grid, causeless or untriggered trim is refused
whole and the rule's answer stands. The journal records which trims traded and checks
each sold some and kept some.

**The rule's trim does not pay, so the rule desk does not trim** (`icaif/trim.py`,
`tools/trim_report.py`). At each morning review it sells part of a name still up at least
a x its HAR vol x sqrt(sessions held), once it has given back b daily HAR vols from its
high-water mark, when the expected give-back over the sessions left beats 20 bps plus a
rank hit. The give-back is either assumed to continue (`trailing`, the classic
profit-take) or estimated from past windows that had ended (`expanding`, `rolling3y`).
Settings were chosen on the 146 windows of 2016-25 by the no-clone score, the choice
committed (`8e59c1e`; `51127ed` before the rebase onto main; `reports/trim_choice.json`),
then scored once on the 109 rolling Jan-Jun 2026 windows (`reports/trim_holdout.json`).
Paired differences on the no-clone field, SE in brackets, negative better:

| Variant, chosen settings | 2016-25 vs rule | vs hold | Jan-Jun 2026 vs rule | vs hold | Trims, 2016-25 / 2026 |
| --- | --- | --- | --- | --- | --- |
| trailing a0.5 b2 f0.25 | -0.017 (0.008) | -0.009 (0.029) | 0.000 (0.027) | +0.101 (0.224) | 154 in 84 windows / 139 in 73 |
| expanding a0.5 b1 f0.25 | 0.000 (0.000) | +0.009 (0.031) | 0.000 (0.000) | +0.101 (0.226) | 3 in 1 / none |
| rolling3y a0.5 b2 f0.5 | -0.003 (0.003) | +0.005 (0.030) | +0.018 (0.039) | +0.119 (0.221) | 5 in 4 / 85 in 50 |

The references score 2.728 (rule) and 2.719 (hold) on 2016-25, and 2.943 and 2.842 on
the holdout.

- **The win rule was step 3's**, fixed with the grid: more than 2 SE below zero against
  both references on 2016-25, then negative on both fields on the holdout. The trailing
  take cleared it against the rule (2.25 SE) but not against the hold (0.3 SE), so
  nothing could win. On the holdout it tied the rule exactly: 4 windows better, 3 worse,
  102 the same.
- **Winners that turn went on to rise.** Over 2016-25 a name still up after giving back at
  least a daily sigma from its high gained on average to the window's end: +15 to +340
  bps, depending on how far up and how far off its high it was, and positive in every
  vol tercile and every horizon left. So the estimated give-back is negative almost
  everywhere and `expanding` trims 3 times in 146 windows. Only 2022, 2024 and 2025 leaned
  the other way (within noise), which is what `rolling3y` picked up. It then trimmed 85
  times in 2026 and lost (+0.018).
- **The trims cost no turnover rank here, and may in the real field.** A quarter of one
  name never moves our turnover past a rival's in the modelled field: we sit between cash
  and the holds. The trailing take's selection gain is return (-0.027) and Sharpe
  (-0.034) rank, in 14 of 146 windows. A field with rivals just above our turnover would
  charge for every trim.
- **One extra look, disclosed.** A smoke test of `--holdout` scored three untuned
  settings on the first 4 holdout windows before the choice existed. The choice is an
  argmin over 2016-25 scores and reads nothing of the holdout. The look is recorded in
  `reports/trim_holdout.json`.

A test holds the scored book (`TrimmedRiskParity`) to the desk that would trade it
(`Desk(trim=...)`), trade for trade with trims firing, so a winner would have gone into
the rule desk as scored. None did: the rule's review proposes no trim, and the agent's
trim lever stands as a lever the paid replay and the live shadow must justify.

**How the news is judged.** The model has read 2016-25, so replays can't score its
reading of headlines. They score only what anonymises: 8-K item types with codes and day
numbers, from 2016 (step 6's paid replay). Headline judgement is scored on live rounds
only, from Validation: the shadow decides on its own paper book with the news in front
of it while the rule's book is submitted, and `tools/news_shadow_report.py --phase P
[--prices]` lists every review or analyst call that had news in front of it, what it
answered and why, and the name's move since its fill. A fallback, or a shadow run on the
rule brain, is labelled as the rule's.

The rule desk reading every new input still equals `q_riskparity_entry_regime` trade for
trade in all 167 windows, and its journal agrees with its ledger in all 167
(`agent_replay.py --ledgers-only`, 173 s).

## Credentials

Registration returns `TEAM_ID` and a **one-time team token that is never reset**.
It lands in `starter-kit/.icaif/credentials.json` (0600), on the one host that runs
the submitter; teammates never need it. Everything that can hold it is gitignored
here. Final reproducibility materials are built from this repo alone, never alphaBT.

## Public feed vs organizer panel (`reports/data_parity.json`)

Yahoo 60m bars sit on the **live :30 grid** back to Oct 2023; 30m/5m bars reach only
~60 days back. Snapshots are dated in `data/public/` because the windows roll forward
and cannot be refetched. Measured 2026-09-25, after spin-off adjustment:

| Check | Result |
| --- | --- |
| Daily close, organizer vs Yahoo (16.4k ticker-days) | median 0 bps, p99 11 bps |
| 09:30 open, organizer vs Yahoo | median 0, p95 26 bps; 11% differ > 10 bps |
| Yahoo 60m vs 30m open at :30 | identical (p99 0 bps) |
| Guessing a :30 fill from the containing :00 organizer bar | best is OHLC/4: median 11 bps, p95 45 bps |

So a backtest on the organizer grid carries fill noise the size of the 10 bp cost on
every round-2–7 trade (unbiased, mean 0.3 bps, but not small). And the 09:30 open is
vendor-dependent: which print the organizers fill round 1 at is unknown until
Validation receipts show it.

## Alpaca is the organizer's vendor (`tools/alpaca_report.py`)

Alpaca's free SIP 30m bars, paired into the live 60m grid (`alpaca.to_60m`), match the
organizer panel **exactly**: 0 bps on every 09:30 open, close, high and low over 37.6k
ticker-days, and identical volume. The organizers built their panel from Alpaca. So
Alpaca's :30 opens, 2016 on, are the best estimate of the competition's fill prices,
including the round-1 09:30 open that differs by vendor (Yahoo vs Alpaca :30 opens:
median 0, p95 2, p99 21 bps).

Label fills and the intraday model's information bars now come from Alpaca from 2016
(`markets.label_exec_prices`, `markets.intraday_info_bars`), with Yahoo and then the
organizer guess filling holes. That gives seven training years before the first test
fold instead of two, all on the live grid. The simulator still fills on Yahoo from
Nov 2023. Moving it to Alpaca would extend scoring back to 2016.

## Setup

Clone into an existing alphaBT checkout, then hide the folder from alphaBT. The exclude
line lives in `.git/info/exclude`, which is never committed. Adding `icaif2026/` to
alphaBT's `.gitignore` instead would put a commit about this work on a branch bound for
CodeCommit. Without either, one `git add -A` in alphaBT stages this whole repo, and
`hf`/`cchf` push to a public HuggingFace mirror.

```bash
cd alphaBT
git clone https://github.com/MO-ShaanilP-39793/icaif2026.git icaif2026
echo 'icaif2026/' >> .git/info/exclude
git status --short          # must not list icaif2026/
```

Own venv (pandas 2.3, for AutoGluon later): `python3 -m venv .venv &&
.venv/bin/pip install -r requirements.txt`. Tests: `.venv/bin/python -m pytest -q`.
Data is not committed. `data/hourly_market_data_2021_2026.parquet` comes from the
Codabench Files tab. The dated Yahoo snapshots in `data/public/` cannot be refetched
once their window rolls past, so copy them rather than refetching.

Data, trained models and report CSVs are kept in `s3://shaanil/icaif2026/`, under the
same paths as the repo (about 4 GB, mostly `output/ag/`). With AWS SSO access to that
bucket:

```bash
cd icaif2026
for d in data output reports; do aws s3 sync s3://shaanil/icaif2026/$d $d; done
```

Push new data or a retrain back the same way, source and destination swapped. The
bucket also holds alphaBT production data, so write only under `icaif2026/` and never
use `--delete` there. Credentials (`.env`, `.icaif/`) are never uploaded.

```bash
cd icaif2026/starter-kit
cp .env.example .env && chmod 600 .env   # CODABENCH_TOKEN, ICAIF_PROFILE=profiles/profile99-production.json
python3 tools/evaluate.py examples/evaluation.json
```
