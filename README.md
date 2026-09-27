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

- **Earnings inside the label window is the strongest single effect.** An upcoming
  release adds a jump to the path, which the drawdown-aware composite penalises and
  the upside target rewards (IC −0.066 there). It depends on release dates being
  announced ahead of time: the feature only looks 10 sessions forward, and live it
  needs an earnings calendar, not EDGAR.
- **Volatility changes sign by era.** Low volatility won in 2011–22 and high volatility
  since 2023. A model trained on all eras will average that away, which is one more
  reason the training start year is a tested setting.
- **Post-earnings drift has faded** since 2020, as the literature says it has.
- **The upside target is again mostly volatility** (Parkinson IC 0.16–0.20 in every era).

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
once their window rolls past, so copy them from a teammate rather than refetching.

```bash
cd icaif2026/starter-kit
cp .env.example .env && chmod 600 .env   # CODABENCH_TOKEN, ICAIF_PROFILE=profiles/profile99-production.json
python3 tools/evaluate.py examples/evaluation.json
```
