# ICAIF 2026 Trading Agent Competition

Entry for [Codabench competition 99](https://hackathon2.deepintomlf.ai/competitions/99/):
hourly long-only target weights for 30 US large caps, $1M, 0.1% cost on traded
notional, ≤30% per name, ≤100% gross. Lives on branch `icaif2026` beside alphaBT
for context only — it never merges into `pipeline/dev` and imports nothing from `src/`.

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
- `adj_close == close` everywhere, and prices are **split-adjusted** (no jump at
  NVDA 2024-06-10, AMZN 2022-06-06, TSLA 2022-08-25, GOOGL 2022-07-18). Live prices
  will be raw — irrelevant unless a split lands during the competition.
- 14:30 / 15:30 bars appear only on early-close days; ~150 ticker-days have fewer
  than 7 bars. A panel pivot will carry NaNs there.
- The kit ships a **metrics calculator, not a backtester**. `kit/evaluation.py` is
  the official formula (self-check passes); the simulator that produces its inputs
  is ours to write.

## Credentials

Registration returns `TEAM_ID` and a **one-time team token that is never reset**.
It lands in `starter-kit/.icaif/credentials.json` (0600). Everything that can hold
it is gitignored here. Never push this branch to `hf`/`cchf` — both push a public
HuggingFace mirror. Final reproducibility materials are built from `icaif2026/`
alone, never the repo.

## Setup

Base `python3` (3.13, has `httpx`/`pandas`/`pyarrow`) is sufficient. Data:
`data/hourly_market_data_2021_2026.parquet` (copied from the Codabench Files tab;
not committed).

```bash
cd icaif2026/starter-kit
cp .env.example .env && chmod 600 .env   # CODABENCH_TOKEN, ICAIF_PROFILE=profiles/profile99-production.json
python3 tools/evaluate.py examples/evaluation.json
```
