# System guide

How the ICAIF 2026 trading system is built, what it runs on, and how to get it running
on a new machine. Written 2026-10-02. For every measured number and data trap, see
`README.md`. The plan is the Roadmap tab of the design doc (linked in `CLAUDE.md`), and
`git log` says why each change was made.

## 1. What the system is

Our entry in the ACM ICAIF 2026 Trading Agent Competition (Codabench 99):
- **Market:** long-only target weights for 30 US large caps, starting from $1M.
- **Trading:** 7 rounds a day, a 0.1% fee on traded value, at most 30% per name, no
  leverage.
- **Ranking:** the mean of four ranks: return, Sharpe, max drawdown and turnover.
- **Dates:** Validation runs Oct 8–9 and Official Oct 12–30. Registration must happen
  before Oct 8 00:00 ET to enter Validation.

It is one system with two desks. A gate decides whose book is submitted:

```
INPUTS                       DESKS                                    OUTPUT
prices + book ──┐    ┌─ Rule desk (submitted until the gate)
HAR volatility ─┤    │    risk parity, exposure set by a regime model,
model scores ───┼───►│    bought at the phase's first round, then held
news, 8-K, macro┤    │
journal ────────┘    └─ LLM desk (shadow until the gate)           ─► gate ─► code checks ─► Codabench
                          Strategist · Risk review · Event analyst          caps, 1e-6 grid,
                          (falls back to the rule on any failure)          deadline, hold guard
```

- **The rule desk's book is what gets submitted.** It trades once a phase, at round 1
  of the first day, and holds after that. Backtests found no rule that trades after
  entry and beats this.
- **The LLM desk decides on its own paper book beside it,** and every call is logged.
  It may trade only after it beats the rule in replays and in live shadowing (Roadmap
  step 6).
- **Code does everything that must never be wrong:** weights, caps, deadlines and
  fallbacks. The LLM only chooses levers.

## 2. How it runs

It is one always-on machine making only outbound connections. Nothing is hosted for
the organizers to call.

| Process | What it does |
|---|---|
| `tools/live_runner.py run --phase <phase> --live` | The scheduler. It keeps the competition server's clock, wakes 12 minutes before each deadline, and re-reads the schedule every 10 minutes |
| Round worker (one process per round) | Fetches data, runs both desks, guards and uploads. The watchdog kills it at the deadline |
| Scorer (child process) | Scores the daily model once a day, after the upload (killed at 4 minutes) |
| Daily data jobs | News before each round; earnings calendar and 8-Ks daily; push to S3 (§7.6) |

**One round, in order:**
1. Fetch Yahoo daily closes for the 30 names (1,200 calendar days; the rule needs at
   least 751 sessions) and the last few days of 30-minute bars.
2. Read the book from the competition API, or the paper book in a dry run.
3. The rule desk decides: an entry at the phase's first round 1, otherwise a hold.
4. Guard: a trade is written as `decision.json` and a hold as `hold.json`. The kit
   uploads only `decision.json`.
5. Upload through the organizers' starter kit. This happens only with `--live` and the
   owner's arm file.
6. Shadow: model scores and HAR (once a day each), then the LLM desk on its paper book.
7. Commit `state.json` in one atomic write: journals, flags and paper books.
   Restarting resumes from it.

**A day (October; IST = ET + 9:30):**

| ET | IST | What happens |
|---|---|---|
| 08:00 | 17:30 | News snapshot |
| 08:58 | 18:28 | Round 1 wakes (deadline 09:10; executes at the 09:30 open) |
| 10:13 … 15:13 | 19:43 … 00:43 | Rounds 2–7 wake (deadlines 10:25 … 15:25; execute at 10:30 … 15:30) |
| after 16:00 | after 01:30 | Daily earnings and 8-K snapshots; push to S3 |

## 3. How it is implemented

| Area | Files | What it does |
|---|---|---|
| Live runner | `icaif/runner.py`, `tools/live_runner.py`, `icaif/watchdog.py` | Scheduler, rounds, hold guard, arming, upload, killing hung workers |
| Live data | `icaif/live.py`, `icaif/public_bars.py`, `icaif/external.py`, `icaif/daily_features.py` | Yahoo daily and 30-minute bars, freshness checks, live model scoring, HAR inputs |
| Competition API | `starter-kit/` (the organizers' kit, never edited), `icaif/kit.py`, `icaif/portfolio.py` | Schedule, portfolio, decision upload; strict portfolio parsing |
| Rule desk | `icaif/quant.py`, `icaif/quant_strategies.py` | Risk parity (shrunk 60-session covariance) and the two-state HMM that sets exposure |
| Agent desk | `icaif/agents/desk.py` | The three roles, fallback to the rule, timeouts, state between rounds |
| Brains | `icaif/agents/brains.py` | `RuleBrain`, `ClaudeBrain` (Opus 5), `GemmaBrain` (Gemma 3 on Bedrock), `CachedBrain` |
| Prompts and answers | `icaif/agents/prompts.py`, `icaif/agents/schemas.py` | One system prompt per role; the JSON each role may answer |
| Observation | `icaif/agents/observe.py`, `icaif/agents/signals.py` | What a role sees, point in time; anonymised in replays |
| Memory | `icaif/agents/journal.py` | One journal per desk, checked against the book every round |
| Signals | `icaif/vol.py`; `icaif/train.py` with `output/ag/` | HAR volatility forecasts; the daily AutoGluon ensemble |
| Event feeds | `icaif/news.py`, `icaif/filings.py`, `icaif/earnings.py`, `icaif/earnings_calendar.py`, `icaif/macro.py` | Headlines, 8-Ks, earnings dates, macro series and the Fed calendar |
| Evaluation | `tools/agent_replay.py`, `tools/holdout_eval.py`, `icaif/sim.py`, `icaif/windows.py` | LLM-vs-rule replays (the step-6 gate), holdout scoring, the simulator |

## 4. The agents

| Role | Runs | Decides |
|---|---|---|
| Strategist | Once, at round 1 of day 1 | Book shape (risk parity or inverse-vol); exposure 0.30–0.95; names to leave out, each with a cause; a Black–Litterman views level |
| Risk review | Round 1 of each later day | Hold (the default), set exposure, exit names, or rebalance (at most twice a window, with a reason) |
| Event analyst | Only when a trigger fires on a held name: a move of 3 or more daily sigmas (rounds 3–7), or earnings before the next open | Hold or exit, per name |

- **Inputs.** Every call gets one point-in-time observation:
  - clock, book, market regime, macro and the Fed calendar
  - per name: returns, volatility, HAR forecast, model rank, earnings timing, recent
    8-Ks, and headlines (live only)
  - the rule's proposal, any role-specific extras, and the memory below

  Live, with headlines for all 30 names, an entry observation runs to about 27,000
  tokens.
- **Memory.** Every call is fresh; there is no chat history. `journal.py` keeps one
  journal per desk, which all three roles read:
  - it records decisions, levers, short reasons, fills and P&L since each decision
  - it is checked against the book every round
  - it is capped at 6,000 characters
  - it is saved in `state.json` after every round
- **Tools.** None. A role acts only through the fields of its answer schema, and code
  carries them out.
- **Brains.** Set with `--shadow rule | gemma | claude | none`. The default is `claude`.
  Without `ANTHROPIC_API_KEY` it falls back to the rule with a warning, so use
  `--shadow gemma` until the Opus key arrives.
  - **Gemma 3 on Bedrock** (`brains.GemmaBrain`):
    - **Models and region:** the 27B by default, or `--gemma-model` for 12B/4B, in
      ap-south-1.
    - **Answer checks:** the role's JSON Schema is in the prompt; the reply is parsed
      and validated, gets one repair, and otherwise falls back to the rule.
    - **Reasoning and records:** an `analysis` field comes before the decision; raw
      replies go to `<round>/shadow_replies.json`.
    - **Settings and cost:** temperature 0, no SDK retries, about $0.0075 per live entry
      call.
  - **Claude Opus 5** (`brains.ClaudeBrain`) is wired for when the API key arrives.
  - **Failures:** any failure, timeout or invalid answer becomes the rule's answer, and
    the log says so.
- **The gate.** `tools/agent_replay.py --brain gemma` replays anonymised past windows and
  ranks the LLM desk against the rule desk. Pass `--end 2025-12-31` to keep the
  Jan–Jun 2026 holdout unseen.

## 5. What it uses

| Input (Roadmap) | Source | Code | Refreshed |
|---|---|---|---|
| Live prices and book | Yahoo daily and 30-minute bars; portfolio from the competition API | `live.py`, `portfolio.py` | Every round |
| HAR volatility | Computed from the `data/public/` bar archive plus 60 days of fresh Yahoo bars | `vol.py`, `live.vol_forecasts` | Once a day |
| Daily ensemble scores | `output/ag/daily/d5_pct/2026/`, fed Yahoo daily bars (~546 symbols plus 16 context series), S&P membership and EDGAR earnings | `live.py`, `daily_features.py` | Once a day |
| News, filings, macro | Yahoo RSS headlines; SEC EDGAR 8-Ks; Yahoo macro series; the Fed's FOMC calendar | `news.py`, `filings.py`, `macro.py` | Headlines before each round; 8-Ks daily; macro at the prior close |
| Portfolio journal | Our own records, checked against the competition API | `agents/journal.py` | Every round |

**Models.**
- **Daily ensemble:** the only trained model loaded live. AutoGluon, 37 features,
  trained on data through 2025.
- **HAR:** code, refitted each quarter from the bar archive.
- **Rule desk:** fitted live from Yahoo closes; nothing is stored.

## 6. Requirements

**Software**
- Python 3.13 (the arm64 build on Apple Silicon), with the packages pinned in
  `requirements.txt`: AutoGluon 1.5, pandas 2.3.3, numpy 2.3.5, pyarrow 20.0.0, boto3,
  anthropic, yfinance, httpx and truststore.
- `uv` (recommended), the AWS CLI v2, and git.

**Accounts, credentials and environment.** Never commit any of these; `.gitignore`
already excludes `.env`, `.icaif/`, `data/`, `output/` and `.venv/`.

| Need | For | How |
|---|---|---|
| Codabench account and team registration | Uploading decisions | Register before Oct 8 00:00 ET. The one-time team token lands in `starter-kit/.icaif/credentials.json`. It is never reset, so back it up |
| `CODABENCH_TOKEN`, `ICAIF_PROFILE` | The kit's client | `starter-kit/.env` (copy `.env.example`, then `chmod 600`) |
| AWS credentials | S3 data (read; write only under `icaif2026/`) and Bedrock (`bedrock:InvokeModel` on Gemma 3 in ap-south-1) | `AWS_PROFILE=<profile>`. An unattended host needs a long-lived, narrowly scoped credential, not an SSO login that expires |
| `SEC_USER_AGENT="<name> <email>"` | SEC EDGAR (8-Ks, earnings) | Environment. The SEC requires a real contact |
| `ANTHROPIC_API_KEY` | The Claude brain (later) | Environment |
| `ICAIF_BEDROCK_REGION` (optional) | Another Bedrock region | Environment (default `ap-south-1`) |

**External sources.** All of these must be named in `disclosures.md` for the final
materials.

| Source | Used for | Access | Notes |
|---|---|---|---|
| Yahoo Finance (yfinance) | Daily and 30-minute prices, macro series, earnings calendar | No key | The organizers' suggested source; a single point of failure |
| Yahoo Finance RSS | Headlines | No key | Archived as it arrives; past snapshots can't be fetched again |
| SEC EDGAR | 8-K filings, earnings release times | `SEC_USER_AGENT` | |
| Federal Reserve FOMC calendar | Fed decision dates | No key | Parsed once (2021–27) |
| fja05680/sp500 (GitHub, MIT) | S&P 500 membership history | No key | Snapshot kept in S3 |
| Alpaca | Historical 30-minute bars (2016 on) behind HAR and the training labels | Free sign-up; keys in `.env` only to refetch | Not used live. Confirm with the organizers that it counts as "publicly available" |
| Codabench competition API | Schedule, portfolio, upload | The tokens above | |
| Amazon Bedrock (Gemma 3) | LLM desk | AWS credentials | |
| Anthropic API | LLM desk (later) | API key | |

## 7. Getting it running

### 7.1 Clone

```bash
git clone https://github.com/MO-ShaanilP-39793/icaif2026.git && cd icaif2026
git switch gemma-brain-and-system-guide      # until this branch is merged
```

If the clone sits inside an alphaBT checkout, hide it from alphaBT first with
`echo 'icaif2026/' >> ../.git/info/exclude` (run from inside the clone). Otherwise a
single `git add -A` in alphaBT stages this whole repo.

### 7.2 Python environment

```bash
uv python install cpython-3.13-macos-aarch64-none   # Apple Silicon; elsewhere: uv python install 3.13
uv venv --python 3.13 --seed .venv
.venv/bin/pip install -r requirements.txt           # about a minute with uv, longer with pip
```

On Apple Silicon, ask `uv` for the arm64 build explicitly. An Intel build of `uv` would
install an x86_64 Python, and PyTorch ships no macOS x86_64 wheels.

Behind a TLS-intercepting proxy, use
`uv pip install --system-certs --python .venv/bin/python -r requirements.txt`.

### 7.3 Data, from S3 (never committed)

`data/` and `output/` are git-ignored. The copy of record is `s3://shaanil/icaif2026/`,
under the repo's own paths. The bucket also holds alphaBT production prefixes, so read
and write only under `icaif2026/`, and never sync with `--delete`.

```bash
B=s3://shaanil/icaif2026; P="--profile <your-profile>"

# The live runner (~340 MB)
aws s3 sync $B/output/ag/daily/d5_pct/2026 output/ag/daily/d5_pct/2026 $P      # the daily model
aws s3 sync $B/data/public data/public $P --exclude "*" \
    --include "alpaca_30m_2*.parquet" --include "yahoo_60m_*.parquet" --include "yahoo_30m_*.parquet"
aws s3 sync $B/data/external data/external $P --exclude "yahoo_daily_*"         # snapshots + news archive

# The daily earnings-calendar job also needs the universe snapshot (106 MB)
aws s3 sync $B/data/external data/external $P --exclude "*" --include "yahoo_daily_universe_*"

# Replays (the step-6 gate) also need the model's past scores and the macro history
aws s3 cp $B/output/preds/daily_d5_pct.parquet output/preds/ $P
aws s3 sync $B/data/external data/external $P --exclude "*" --include "yahoo_daily_context_*"

# Everything, for research and retraining (~4 GB)
for d in data output reports; do aws s3 sync $B/$d $d $P; done
```

The organizer's panel (`data/hourly_market_data_2021_2026.parquet`) is licensed to
registered participants. Keep it private. It is only needed for research and for 8
tests, which are skipped without it.

### 7.4 Secrets and environment

```bash
cp starter-kit/.env.example starter-kit/.env && chmod 600 starter-kit/.env
#   CODABENCH_TOKEN=...   ICAIF_PROFILE=profiles/profile99-production.json
export AWS_PROFILE=<profile>  SEC_USER_AGENT="<name> <email>"     # ANTHROPIC_API_KEY later
```

### 7.5 Check that it works

```bash
.venv/bin/python -m pytest -q          # about 80 s: 351 pass, 8 skip without the organizer panel

# A past day's 7 rounds, back to back, dry (nothing uploads); about a minute
.venv/bin/python tools/live_runner.py rehearse --date 2026-09-30 --fast --shadow rule
.venv/bin/python tools/live_runner.py rehearse --date 2026-09-30 --fast --shadow gemma \
    --out output/live/rehearsal-gemma
.venv/bin/python tools/live_runner.py journal --phase rehearsal-2026-09-30-fast \
    --out output/live/rehearsal-gemma                          # what each desk's roles were shown
```

What to expect:
- Round 1 trades (the rule's entry, as a dry-run upload), and rounds 2–7 hold.
- With `--shadow gemma`, the shadow's entry shows `source: brain` and the phase records
  its spend.
- Rerunning into the same phase folder skips rounds already done, so use a fresh
  `--out`.

### 7.6 Keep the inputs fresh (daily)

```bash
.venv/bin/python tools/news_archive.py           # before every round's deadline (09:05, 10:20 … 15:20 ET) and at 08:00 ET
.venv/bin/python tools/earnings_calendar.py      # every day before 09:10 ET
.venv/bin/python tools/filings_events.py         # daily, with SEC_USER_AGENT set
aws s3 sync data/external s3://shaanil/icaif2026/data/external $P    # then push; never --delete
aws s3 sync output/live   s3://shaanil/icaif2026/output/live   $P
```

- **A missed snapshot is gone.** Yahoo later shows what happened, not what was
  announced or on the wire at the time.
- **Scheduling:** `tools/install_news_launchd.py` schedules the news job on macOS. On
  Linux, use cron or systemd timers at the same US Eastern times.
- **The machine must stay awake** from 08:45 to 15:30 ET (18:15–01:00 IST) on every
  trading day of a phase.

### 7.7 Going live (Validation, Oct 8–9)

1. Register the team (owner). The credentials land in `starter-kit/.icaif/`.
2. Run `.venv/bin/python tools/live_runner.py portfolio --phase validation`. It checks
   that the server's book parses; fix `portfolio.parse` if the format differs.
3. Rehearse a full day at real times on the machine that will run the phase:
   `.venv/bin/python tools/live_runner.py rehearse --shadow gemma`.
4. Arm: the owner runs `.venv/bin/python tools/live_runner.py arm --phase validation`
   at a terminal. Nothing ever uploads without this.
5. Run `.venv/bin/python tools/live_runner.py run --phase validation --live --shadow gemma`.
6. Watch it with `status --phase validation` and `journal --phase validation`. Round
   records are kept under `output/live/validation/`.

## 8. Status on 2026-10-02

**Working, verified on a fresh machine:**
- data and model pulled from S3
- the environment builds from `requirements.txt`
- the full test suite passes
- dry rehearsals of 2026-09-30 pass with the rule shadow and with the Gemma shadow
- a real Gemma 3 27B entry call answered validly on the first attempt

**Open before Validation:**
- register the team and check the portfolio format
- choose a host that stays awake, with its credentials
- schedule the daily data jobs; the snapshots in S3 end Sep 28–Oct 1
- rehearse a full day in real time
- add alerts for missed rounds

**Open before an LLM may trade:**
- **Step 5:** news for the Event analyst, and the trim (profit-booking) decision with a
  rule version as its bar.
- **Step 6:** run the gate replay with Gemma.
- **Model earnings feature:** wire the Yahoo earnings calendar into the model's
  "sessions to next earnings" feature. It is empty live, and that costs about half the
  model's signal among the 30.
- **Per-role views:** each role currently reads the full ~27K-token observation.
- **The rule:** decide whether it stays risk parity at the regime exposure or becomes
  the 75% inverse-vol hold.

## 9. Ground rules (from the Roadmap)

- **Hold is the default.** Every trade after day 1 must expect to earn more than about
  20 bps round trip.
- **The rule desk is the bar.** An LLM role trades only after it beats the rule in
  replays and has shadowed live rounds.
- **News can't be backtested honestly.** Replays hide names, dates and headlines, so the
  evidence on news comes from live shadowing.
- **Code does what must never be wrong.** The LLM only chooses.
- **No look-ahead.** Every new input gets a test that rewrites the future and checks
  that the past is unchanged.
- **Final materials come from this repo alone,** with every model and data source named
  in `disclosures.md`.
