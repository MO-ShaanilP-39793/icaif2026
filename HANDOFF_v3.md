# Desk v3 handoff (2026-10-11)

For the sessions that continue v3's development and its deployment on the GCP VM, from
another machine and account. Read this, then `CLAUDE.md`, then the README sections
"Live runner", "Gemini 3 since 2026-10-11" and "Desk v3". `git log` holds the reasoning
behind every change; the commit messages are long on purpose.

Nothing in this file is a secret, and no secret belongs in it.

**Since this was written:** see `JOURNAL_2026-10-11.md` (the VM moved to Vertex AI, and
alerts went live) and **`JOURNAL_2026-10-12.md`: Official runs at 99.9% cash
(`--overlay 0.001`, not 0.1), which needs the guard fix on main pulled first.**

## Where things stand

- **Official (Oct 12-30, 2026) enters with desk v3, stage 1.** Round 1 on Oct 12
  (deadline 09:10 ET, 18:40 IST) is the decision that matters: the book is bought once and
  held. Code is on `main` (head `7235c13` at the time of writing) and pushed.
- **Stage 1 is built, tuned and rehearsed.** It is what goes live.
- **Stage 2 (the escalation chain) is designed, not built.** That's the main development
  work left.
- **Arming is the owner's, always.** No session arms Official or uploads on its own.

## What v3 is

Three roles in the design; stage 1 uses the first.

- **Portfolio manager (PM), day 1, round 1.**
  1. Four analysts (market, earnings, news, quant) each read their own slice of the
     observation and write a report.
  2. The PM reads only those reports plus a minimal market table, and writes a book, a
     thesis and **plan conditions**: when it would act, in a fixed vocabulary code can
     evaluate (`schemas.Condition`).
  3. Code's self-check (`agents/selfcheck.py`) shows the draft's concentration, expected
     vol, beta, recent drawdown and the reference books at the same gross.
  4. The PM confirms or revises once.
  5. Code scales the book to the **overlay**, the invested share code holds. The PM is
     never told.
- **Analyst, rounds 2-7 (stage 2):** a note every round; escalates only by citing a
  checkable fact; cannot trade.
- **Senior associate, each morning (stage 2):** a progress note against the plan; asks the
  PM to intervene only when necessary; cannot trade.
- Fired conditions bind: one goes to the PM with its stated action as the proposal, and the
  PM confirms, or declines and rewrites it. The budget starts at 2 PM interventions and
  1 book of turnover after entry.

The full design is `v3_desk_plan.md`.

## The configuration Official runs, and why

| Setting | Value | Evidence (`stage1_doe.md`) |
| --- | --- | --- |
| Architecture | analysts' reports only | analysts beat a single PM by 0.23-0.24, about 4-5x run-to-run noise |
| Data streams | all but 8-K filings | filings were the only stream worse on return, Sharpe and drawdown |
| Prompt | the four-rank contest prompt | a "maximise Sharpe" prompt picked worse stocks (Sharpe 1.14 against 1.83 pooled) |
| Cash | overlay 0.10 (90% cash), set by code, PM never told | the overlay beat the PM's own cash choice at every level; 10% is the owner's call |
| Model | Gemini 3.1 Pro in every role (analysts at thinking level medium, PM at high) | level with Gemini 2.5; clean on the memory probe |

**Hold-out (9 windows from 2026, never used to choose):**
- v3's books had Sharpe 1.66 against the 75% inverse-vol hold's 0.08, higher in 6 of 9.
- v3 scored 2.64 against the hold's 2.81, within noise (SE 0.24).
- Cash scored 2.17 on the modelled field, a caveat about our guess at the rivals: a
  pure-cash entrant takes rank 1 on drawdown and turnover in every window.

Why 90% cash: Sharpe has a zero risk-free rate, so scaling a book leaves it unchanged
while drawdown and turnover shrink. Below about 25% invested, the modelled field's
drawdown and turnover ranks stop improving, because its cash entrant already holds
rank 1. Whether 10% beats 25-50% turns on whether the real field holds a pure-cash team.
The selection windows favoured 50%; the hold-out favoured 75%. Both differences are small
against the noise.

## Code map

| Path | What |
| --- | --- |
| `icaif/agents/v3.py` | `V3Desk`, `V3Config` (analysts, gross, streams, evidence, self_check, objective, overlay), `strip`, `STAGE1` windows, `STREAMS16`, `BoughtBook`, `capped_book`, `HoldBrain` |
| `icaif/agents/prompts_v3.py` | frozen system prompts: four-rank and Sharpe variants, with and without evidence |
| `icaif/agents/selfcheck.py` | the PM's self-check report (no trailing return, on purpose) |
| `icaif/agents/schemas.py` | `PMEntry`, `Condition`, `PMCheck` (the desk v3 section) |
| `icaif/agents/v2.py` | v3 subclasses `V2Desk` for its parallel calls, slots and chain log |
| `icaif/agents/brains.py` | Gemini 3 models, prices, thinking levels; `CachedBrain(repeat=N)` |
| `icaif/runner.py` | live runner; `Config.desk/quick_model/overlay`, `v3_config`, `overlaid` |
| `tools/live_runner.py` | CLI: `--desk v3 --quick-model --overlay` |
| `tools/entry_replay.py` | stage-1 arms on the selection or hold-out windows; `round0`, `ledgers`, `arm` |
| `tools/doe_report.py` | the decision rules: `arch`, `streams` (raw rule), `cash`, `noise`, `overlay`, `table` |
| `tools/memory_probe.py` | does a model remember a window? Run it before replaying a new model |
| `tests/test_v3.py`, `tests/test_doe_report.py`, `tests/test_runner.py` | the guarantees, each named for the silent failure it prevents |

## Running experiments

```bash
.venv/bin/python tools/entry_replay.py ledgers        # code-answered v3 = the hold, all 22 windows (free)
.venv/bin/python tools/entry_replay.py arm --analysts reports_only --drop filings --overlay 0.10 \
  --quick-model gemini-3.1-pro-preview --quick-effort medium \
  --deep-model gemini-3.1-pro-preview --deep-effort high          # prints the cost; add --yes
.venv/bin/python tools/doe_report.py overlay <tag> ...            # score any arm's books at fixed cash splits
```

- **Windows.** `v3.STAGE1["select"]` (13 windows, Feb 2025 - Jan 2026) and `["confirm"]`
  (9 hold-out windows, 2026). They step around the four official4 windows. A hold-out run
  needs `--split confirm --confirm-choice <commit>`.
- **News.** Every real-names window needs its headlines and 8-K texts fetched first
  (`tools/replay_sources.py --on <day>`). All 22 stage-1 windows are fetched and on S3.
- **Cost and time.** About $0.15-0.20 a window with 3.1 Pro, and about 1 minute a window.
  An arm of 13 windows takes about 15-20 minutes. Arms are independent processes, so run
  several in parallel (separate processes, never a process pool) under `caffeinate` on a
  Mac. A Gemini 503 or a sleep timeout falls back to the hold for that window: rerun the
  arm, and only the failed calls are asked again (failures are never cached).
- **Noise.** A setup's 13-window mean score moves 0.05-0.13 from run to run. A gap smaller
  than that is not a finding. Use `--repeat N` to measure it.
- **Memory.** Gemini 3.8 Flash remembers February 2025, so it must not answer in replays
  on the selection windows. Live use is fine, since the future is unknown to every model.
  Probe any new model first.

## Deployment (the GCP VM)

- **The VM.** `icaif-live`: GCP project `icaif2026` on the owner's personal account (a free
  trial; never upgrade billing), zone us-east4-a, e2-standard-4 (4 vCPU, 16 GB), Ubuntu
  24.04. SSH key only, only tcp:22 open. The owner reaches it from their personal laptop
  (GCP is blocked on the office Mac).
- **The session there.** A Claude session runs on the VM in tmux session `live` and
  operates the runner. Repo at `~/icaif2026` through a **read-only** deploy key: it pulls
  (`git pull --ff-only`), tests and runs, and never commits. Code changes are made
  elsewhere and pushed to `main`.
- **Data and model.** The private HF dataset `MO-AI-Inv/icaif2026-vm` (`data/external`,
  `data/public`, `data/derived`, `output/ag/daily/d5_pct/2026`; about 520 MB). It excludes
  the organizer panel (8 tests skip on the VM) and every secret. The snapshot is from
  Oct 9.
- **Secrets.** These reached the VM by an encrypted tarball and are chmod 600: `.env`
  (Gemini, Alpaca), `starter-kit/.env` (Codabench token, profile) and
  `starter-kit/.icaif/` (the team token, issued once and never reset).
  `SEC_USER_AGENT` is exported in `~/.profile`; confirm it is in the environment of the
  shell that runs `run`. The HF read token used for the data download appeared in the VM
  session's transcript; the owner deleted it on HF (confirmed Oct 11).
- **CPU.** The e2 landed on a 2014 Broadwell, 4-8x slower than the Mac. In `--submit agent`
  mode, scoring and inputs run **before** v3 decides, so they sit on the entry's critical
  path. Measured in a dry run: scoring about 44 s, inputs about 71 s. v3's own chain was
  46 s on the Mac (mostly network-bound LLM calls). That fits the 12-minute lead, but
  measure it on the VM (the check below). Moving to n2d-standard-4 is the owner's call.
- **The live command (the owner's, after arming):**

```bash
cd ~/icaif2026 && git pull --ff-only
.venv/bin/python tools/live_runner.py portfolio --phase official
.venv/bin/python tools/live_runner.py arm --phase official --submit agent     # the owner, at a terminal
.venv/bin/python tools/live_runner.py run --phase official --live --submit agent \
  --desk v3 --model gemini-3.1-pro-preview --quick-model gemini-3.1-pro-preview --overlay 0.1
```

- **What the runner does.**
  - Round 1 from cash: analysts, then the PM's draft, the check, the overlay to 10%, the
    guard, and the upload (only if armed).
  - Every later round holds without a call.
  - A failed PM buys inverse-vol at 10%. A crashed v3 desk hands the entry to the rule,
    scaled to 10%. A missed round holds the book.
  - Spend is capped at $10 a phase (`shadow_cost_cap`); an entry costs about $0.20.
- **The dry rehearsal to repeat on the VM, and what the Mac got (Oct 11):**

```bash
.venv/bin/python tools/live_runner.py rehearse --date 2026-10-09 --fast --submit agent --desk v3 \
  --model gemini-3.1-pro-preview --quick-model gemini-3.1-pro-preview --overlay 0.1 --shadow claude \
  --out output/live/vm_rehearse_v3_2026-10-09
```

  On the Mac: every input "ok", the v3 chain 46 s, round 1 ready 516 s before its deadline,
  the written gross 0.10, rounds 2-7 hold, $0.16. A one-day rehearsal tells the PM its
  window is 1 session; Official's schedule has 15 days, so the PM sees 15.
- **One machine armed, ever.** The office Mac is a cold standby, never armed.

## Open work, in priority order

1. **Before Oct 12, on the VM:**
   - pull `7235c13`;
   - the test suite;
   - Gemini 3.1 Pro reachable with the VM's key;
   - the dry rehearsal above;
   - `portfolio --phase official` parses;
   - `status --phase official --submit agent` reports not armed.

   If round 1's `ready_before_deadline_s` is under about 300 s, precompute (item 2) or move
   the VM to n2d.
2. **Deployment hardening:**
   - systemd units for the runner;
   - a morning precompute of scores, HAR and macro, so only news and LLM calls run
     against round 1's clock;
   - the news archiver (the Mac ran it by launchd);
   - a backup of `output/live/official/state.json` to a GCS bucket;
   - an alert channel (ntfy.sh was suggested; undecided).
3. **Live headlines parity.** Replays read Alpaca (Benzinga) news; live reads the Yahoo
   archive. Consider fetching Alpaca live as well.
4. **Stage 2, the escalation chain** (`v3_desk_plan.md`, build steps 7-9):
   - the condition evaluator (point in time, from the journal's entry prices);
   - the analyst each round 2-7 and the associate each morning;
   - escalation routing; the budget (2 interventions, 1 book after entry);
   - settlements (v2's `_settle` is most of it); plan rewrites.

   Its acceptance test is the same as stage 1's: a code-answered desk equals the hold in
   all 22 windows. Evaluate it against the same entry, cached, so the chain's value is
   isolated. Sweep the budget (caps 0, 1, 2, unlimited). In a calm window a working chain
   does nothing.
5. **The spend cap.** Resize the $10 a phase once stage 2's cost per round is known.
6. **Cheaper analysts live.** Gemini 3.8 Flash costs about a third of 3.1 Pro, but can only
   be evaluated on windows after its likely cutoff (March 2026): 5 hold-out windows, too
   few on their own.
7. **Records.** The final materials are due Nov 3. The journal under
   `output/live/official/` and the stage-1 evidence in `stage1_doe.md` are the record.

## Traps learned the hard way

- **S3** (`s3://shaanil/icaif2026/`, the copy of record):
  - write only under `icaif2026/`, never `--delete`;
  - **always `--exclude "*decision.json" --exclude "*portfolio_raw*"`**, since live
    folders hold the team token;
  - dry-run and grep the list before the real sync. Two decision files leaked on Oct 1 and
    were deleted on Oct 11.
- **A worktree's `output/` is a real folder.** Copy the results into the main checkout
  before removing the worktree, or they are gone.
- **`main` is shared by several sessions.** Check `git status` and `git log origin/main`
  before committing, commit by path, and rebase before fast-forwarding.
- **Caches and variants.** Cached answers are keyed by model, role, prompt and payload. A
  prompt change re-asks everything; a repeat run needs `--repeat N`. Runs on other models
  get a name suffix so they never share a directory.
- **Gemini schemas.** Gemini refuses schemas with capped text in long lists, so caps travel
  as words (`gemini_schema`). Gemini 3 takes a thinking *level*, not a budget.
- **Write tool.** It turns backslash-u escapes into real characters. Write such sources
  through a Python script.
- **Never forward-fill a price, and never read past a deadline.** Each input has a test
  that rewrites the future and requires the past unchanged. Add one for anything new.
- **Questions first.** Ask the owner before any run measured in minutes or any spend.
  Don't commit unless asked; "push" means `git push origin main` (GitHub).
