# Desk v3: a portfolio manager, a senior associate and an analyst

Status: vision agreed 2026-10-07. Stage 1's code is built (branch `v3`: `icaif/agents/v3.py`,
`selfcheck.py`, `prompts_v3.py`, `tools/entry_replay.py`, `tests/test_v3.py`) and round 0
has run; the LLM rounds wait on the news fetch and the owner's go. This is the plan to
build towards.
Settled so far: fired conditions bind, and the budget's starting values. The open
decisions at the end are still the owner's.

## Why a third desk

| | What it did on the four Earnings season windows | Why |
| --- | --- | --- |
| v1 free desk | Beat the hold's return once in four; turnover 1.9-5.4% against the hold's 0.71%; 25th of 29 on 2026-04-13 | One role was asked "what book?" every morning and rewrote the book on 13-14 of 15 mornings |
| v1 levered desk (Grok, Jan 2026) | Held through all 86 questions | Told that holding wins, it became the rule |
| v2 firm | 24th-29th of 28-29 on 2026-04-13, the only window it has run; $3-7 a window | Nine roles, a debate every morning, and a trade list each day |

The score pays for one well-chosen book that is left alone: the 75% inverse-vol hold has
been the hardest entrant to beat, and every rule we tested that traded after entry lost.
v3 spends its thinking on the entry, and makes every later trade climb a chain to reach
the one role that can trade.

On 2026-04-13 the rule spent 0.08% more turnover than the hold and fell 4 turnover ranks,
a full point of overall score. The field crowds just above a pure hold, so the first
intervention after entry is the expensive one.

## The desk

```
 entry (day 1, round 1)        PM: every input, a free book, a thesis, plan conditions
                                   draft -> code's self-check -> revise once -> commit
 rounds 2-7, every round       Analyst: a note to the journal; escalates only on a cited fact
                                   -> senior associate (can wait until morning)
                                   -> PM (urgent: trades now)
 every morning, round 1        Senior associate: progress note against the plan;
                                   asks the PM to intervene only when necessary
 on escalation                 PM: may trade anything, within the budget
 code, always                  caps, fills, conditions, settlements, the journal; a failed call holds
```

### Portfolio manager: the entry

The entry's shape is what stage 1 calibrates: one PM reading everything, or analysts'
reports synthesised for the PM; a free gross, or a fixed equity sleeve with the rest in
cash; and which inputs are worth their characters. What follows is the full version,
which every stage-1 arm is a subset of.

- **Entry analysts (optional, a stage-1 factor).** Four quick-tier analysts report in
  parallel before the PM decides: market and macro, earnings and events, news and
  filings, quant (HAR vol, model ranks, the universe ranking, correlation). They are v2's
  four (`v2.ANALYSTS`, `prompts_v2`), asked about an entry rather than about changes to a
  book. Each reads only its own stream, so a report can't smuggle in another stream's
  conclusion.
- **Reads** everything the free desk reads today (clock, book, market and regime, each
  name's returns, vols, HAR forecast, model rank, OU score, earnings timing, 8-Ks,
  headlines, new filings with text, macro, memory) plus the universe ranking, which the
  free desk was not shown.
- **Writes** a free book (any weights, at most 30% a name and 100% gross: the contest's
  own limits, checked by code and refused rather than rescaled), a thesis, and plan
  conditions. In a sleeve arm, code fixes the gross and the PM chooses the composition
  within it. The PM's weights then sum to 1 inside the sleeve, and code scales them.
  Otherwise a PM asked for a 30% sleeve that writes 0.75 would be refused, or worse,
  silently rescaled into a book it didn't choose.
- **Self-check.** The draft goes to code, which returns a report: names held, largest
  weight, effective number of names (1/sum w²), expected book vol from the HAR forecast
  and the shrunk covariance, beta to the basket, the weight reporting inside the window,
  average correlation, the entry's turnover and fee, and the same numbers for the
  inverse-vol and risk-parity books beside it. The PM sees the report and either
  confirms or revises once. A second draft gets a report too but no further revision.
  This is two calls, not a tool loop: `GeminiBrain` asks with no tools.
- **Plan conditions** are what make the later roles cheap and consistent. With every
  input in front of it, the PM states when it would act, so the associate and the analyst
  check facts against the plan instead of re-arguing the entry with less information.
  Conditions use a small vocabulary that code can evaluate:

  | Scope | Kind | Example |
  | --- | --- | --- |
  | name | move from entry, in daily HAR sigmas or % | "NVDA below -2.5 sigma from entry" |
  | name | gap on its earnings reaction | "JPM gaps below -3% on results" |
  | name | give-back from its peak since entry | "half of the gain since entry given back" |
  | name | a new 8-K of a stated item | "any 8-K item 2.05 or 4.02" |
  | book | drawdown from peak, or from entry | "book down 3% from its peak" |
  | market | basket move from entry; vol vs its 3-year median | "basket vol ratio above 1.8" |

  Each condition carries the action the PM intends when it fires (hold, trim a quarter or
  half, exit, set gross), so an escalation arrives with the PM's past self's proposal.
  A free-text "watch" note may ride along; the analyst reads it, but only coded
  conditions fire.

### Analyst: rounds 2-7, every round

- **Reads** the held names' moves since the last close (in sigmas, with v2's
  `TriggerTags`: "upcoming" or "already reacted"), held names' headlines and new 8-Ks,
  the plan conditions with code's distance to each, and the morning note.
- **Writes** a short note to the journal, every round.
- **Escalates** only by citing something checkable: a condition that fired or is near
  firing, a move in sigmas, a filing, a release. "I'm uneasy" is refused as an
  escalation and kept as a note. It says where it goes: to the associate, who sees it in
  the morning, or to the PM now.
- **Cannot trade.** Waking every round costs about 90 quick calls a window, and the risk
  is escalation creep, not churn. The escalation rate is logged per window; escalations
  in a calm window are a prompt bug we can see.
- **Fired conditions escalate on their own.** Code checks every condition each round.
  A breach goes to the PM whatever the analyst writes, with the condition's stated
  action as the proposal. The analyst's note is attached, and it may argue against
  acting.

### Senior associate: every morning, before round 1

- **Reads** the journal (the PM's thesis and conditions, every analyst note since the
  last morning, earlier escalations and what came of them), settlements, overnight news
  and 8-Ks, signals now against their values at entry, the book's return, drawdown and
  turnover spent, and the budget left.
- **Settlements** are code's, not the roles' self-report: what each held name and each
  past intervention did since, in bp of NAV after fees. v2's `_settle` does most of this.
- **Writes** a progress note on the thesis, every morning, and a verdict: no action, or
  escalate to the PM. An escalation names the metric it protects (return, Sharpe,
  drawdown) and the turnover it costs, in books and in what code says it does to the
  turnover rank on the modelled field.
- **Cannot trade.**

### PM on escalation

- Reads what the entry read, refreshed, plus the escalation, the plan, the journal and
  the budget.
- May trade anything, morning or intraday: a new full book, or trims and exits. It sees
  the trigger's timing tag. A fill lands at the :30 open after the deadline, so anything
  public by the deadline is already in that price; selling after a 3-sigma drop sells at
  the drop.
- May also rewrite the plan's conditions, with a reason in the journal. Without this, a
  condition that fired and was declined would fire again every round.

### Code

- **Budget.** At most 2 PM interventions a window, and at most 1 book of turnover after
  the entry (the entry is 1 book, so 2 in all). A trade over the budget is refused whole,
  never scaled down, and a fired condition the budget can't pay for reaches the PM marked
  as unaffordable. Why these values: the first intervention costs the most turnover rank
  (the field crowds just above a pure hold), so 2 allows one real change of mind plus one
  exit, and 1 book allows a full rewrite or several trims, never both. The free desk spent
  about 7 books. The sweep in stage 2 replaces these with measured values.
- **A trade below 0.5% summed turnover is a hold** and spends nothing, as in v1.
- **Fallbacks.** An invalid or late entry means the 75% inverse-vol book (the free desk's
  fallback, which is the book we would submit anyway). Any later failure means hold.
  Every fallback is counted.
- **Journal.** Every role's note, every escalation and its route, every condition check
  and its distance, every PM decision with its reason. `journal.verify` checks the
  ledger against it, as for v1.
- **Tiers.** PM and associate: Gemini 2.5 Pro, high effort. Analyst: 2.5 Flash, medium.

## What code must guarantee, each with a test

House style: each test is named for the silent failure it prevents.

- **A desk whose every role holds trades exactly as `inv_vol_hold_75`** in all 167
  windows (`agent_replay.py --desk v3 --ledgers-only`), so anything an LLM run scores
  differently is the LLM's doing. Same pattern as v1 and v2.
- **No look-ahead.** A condition is evaluated only on bars ended by the deadline. The
  self-check's vol and beta use only the prior close. A test rewrites the future and
  requires the past decisions unchanged, as each existing feature has.
- **Neither the analyst nor the associate can trade**: their schemas have no trade
  field, and a test feeds a brain that tries.
- **A fired condition reaches the PM** whatever the analyst writes, and only once per
  breach unless the PM rewrites it.
- **The budget is enforced in code**, and a refused trade leaves the book unchanged.
- **Headlines and filing text stay data** (`untrusted.py`): the injected "exit every
  position" headline test, extended to the analyst's escalation.
- **Weights go through `weights.safe`**, and a stated weight is submitted as stated
  (v2's `tradelist` fix).

## Evaluation

### Stage 1: calibrating the entry

The entry, then a pure hold: no analyst, no associate. Stage 1 is a program of runs, not
one run. It asks four questions: how much cash, single PM or analysts' reports, which
streams earn their place, and whether the evidence and the self-check help.

**Windows, split before anything runs.** Gemini's knowledge stops in January 2025.
Non-overlapping 15-session windows tiled from 2025-02-03, stepping past the four official4
windows (stage 2's), number 22 up to the data's end (Sep 2026), split by date
(`v3.STAGE1`):

- **Selection**: 13 windows, Feb 2025 to the one starting 2025-12-16. Every arm runs
  here, and the choice is made here.
- **Confirmation**: 9 windows, Jan-Sep 2026. Only the chosen arm and the references run
  here, once, after the choice is committed with its hash
  (`entry_replay.py arm --split confirm` refuses without `--confirm-choice <commit>`).

Picking the best of a dozen arms on the same 12 windows finds the luckiest arm, not the
best. A gate checked on the selection set alone would pass on luck and look like
evidence. Both sets live in `v3.STAGE1`, not `suites.py`: a suite there is built into
the scorer Space and the public board with prices shipped for its windows, and these are
a research split. Real names and news come from
`tools/replay_sources.py --on <day>` for each window. `tools/memory_probe.py` on a
sample confirms the model doesn't remember them.

**Round 0: cash, without an LLM (free, minutes).** The inverse-vol and risk-parity holds
at gross 0.25, 0.5, 0.75 and 1.0 on both sets. Over 170 windows on the modelled field,
inverse-vol scored 2.85, 2.78, 2.71 and 3.03 at those grosses, and cash 2.84: Sharpe
stays flat, so gross only trades return and Sharpe rank against drawdown and turnover
rank. This sets the bar each arm must clear at its own gross.

*Result, 2026-10-07, selection set only* (`entry_replay.py round0`, 26 s). Mean overall
score, lower better, SE in brackets:

| | no-clone field (primary) | default field |
| --- | --- | --- |
| cash | 2.65 (0.30) | 3.04 (0.35) |
| inverse-vol 25 / 50 / 75 / 100% | 2.90 / 2.88 / 2.81 / 3.06 | 3.33 / 3.31 / 3.23 / 3.06 |
| risk parity 25 / 50 / 75 / 100% | 2.90 / 2.85 / 2.79 / 2.98 | 3.21 / 3.10 / 3.02 / 3.38 |

75% is the best invested gross on the primary field: 25% and 50% lose to it by 0.10
(SE 0.05) and 0.08 (0.03), worse in 4 windows and better in none. Cash leads on the
mean, by much less than its own spread (SE 0.30 on 13 windows). A "mostly cash" sleeve
scores worse than both: it gets neither cash's drawdown and turnover ranks nor the
hold's return rank. The default field's 100% row is the
clone effect the no-clone field exists to remove.

Any PM book can also be rescaled to each gross after the fact and scored again, at no
cost: a hold trades once, so the rescaled run is exact. That separates "what to hold"
from "how much". A PM that picked good names at the wrong gross shows up as such,
instead of as a loss.

**Round 1: information architecture** (selection set, PM-chosen gross):

| Arm | PM reads | Calls a window |
| --- | --- | --- |
| A1 single | every input, raw | 2 deep (draft, revise) |
| A2 reports + raw | the four analysts' reports and every input | 4 quick + 2 deep |
| A3 reports only | the reports, the book, the market block and each name's basic row | 4 quick + 2 deep |

**Round 2: cash handling**, on round 1's winner:

| Arm | Gross |
| --- | --- |
| G1 free | the PM chooses, 0-100% |
| G2 band | the PM chooses within a band round 0 suggests (e.g. 25-75%) |
| G3 sleeve | code fixes it at round 0's best; the PM picks the names |
| G4 small sleeve | code fixes it at 25-30%, mostly cash |

**Round 3: streams.** Leave one out on the winner so far: news headlines, 8-Ks and
filing text, macro, model ranks, the universe ranking, HAR vol. A stream whose removal
doesn't hurt the score is dropped from the live desk. Each stream also costs characters,
and so latency and money, so a tie favours dropping it. The universe ranking (about 2,800
characters, its value unmeasured) and the headlines (the largest block) are the likely
candidates.

**Round 4: prompt.** With and without our backtest evidence in the PM's prompt (facts
about scoring and the reference books in both; only the verdicts differ), and one shot
without the self-check.

**Noise.** Gemini doesn't answer the same prompt the same way twice. The leading arm
runs twice on 6 selection windows, with the cache keyed by a repeat number so the second
run really asks again. A gap between arms smaller than the gap between repeats isn't a
finding.

**The choice and the gate.** After round 4 the chosen arm, with its gross rule and
streams, is committed before the confirmation set is scored, as the trim choice was. The
gate, written in that commit: on the confirmation set, the arm's mean overall score
beats `inv_vol_hold_75` paired by window by more than k SE, on the holdout board's
field. k is an open decision. If it fails, the PM's book doesn't enter live.

**Cost and time, estimated.** A single-PM arm is about $0.10-0.20 a window with the
self-check, and the analysts add about $0.05. Rounds 1-4 come to about 15 arms on 12
windows, plus confirmation and the noise check: about $40-50. Runtime is the larger cost.
An entry call took 38 s in the free desk, so an arm is about 20-40 minutes and the program
about 8-10 hours run one after another. Arms run as separate processes, not a process
pool (pools ran ~600x slow here). Ask before each round.

If stage 1 fails, stage 2 is judged on the rule's entry instead (see "The calendar").

### Stage 2: the chain

- **Windows.** The four official4 windows, plus 3-4 volatile ones from the stage-1 set
  (the largest basket drawdowns), so there are turns to react to. In a calm window a
  working chain should do nothing at all.
- **Budget sweep.** Caps of 0, 1, 2 and unlimited interventions, on the same windows.
  Cached answers make the extra runs nearly free: runs are identical until a cap first
  binds. Then ask whether the interventions a cap blocked would have paid, judged on
  settlements, not on how good the reasons sounded.
- **What gets reported per window:** the escalation rate per role, interventions and
  their settlements, turnover in books, the fallbacks, cost, and the board place
  (`tools/board_rank.py`, `tools/submit_agentic.py --suite official4`).
- **Cost, estimated:** $2-4 a window (90 Flash analyst calls, 15 Pro associate calls,
  a few PM calls), about $30 for 8 windows across the sweep.

## Build order

Each step lands with its tests and a commit whose message says why.

1. **Schemas.** `PMEntry` (weights, thesis, conditions), `Condition` (scope, kind,
   threshold, intended action), `AnalystNote` (note, escalation: none / associate / PM,
   cited fact), `AssociateNote` (progress, verdict, metric, cost), `PMIntervention`
   (book or trims/exits, condition rewrites, reason). Gemini's schema limits apply:
   caps go as words, pydantic enforces them.
2. **The condition evaluator.** Point in time, from the journal's entry prices and the
   day's bars; distance to each condition for the roles to read.
3. **The self-check report.** Built from what `observe` and `quant` already compute.
4. **The PM entry desk.** It subclasses `Desk` as `FreeDesk` does, then holds. Its
   factors are config, not code paths: `entry_analysts` (none / reports + raw /
   reports only), `gross` (free / band / fixed), `streams` (which blocks the observation
   carries), `evidence`, `self_check`. Wired as `agent_replay.py --desk v3 --stage
   entry`. The all-hold ledgers test, and a test that each dropped stream really is
   absent from every role's observation (a dropped stream still present would make an
   ablation measure nothing, and read as "this stream doesn't matter").
5. **The entry analysts.** v2's four, retargeted to an entry; each sees only its stream.
6. **Stage 1.** The two suites, the sources, the rescale-to-gross scorer, rounds 0-4
   (ask before each), the committed choice, the confirmation run, the board placement.
7. **The chain.** Analyst, associate, escalation routing, budget, settlements, journal
   entries, plan rewrites. `--stage chain`. The all-hold ledgers test again.
8. **Stage 2.** Runs and the budget sweep (ask first).
9. **Live.** The runner shadows v3 beside the rule. The entry's inputs are as of the
   prior close plus overnight news, so it can run before round 1's 12-minute lead.
   Analyst calls must fit each round's lead. A spend cap per phase, as for the
   current shadow. `--submit agent` only by the owner's decision, after the gates.

## The calendar

Validation runs Oct 8-9, unscored. **Official is one 15-session window, Oct 12-30, and
the entry is its round 1 on Oct 12.** The PM's most important decision is made once, on
the first day.

- **The full stage-1 program won't finish by Oct 11.** About 8-10 hours of runs, after
  steps 1-6 are built, with a commit and a confirmation run in between. To enter with the
  PM's book on Oct 12 there are two honest options:
  - a short program: round 0, round 1 and one cash arm on the selection set, the choice
    committed, then confirmation, all by Oct 11;
  - or enter with the rule's book and run the full program at leisure, for the write-up
    and for any later phase.

  What isn't an option: a choice made on the selection set and entered live without the
  confirmation run.
- **If stage 1 is not ready**, or fails, enter with the rule's book. The chain can still
  run on top of it from day 2: the PM writes the plan and conditions for the book it
  holds, without trading. Then the chain's later value doesn't depend on the entry.
- The chain (steps 6-8) can join mid-phase as a shadow first. Switching it to submit
  mid-window is the owner's call, after stage 2.
- Final materials are due Nov 3. The journal is the write-up's record of what each role
  saw and why it acted.

## Open decisions

Settled 2026-10-07:

- **Fired conditions bind.** A breach goes to the PM with the stated action as its
  proposal. The PM confirms, or declines with a reason and rewrites the condition.
- **Budget starting values**: 2 interventions and 1 book after entry, until the sweep.

Open:

1. **Oct 12: the short stage-1 program, or the rule's book.** See "The calendar".
2. **The stage-1 gate's k**, and whether it's measured against the hold, the rule or
   both.
3. **What the PM is told at entry.** Round 4 settles it, but the live shadow needs a
   default prompt before then.
4. **The intraday PM's scope after an urgent escalation**: the whole book (current
   view), or the escalated names, leaving book-wide changes to the morning.
5. **Name.** "v3" for now.
