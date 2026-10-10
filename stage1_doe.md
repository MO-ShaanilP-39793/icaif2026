# Stage 1 design of experiment: tuning v3's entry

Drafted 2026-10-08, revised the same day: **v3 is what we submit** (the owner's decision),
so these experiments choose its settings; they are not a test of whether to use it.
Stage 1 only: the PM decides the entry on day 1, and the book is held
untouched to the window's end. Stage 2 (the escalation chain) comes later, on its own
windows. This replaces the "Rounds 1-4" section of `v3_desk_plan.md` once agreed.

## The question

Which settings give v3's entry the best contest score: the mean of four ranks (return,
Sharpe, max drawdown, turnover) against a field of rivals, paired window by window, lower
better. Three design choices are tested:

| Factor | Levels |
| --- | --- |
| **S. Data streams** (7, each in or out) | HAR vol forecast, model ranks, headlines, 8-K filings, macro, universe ranking, regime read |
| **A. Architecture** (3) | single PM on raw data; four analysts' reports + raw data; reports only |
| **C. Cash floor** (7) | none; at least 20%, 40%, 50%, 75%, 90%, 95% cash (gross capped at 100, 80, 60, 50, 25, 10, 5%) |

Always in, never tested: returns, 20-day vol, earnings timing, the OU score and the two
reference shapes' weights. Those are the minimum a book is chosen from. Held fixed:
evidence off, self-check on, Gemini 2.5 Pro (high) for the PM, 2.5 Flash (medium) for
analysts. The PM chooses gross freely under the floor; a floor is a limit, not a target.

A full factorial is 2^7 x 3 x 7 = 2,688 arms. On 13 windows that's ~35,000 entries and
~$3,500, so the design below is sequential and fractional.

## Units, blocks and responses

- **Unit**: one 15-session window, run from cash at its first round, as Official will run.
- **Windows**: the 13 selection windows in `v3.STAGE1["select"]` (Feb 2025 - Jan 2026).
  That's one more than the 12 proposed, and it costs nothing extra to keep. The 9
  hold-out windows (2026) are not touched until the choice is committed.
- **Blocks**: every arm runs on the same 13 windows. The market moves a window's score far
  more than any design choice does, so every comparison is **paired within a window**,
  and the window's own swing cancels.
- **Responses, per window per arm**:
  - the four raw metrics: return, Sharpe, max drawdown, turnover. These don't depend on
    any field;
  - each metric's rank, and the overall score, against the modelled field: the no-clone
    field as primary, the default field beside it. Each arm is ranked **alone** against
    the field, never against the other arms: arms that crowd each other would favour
    whichever arm is least like its neighbours, not the one that beats the field;
  - diagnostics: gross chosen, names held, effective names, largest weight, beta,
    fallbacks, revisions, cost and latency.
- **Note on turnover**: a hold trades once, so its turnover equals its gross. Turnover
  rank is decided by the cash choice alone, and streams or architecture can't move it.

## The design, in four rounds

### Round 0: baselines (done, free)

Inverse-vol and risk-parity holds at 25, 50, 75 and 100% gross, and cash, on the 13
windows. 75% is the best invested gross (25% and 50% lose by ~0.1, SE 0.03-0.05); cash
leads the mean at 2.65 against 2.81, within its SE of 0.30. These run beside every arm as
reference columns, so a result can be read against something simple. They decide
nothing.

### Round 1: architecture (3 arms)

A1 single, A2 reports + raw, A3 reports only, each with **all seven streams** and no cash
floor. All streams in is the condition under which the architectures differ most (the
analysts have the most to digest).

*Decision rule:* the lowest mean score on the primary field. If the best is within one
SE of a cheaper arm, take the cheaper arm (A1 < A3 < A2 in cost and latency).

*Result (2026-10-09/10): **reports only**, fixed by the owner.* Mean score 3.24 (single PM),
3.00 (reports + raw), 3.01 (reports only); the rule picks reports only as the cheaper of a
tie. On raw numbers reports only led all three measures that differ: mean return 0.49%
(0.22%, 0.09%), Sharpe 2.15 (1.68, 2.02), max drawdown 2.98% (3.20%, 3.22%), and it
finished best of the three in 7, 6 and 6 of 13 windows on them. Turnover tracks the
gross alone (all ~85% invested). Two of its PM calls first timed out while the Mac
slept; the arm was rerun awake, re-asking only those two.

### Round 2: streams (16 arms, a 2^(7-3) fractional factorial, resolution IV)

On round 1's architecture, no cash floor. Sixteen arms, each including a different half
of the streams, chosen so that every stream is in for 8 arms and out for 8, and each
stream's effect is estimated clear of any pairwise interaction (it is aliased only with
three-way and higher ones). Generators: E = ABC, F = BCD, G = ACD. Run 1 (all out) and
run 16 (all in) are among the sixteen.

| Run | HAR vol | model rank | headlines | filings | macro | universe | regime |
| ---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | - | - | - | - | - | - | - |
| 2 | - | - | - | + | - | + | + |
| 3 | - | - | + | - | + | + | + |
| 4 | - | - | + | + | + | - | - |
| 5 | - | + | - | - | + | + | - |
| 6 | - | + | - | + | + | - | + |
| 7 | - | + | + | - | - | - | + |
| 8 | - | + | + | + | - | + | - |
| 9 | + | - | - | - | + | - | + |
| 10 | + | - | - | + | + | + | - |
| 11 | + | - | + | - | - | + | - |
| 12 | + | - | + | + | - | - | + |
| 13 | + | + | - | - | - | + | + |
| 14 | + | + | - | + | - | - | - |
| 15 | + | + | + | - | + | - | - |
| 16 | + | + | + | + | + | + | + |

**Why a fraction rather than leave-one-out.** Leave-one-out (8 arms) estimates each
stream from one contrast: all-in minus all-but-one. The factorial estimates each stream
from all 16 arms, 8 against 8, in every window. For twice the runs, each effect's
variance from the LLM's own noise is about 8 times smaller (1/8 + 1/8 of one arm's,
against 1 + 1), and it says whether a stream helps in general, not
only when every other stream is present.

**Analysis.** For stream s in window w: effect(s, w) = mean score of the 8 arms with s,
minus the mean of the 8 without. Its effect is the mean over the 13 windows, with SE from
those 13 values. The same is done for each raw metric, so we can see how a stream helps,
not only whether (headlines may lower drawdown and cost return, for instance).

*Decision rule, as first written (ranks):* keep a stream if its effect is negative (helps)
by more than one SE. Drop it if it's positive by more than one SE (noise that hurts). In
between, **drop it**: each stream costs prompt length, latency and money, and a stream
that can't show its worth in 16 x 13 entries isn't earning its place.

*Decision rule, as revised by the owner after seeing the results (raw numbers):* keep a
stream if it improves at least two of return, Sharpe and max drawdown, and worsens none
by more than one SE (`doe_report.keep_streams_raw`). Raw numbers don't depend on our
guess at the rival field. This changed the rule after the data was seen, which is
recorded here so the hold-out check is read with that in mind.

*Result (2026-10-10), z = effect / SE, signed so positive is better:*

| Stream | Return | Sharpe | Max drawdown | Rank rule | Raw rule |
| --- | ---: | ---: | ---: | --- | --- |
| HAR vol | +0.29% (1.6) | +0.27 (0.9) | -0.19% (1.3) | drop | keep |
| regime read | +0.21% (1.3) | +0.20 (1.0) | -0.20% (1.1) | drop | keep |
| headlines | +0.18% (1.2) | +0.15 (0.5) | -0.06% (0.9) | drop | keep |
| model ranks | +0.21% (0.9) | +0.53 (1.2) | +0.15% (-0.8) | keep | keep |
| macro | +0.08% (0.5) | -0.15 (-0.8) | -0.14% (0.8) | drop | keep |
| universe ranking | +0.05% (0.2) | +0.32 (1.3) | +0.21% (-1.3) | keep | drop |
| 8-K filings | -0.20% (-1.2) | -0.08 (-0.3) | +0.10% (-1.0) | drop | drop |

**Fixed by the owner: every stream but the 8-K filings.** That is the raw rule's set plus
the universe ranking, whose Sharpe gain the owner kept over the rule's drawdown guard.
Round 2 cost about $22; one window's PM call hit a Gemini 503 and was re-asked.

### Round 3: cash floor (2 arms, plus free scoring of the rest)

**The free part.** A hold trades once, so any round-2 book bought again at a lower gross
is exactly what that gross would have done (`v3.BoughtBook`). Each cash floor can
therefore be scored by taking the chosen configuration's books and capping each at the
floor's gross where the PM went higher. That needs no new calls. It assumes the PM would
have picked the same names under a floor.

**The check on that assumption.** Two real arms run with the floor in the PM's prompt:
40% and 90% cash, one moderate and one extreme. For each window, compare the real
floor's book with the capped one: name overlap, weight correlation, and score
difference. If they agree within the noise level (round 4), the free scores stand for
all seven floors. If not, the remaining floors are run for real (5 more arms).

*Decision rule:* the floor with the lowest mean score on the primary field. Within one SE
of "no floor", choose no floor, since a constraint has to earn its place too. For
reference, round 0's holds scored worse at every deep floor than at 75% invested, so a
deep floor wins only if the PM's own names behave differently under it.

*Result so far (2026-10-10), base: reports only, every stream but 8-K filings.* The free
scores ranked 50% cash first (3.00 against 3.28 for no floor), but the check failed: told a
floor, the PM picks a different book, not a scaled one (about half the names shared with
its no-floor book, share correlation 0.14-0.23), and the real floors scored 0.3 better
(40%) and 0.3 worse (90%) than their capped stand-ins. Capping keeps Sharpe flat (1.58 to
1.53, as the owner reasoned); a real floor doesn't.

| | Mean return | Sharpe | Mean max DD | Turnover | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| no floor (65-96% invested) | -0.04% | 1.58 | 3.24% | 0.81% | 3.28 |
| real 40% floor (40-60% invested) | **+0.42%** | **2.52** | 2.05% | 0.53% | **2.79** |
| real 90% floor (10% invested) | -0.06% | 0.55 | 0.45% | 0.10% | 3.38 |
| *inverse-vol hold 75%* | *+0.12%* | *2.01* | *2.65%* | *0.72%* | *2.81* |

So, as the design says when the check fails, the other floors run for real (20, 50 and 75%,
run in parallel). The no-floor arm here (3.28) and round 2's all-streams row (3.01) differ
only by the filings stream, and round 2 says dropping it should help slightly, yet
the arm without it scored 0.27 worse: run-to-run noise is large, and round 4 measures it before any floor's lead is read as real.

*All real floors (2026-10-10; 20, 50 and 75% ran in parallel):*

| PM told: at least ... cash | Invested | Return | Sharpe | Max DD | Turnover | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 84% | -0.04% | 1.58 | 3.24% | 0.81% | 3.28 |
| 20% | 73% | +0.37% | 2.23 | 2.73% | 0.70% | 2.90 |
| 40% | 55% | +0.42% | 2.52 | 2.05% | 0.53% | 2.79 |
| 50% | 48% | -0.04% | 1.44 | 2.02% | 0.46% | 3.10 |
| 75% | 25% | +0.03% | 1.35 | 0.92% | 0.24% | 3.08 |
| 90% | 10% | -0.06% | 0.55 | 0.45% | 0.10% | 3.38 |

Drawdown and turnover fall steadily with cash; return and Sharpe don't follow it. 40% and
50% invested almost alike (55%, 48%) yet scored Sharpe 2.52 and 1.44: cash can't do that,
so the gap is which names the PM happened to pick. A real floor mostly measures the PM's
run-to-run variation, not the floor.

### The overlay: code sets the cash split, the PM is never told (owner's design)

Sharpe has a zero risk-free rate (`starter-kit/docs/evaluation.md`), so scaling a book
leaves it unchanged, while drawdown and turnover shrink with the scale. The owner's design
takes the cash split out of the PM's hands: the PM writes its book, and code scales it.
Every round-2 and round-3 book was already written by a PM that knew of no later scaling,
so the design is scored on all 221 of them (`doe_report.py overlay`, no calls):

| Invested | Return | Sharpe | Max DD | Turnover | DD rank | Turnover rank | Score | vs as chosen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| as the PM chose (65-96%) | 0.06% | 1.53 | 2.99% | 0.82% | 3.09 | 2.57 | 3.30 | |
| 50% | ~0 | 1.51 | 1.84% | 0.48% | 2.15 | 2.00 | 3.00 | -0.31 (better in 121, worse in 21) |
| 25% | ~0 | 1.49 | 0.92% | 0.24% | 2.00 | 2.00 | 3.02 | -0.28 |
| 10% | ~0 | 1.48 | 0.37% | 0.10% | 2.00 | 2.00 | 3.06 | -0.25 |
| 1% | ~0 | 1.47 | 0.04% | 0.01% | 2.00 | 2.00 | 3.07 | -0.23 |

Every overlay beats the PM's own cash choice. Below about 25% invested the drawdown and
turnover ranks stop improving (on the modelled field a pure-cash rival always holds rank
1 and every invested rival is already beaten), while the return rank keeps falling. So
whether 1% beats 25% turns on whether the real field holds a pure-cash team.

**A "maximise Sharpe" PM, under the overlay** (`--objective sharpe --gross fixed:1`: every
role told the book is judged on Sharpe alone; no prompt or input names a cash level).
Its books' Sharpe was 1.14 against the four-rank PM's 1.52 (difference -0.38, SE 0.60,
higher in 6 of 13), and under every overlay it scored 0.04-0.06 worse (SE 0.11-0.14).
Within noise both ways; the four-rank prompt stays.

### Round 4: noise (2 arms)

Gemini doesn't answer the same question the same way twice. The chosen configuration runs
twice more on all 13 windows, asked fresh each time (`--repeat 1`, `--repeat 2`). The
spread between the three runs of one arm is the noise floor: a difference between
arms smaller than it is not a finding. This runs last so it measures the configuration
that will actually be used. If it shows round 1 or 3's winners were within noise, the
tie rules above apply retroactively, and that is recorded.

## Hold-out check: once, no gate

The chosen settings are committed, then run once on the 9 hold-out windows (2026) beside
the runner-up of the closest call. This isn't a pass/fail gate; v3 ships either way. It
answers one question the 13 can't: did we pick the settings that are best, or the ones
that were luckiest on those 13 windows? Choosing the best of ~20 arms on the same windows
that score them favours luck. If the runner-up beats the choice on the hold-out, the
edge on the 13 didn't hold up: it was noise, and the tie rule (cheaper or simpler wins) decides. The
hold-out numbers are also the honest estimate of tuned v3 for the write-up.

## Size, cost and time

| Round | Arms | Entries | Cost (estimate) | Runtime (estimate) |
| --- | ---: | ---: | ---: | ---: |
| 0 | - | - | done | done |
| 1 | 3 | 39 | ~$5 | ~1 h |
| 2 | 16 | 208 | ~$20-30 | ~4-5 h |
| 3 | 2 (+5 if needed) | 26 (+65) | ~$3 (+$7) | ~30 min (+1.5 h) |
| 4 | 2 | 26 | ~$3 | ~30 min |
| Hold-out | 2 | 18 | ~$2 | ~20 min |
| **Total** | **25-30** | **~320-385** | **~$30-50** | **~6-8 h** |

Costs scale from the smoke test ($0.10 a window for the single PM); analyst arms add
about $0.05 a window. Arms run as separate processes, one after another, and each round
is asked for before it runs.

**Against the calendar.** Official's entry is round 1 on Oct 12, and it is v3's. Started
today, rounds 1-4 can finish by Oct 10 and the hold-out check by Oct 11. If a round slips,
v3 enters with the settings decided so far, and defaults for the rest: single PM, all
streams, no cash floor.

## What this design can and can't tell us

- **It can** find which streams carry signal for the PM, tell the three ways of
  feeding it apart if they differ by more than the noise, and price the cash choice
  exactly.
- **It can't** separate settings that differ by less than the noise round 4 measures.
  Those are ties, and the tie rules decide them by cost and simplicity, which is the
  right call when the data can't.
- **Every number is conditional on the modelled field.** The real field is unknown until
  Official. A conclusion that flips between the two fields depends on our guess about
  rivals, and is reported as such.
- **News is judged on post-cutoff windows only**, where the model can't remember the
  outcome. Each window's memory is spot-checked with `tools/memory_probe.py`.

## Built (2026-10-08)

- `entry_replay.py arm --cash-floor F`: the PM is told the floor (a band 0 to 1 - F).
- `entry_replay.py arm --design streams16 --run N`: row N of the factorial (`v3.STREAMS16`,
  built from its generators; a test checks it is balanced, orthogonal and resolution IV).
- Every arm also scores its books capped at each floor (`arm_floor_<pct>`), at no cost.
- `tools/doe_report.py arch | streams | cash | noise | table`: applies the rules above.
  A rule that can't compute an SE (too few windows) treats the comparison as a tie, so
  the simpler choice stands rather than whichever won by chance.

**One property of the streams rule to know.** Keeping a stream when it helps by more than
one SE lets a stream with no real effect through about 16% of the time. Across seven
streams that's about one useless stream kept, on average. Raising the bar to two SE cuts
that to ~2%, but then a real effect has to be about twice as large to be kept. One SE is
the current rule; a useless stream costs prompt length, not much else.
