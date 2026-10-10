# icaif2026 for MLSys: start here

> Snapshot 2026-10-10, main @ 81dcfb5, branch `mlsys-handoff`. **Draft status:** each topic doc
> was written by an agent from the code, README, reports and git history. A fact-checking pass
> started on every doc but was stopped before it finished, so some carry partial fixes. Treat
> every number as a pointer to its source, and re-check it there before it goes into the paper.

These docs are for a colleague adapting our ICAIF 2026 Trading Agent Competition entry into an
MLSys paper, and for his coding agents. They cover the data and the architecture as they stand.
Each topic doc ends with what is contest-specific and how to generalise it for the paper:
stronger models than Gemini 2.5 Pro, more data, a larger universe, other markets and strategies.

## Reading order

| Doc | Read it for |
| --- | --- |
| [00_contest_and_evaluation.md](00_contest_and_evaluation.md) | The task, the score (four ranks against a field), the modelled field, windows and suites, holdout discipline, baselines |
| [01_data_streams.md](01_data_streams.md) | Every data source, point-in-time rules, vendor parity, storage, licensing |
| [02_daily_ensemble_model.md](02_daily_ensemble_model.md) | The daily cross-sectional AutoGluon model, and the intraday and GNN side branches |
| [03_har_vol_forecaster.md](03_har_vol_forecaster.md) | The HAR realized-volatility forecaster and where it is used |
| [04_ou_process_and_quant_signals.md](04_ou_process_and_quant_signals.md) | The OU residual s-score, the regime read, risk parity, the rule desk, the quant race |
| [05_news_feeds.md](05_news_feeds.md) | Headline sources, point-in-time stamps, selection, untrusted text, contamination |
| [06_sec_filings.md](06_sec_filings.md) | EDGAR 8-Ks, earnings times, the acceptance-time fault, triggers |
| [07_three_level_hierarchy.md](07_three_level_hierarchy.md) | Desk v3 (PM, senior associate, analyst) as designed and as built |
| [08_agent_runtime_and_lineage.md](08_agent_runtime_and_lineage.md) | Brains, schemas, caching, cost, journal, triggers; the rule desk, v1 and v2 |
| [09_stage1_doe.md](09_stage1_doe.md) | The stage-1 designed experiment, its tooling, and results so far |
| [10_stage2_escalation_chain.md](10_stage2_escalation_chain.md) | The planned escalation chain and its experiments |
| [11_systems_infrastructure.md](11_systems_infrastructure.md) | Live runner, process model, failure handling, replays, scorer and leaderboard, tests |
| [13_repo_map_and_glossary.md](13_repo_map_and_glossary.md) | Module, tool, test and report maps, commands, branches, glossary |

Not written yet: `12_paper_directions.md` (candidate framings, the contest-to-paper relaxations,
the knowledge-cutoff problem for newer models, a plan to the deadline).

## Where the work stands (2026-10-10)

- The contest's Official phase is one 15-session window, Oct 12-30; final materials are due
  Nov 3. Desk v3's entry is what the owner submits on Oct 12 (`stage1_doe.md`).
- Stage 1 tunes that entry by a designed experiment. Rounds 0 and 1 are done, and round 1
  chose the reports-only architecture (commit 6ae9ed6). Round 2, the 16-run streams factorial,
  is running in the owner's v3 worktree. Rounds 3-4 and the hold-out check follow.
- Stage 2, the escalation chain, is designed in `v3_desk_plan.md` and not built.

## Ground rules for you and your coding agents

- The contest is live until Oct 30. Never run `tools/live_runner.py` (any subcommand),
  `tools/live_dry_run.py` or `arm`, and never upload to Codabench or the HuggingFace Spaces.
- Don't write into the owner's worktrees (`.claude/worktrees/v3` and others), or onto `main`
  without the owner. Work on your own branch.
- Paid runs print an estimate and stop without `--yes`. Get the owner's approval before
  spending, and bring your own API keys.
- `s3://shaanil/icaif2026/` is the copy of record, and the same bucket holds alphaBT production
  data. Pull freely; write only under a prefix the owner agrees to; never use `--delete`.
- Never commit, print or log `.env`, `.icaif/` or any key. Use no alphaBT or client data in the
  paper. The organizer's hourly panel is licensed to participants; check its terms and Alpaca's
  before redistributing either.
- The repo's `CLAUDE.md` and `README.md` hold the invariants (no look-ahead, never forward-fill
  a price, weights through `weights.safe`) and the house style. Follow them.

The design doc "ICAIF 2026 - System Overview" linked from `CLAUDE.md` could not be read when
these were written. Its Roadmap tab is the owner's plan of record.
