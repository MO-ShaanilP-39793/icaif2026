---
title: ICAIF 2026 holdout harness
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
---

# ICAIF 2026 holdout scorer (private)

This page scores an agent's `decisions.json` on Jan–Jun 2026 with the icaif2026 simulator
and the organizers' own metric calculator. The file holds one run per rolling 15-day
window, each from $1M in cash, and you get the four metrics for every window.

Submitting needs no sign-in. It uses the Space variable `SUBMIT_TOKEN`, a fine-grained
token that can write only to the entry dataset (MO-AI-Inv/icaif2026-holdout-entries,
private, the record) and to the public board (MO-AI-Inv/icaif2026-leaderboard). This Space
must stay private: it serves that token and the price file to whoever can open it.

It runs in the browser through Pyodide 0.29.5 (Python 3.13, pandas 2.3.3). The uploaded
file never leaves the page.

It is built by `tools/build_holdout_space.py` in the private icaif2026 GitHub repo. That
is the source of truth, so don't edit here. It ships only the harness, the ledger and
the vendored validator. It has no models and no strategy code.

The price files hold Alpaca execution opens and daily closes for Dec 2025 onwards only.
They are here for team use, and the Space must stay private.
