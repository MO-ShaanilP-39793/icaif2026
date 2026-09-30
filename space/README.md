---
title: ICAIF 2026 holdout harness
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - write-repos
hf_oauth_authorized_org: MO-AI-Inv
---

# ICAIF 2026 holdout harness (private)

This page scores an agent's `decisions.json` on Jan–Jun 2026 with the icaif2026 simulator
and the organizers' own metric calculator. You get the four metrics for one continuous
run from $1M, and for every rolling 15-day window.

Submissions need "Sign in with HuggingFace" and write access to MO-AI-Inv. Each entry is
recorded in the private dataset MO-AI-Inv/icaif2026-holdout-entries and copied here under
`entries/`.

It runs in the browser through Pyodide 0.29.5 (Python 3.13, pandas 2.3.3). The uploaded
file never leaves the page.

It is built by `tools/build_holdout_space.py` in the private icaif2026 GitHub repo. That
is the source of truth, so don't edit here. It ships only the harness, the ledger and
the vendored validator. It has no models and no strategy code.

The price files hold Alpaca execution opens and daily closes for Dec 2025 onwards only.
They are here for team use, and the Space must stay private.
