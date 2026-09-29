"""The agent desk: LLM judgement on top of the quant models and the compiler.

Roles (the design doc's Overview tab, reshaped by the quant race in
`tools/quant_report.py`, which found that any trade after entry costs turnover ranks
the drawdown it saves does not buy back):

- **Strategist**, once, at the window's first round: the book's shape, its entry
  exposure and names to leave out. This is where judgement is cheapest, since the
  entry trade is paid for anyway.
- **Risk review**, each morning after entry: hold (the default the evidence backs) or
  move the exposure dial, when something the models cannot see justifies the ranks.
- **Event analyst**, only when a trigger fires on a held name (earnings before the
  next open, a move of several daily sigmas, a burst of news): hold or exit.

Code, not a model, does everything that must never be wrong: the observation (point in
time), the weights (the 30% cap, the 1e-6 grid), the budget and the fallback. Every
LLM answer is validated against a schema and the book; an answer that fails, times
out or is refused is replaced by the rule's answer for that role, and the swap is
logged. The rule itself (`brains.RuleBrain`) is the best entry-only quant candidate,
so the desk can never do worse than its fallback by failing, only by judging.
"""
