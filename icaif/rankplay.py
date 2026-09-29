"""Play for rank, not return: choose each morning's exposure by its expected final score.

The score is the mean of four ranks against the field at the end of a 15-day window,
so the right action depends on where the book stands. Ahead on return with a clean
drawdown, cutting risk protects three ranks for little cost. Behind, only risk can
climb. That is a tournament (Brown, Harlow & Starks 1996), and the policy that plays it
is a planner over the rest of the window:

1. **Where everyone stands.** Each entrant of a stand-in field (and the book itself)
   has metrics to date: completed period returns, turnover spent, drawdown so far.
   They come from the simulator run on prices up to the deadline only.
2. **What could happen.** The rest of the window is bootstrapped from past days, in
   blocks of consecutive days so volatility clusters as it does. A day is the seven
   period relatives the competition scores: the overnight from the previous 15:30 fill
   to 09:30, then six intraday rounds.
3. **How each choice would finish.** For every candidate exposure, every entrant is
   carried along the same simulated paths (common random numbers, so the choice, not
   the draw, separates them) and ranked on the kit's own four formulas, ties sharing
   the average rank. The book takes the exposure with the lowest expected score, and
   moves only if that beats holding by a margin: a move on a difference inside the
   Monte Carlo noise is turnover bought for nothing.

**What the field emulation assumes.** A hold is carried exactly. An active entrant is
carried as its current book, charged its own to-date turnover rate as fee drag and
turnover; its picks are not re-simulated. Before anyone has traded (a window's first
morning), each entrant's book and turnover rate come from its run on the 15 sessions
before the window. The real field is unobservable until the end, so the planner's
field is a model; `tools/rankplay_report.py` scores it against a field it did not plan
against for exactly that reason.

**Look-ahead.** Entrants' to-date values are read from full-window simulator runs,
cut at the deadline. A strategy's path up to t depends only on prices up to t, so the
cut is exact, and the test that rewrites every later price and requires the same
decision guards the seam.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from icaif import calendar, quant_strategies as qs, sim, windows
from icaif import weights as W

FEE = sim.FEE_RATE
SHARPE_SCALE = np.sqrt(1764.0)  # the kit's: 252 days x 7 decision periods
ROUNDS = 7


# ----------------------------------------------------------------------------- entrant states

@dataclass
class EntrantState:
    """An entrant at the start of the period in progress at the deadline."""

    shares: np.ndarray    # after its trade at that period's start
    cash: float
    nav_start: float      # nav_before of the period in progress, pre-fee
    ref_px: np.ndarray    # execution prices at that period's start
    rets: np.ndarray      # completed period returns
    ratios: np.ndarray    # notional / nav_before, every period started so far
    peak: float           # valuation high-water mark to date
    mdd: float            # maximum drawdown to date
    tau: float            # future turnover per period (fee drag for active entrants)
    target: Optional[np.ndarray] = None  # weights to buy at the next fill (first morning)
    started: bool = True  # False before the window's first period: nothing in progress


def state_at(res: sim.Result, deadline: pd.Timestamp, market: sim.Market) -> Optional[EntrantState]:
    """The entrant's state at `deadline`, read only from what had happened by then.

    None when no period has started yet (the window's first morning).
    """
    execs = [p["execution"] for p in res.periods]
    k = int(np.searchsorted(pd.DatetimeIndex(execs).asi8, pd.Timestamp(deadline).value,
                            side="right")) - 1
    if k < 0:
        return None
    row = res.ledger.iloc[k]
    tickers = market.tickers
    rets = np.array([p["nav_after_period"] / p["nav_before"] - 1 for p in res.periods[:k]])
    ratios = np.array([p["traded_notional"] / p["nav_before"] for p in res.periods[:k + 1]])
    pts = [v for v, t in zip(res.valuation_points, res.valuation_times)
           if t is None or t <= deadline]
    peak, mdd = pts[0], 0.0
    for v in pts[1:]:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    # Active entrants keep trading; charge them the rate they have shown after entry.
    later = ratios[1:] if len(ratios) > 1 else ratios[:0]
    return EntrantState(shares=row[tickers].to_numpy(dtype=float), cash=float(row["cash"]),
                        nav_start=res.periods[k]["nav_before"],
                        ref_px=market.exec_prices.loc[execs[k]].to_numpy(dtype=float),
                        rets=rets, ratios=ratios, peak=peak, mdd=mdd,
                        tau=float(later.mean()) if len(later) else 0.0)


def profile(res: sim.Result, market: sim.Market) -> tuple[np.ndarray, float]:
    """(final weights, turnover rate after entry) from a completed run: an entrant's
    habit, used before it has traded in the window being planned."""
    last = res.ledger.iloc[-1]
    px = market.exec_prices.loc[res.periods[-1]["execution"]].to_numpy(dtype=float)
    value = last[market.tickers].to_numpy(dtype=float) * px
    nav = float(last["cash"] + value.sum())
    ratios = np.array([p["traded_notional"] / p["nav_before"] for p in res.periods])
    first = int(np.argmax(ratios > 0)) if (ratios > 0).any() else len(ratios)
    after = ratios[first + 1:]
    return value / nav, float(after.mean()) if len(after) else 0.0


# ----------------------------------------------------------------------------- scenarios

def day_units(market: sim.Market, deadline: pd.Timestamp) -> np.ndarray:
    """(days, 7, tickers) price relatives of every full day completed before `deadline`.

    Unit u = [exec1(u) / exec7(u-1), exec2(u) / exec1(u), ..., exec7(u) / exec6(u)].
    Only pairs of full seven-round days count: a half-day has four rounds and no
    unit shape. A unit whose last fill is not before the deadline is not yet known.
    """
    if not hasattr(market, "_rank_units"):
        ex = market.exec_prices
        days, units, ends = [], [], []
        by_day = {d: [r["execution"] for r in calendar.rounds_for(d)] for d in market.days}
        for prev, d in zip(market.days, market.days[1:]):
            a, b = by_day[prev], by_day[d]
            if len(a) != ROUNDS or len(b) != ROUNDS:
                continue
            px = ex.loc[[a[-1]] + b].to_numpy(dtype=float)
            units.append(px[1:] / px[:-1])
            ends.append(b[-1])
        market._rank_units = np.array(units)
        market._rank_unit_ends = pd.DatetimeIndex(ends).asi8
    n = int(np.searchsorted(market._rank_unit_ends, pd.Timestamp(deadline).value, side="left"))
    return market._rank_units[:n]


def bootstrap(units: np.ndarray, n_days: int, n_paths: int, rng: np.random.Generator,
              block: int = 5, pool: int = 250) -> np.ndarray:
    """(paths, n_days * 7, tickers) relatives: blocks of consecutive past days.

    Holes (NaN relatives, a name missing a fill) become 1: in a scenario the name
    simply doesn't move that period, which is what a held position does over a hole.
    """
    u = units[-pool:]
    if len(u) < block:
        raise ValueError(f"only {len(u)} past days to bootstrap from")
    n_blocks = -(-n_days // block)
    starts = rng.integers(0, len(u) - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :n_days]
    paths = u[idx]                           # (paths, days, 7, tickers)
    paths = np.nan_to_num(paths, nan=1.0)
    return paths.reshape(n_paths, n_days * ROUNDS, u.shape[-1])


# ----------------------------------------------------------------------------- finishing

@dataclass
class Finish:
    """Each path's final four metrics for one entrant."""

    ret: np.ndarray
    sharpe: np.ndarray
    mdd: np.ndarray
    turnover: np.ndarray


def finish(s: EntrantState, rel: np.ndarray, n_periods: int,
           target: Optional[np.ndarray] = None) -> Finish:
    """Carry an entrant along `rel` (paths, steps, tickers) and score it.

    Step 0 is the next fill (09:30 today, after the overnight relative). With a
    `target`, the entrant re-targets there to those weights of its NAV, pays the fee
    and holds; otherwise it holds, less its fee drag. The scored window has
    `n_periods` periods; periods past the simulated steps are not modelled (the last
    15:30-to-close stub), and every entrant is cut at the same step, so none gains.
    """
    px = s.ref_px[None, None, :] * np.cumprod(rel, axis=1)   # prices at each future fill
    p0 = px[:, 0, :]
    v_pre0 = s.cash + p0 @ s.shares                          # (paths,)
    ratio0 = np.zeros(len(v_pre0))
    tgt = target if target is not None else s.target
    if tgt is not None:
        new = tgt[None, :] * v_pre0[:, None] / p0
        notional = (np.abs(new - s.shares[None, :]) * p0).sum(axis=1)
        cash = v_pre0 - (new * p0).sum(axis=1) - FEE * notional
        shares = new
        ratio0 = notional / v_pre0
        values = cash[:, None] + np.einsum("psk,pk->ps", px, shares)
    else:
        values = s.cash + px @ s.shares
    steps = values.shape[1]
    drag = (1 - s.tau * FEE) ** np.arange(steps)               # active entrants' fees
    values = values * drag[None, :]
    # Period-end values: the period in progress ends at step 0 before any trade there;
    # each later period ends at the next step. A period starts from the NAV *before*
    # its own trade, so step 0's fee lands in the period it opens, as the kit charges
    # it; starting from the post-fee value would lose the fee from every return.
    ends = values.copy()
    ends[:, 0] = v_pre0
    starts = np.concatenate([np.full((len(v_pre0), 1), s.nav_start), ends[:, :-1]], axis=1)
    fut = ends / starts - 1
    if not s.started:
        # Before the first fill nothing is in progress: step 0 opens period 0.
        fut, ends = fut[:, 1:], ends[:, 1:]
    k0 = len(s.rets)
    want = max(0, n_periods - k0)
    if fut.shape[1] < want:
        # The last day's 15:30-to-close stub (and any period past the simulated days)
        # is carried flat: an unmodelled period, the same for every entrant.
        pad = want - fut.shape[1]
        fut = np.concatenate([fut, np.zeros((len(fut), pad))], axis=1)
        ends = np.concatenate([ends, np.repeat(ends[:, -1:], pad, axis=1)], axis=1)
    fut, ends = fut[:, :want], ends[:, :want]
    count = k0 + fut.shape[1]
    mean = (s.rets.sum() + fut.sum(axis=1)) / count
    ss = (((s.rets[None, :] - mean[:, None]) ** 2).sum(axis=1)
          + ((fut - mean[:, None]) ** 2).sum(axis=1))
    sd = np.sqrt(ss / max(count - 1, 1))
    sharpe = np.where(sd > 1e-15, SHARPE_SCALE * mean / np.where(sd > 0, sd, 1.0), 0.0)
    peak = np.maximum(s.peak, np.maximum.accumulate(ends, axis=1))
    mdd = np.maximum(s.mdd, (1 - ends / peak).max(axis=1))
    # Trades: the book's own at step 0 (ratio0), then its turnover habit per period.
    later = max(fut.shape[1] - (1 if s.started else 0) - 1, 0)
    turn = (s.ratios.sum() + ratio0 + s.tau * later) / count
    ret = ends[:, -1] / sim.INITIAL_NAV - 1
    return Finish(ret, sharpe, mdd, turn)


def expected_scores(ours: dict[str, Finish], field: list[Finish]) -> dict[str, float]:
    """Mean over paths of our overall rank score, per candidate action.

    Ranked as `ranking.rank_window` does: return and Sharpe high-to-low, drawdown and
    turnover low-to-high, ties sharing the average rank.
    """
    out = {}
    for name, f in ours.items():
        entrants = field + [f]
        score = 0.0
        for attr, sign in (("ret", -1), ("sharpe", -1), ("mdd", 1), ("turnover", 1)):
            m = np.stack([getattr(e, attr) for e in entrants], axis=1) * sign
            # Rounded so floating-point dust doesn't split what the kit's Decimals tie.
            score = score + rankdata(np.round(m, 12), method="average", axis=1)[:, -1]
        out[name] = float((score / 4).mean())
    return out


# ----------------------------------------------------------------------------- the strategy

class _Replay:
    def __init__(self, decisions):
        self.decisions = decisions

    def __call__(self, ctx):
        return self.decisions.get((ctx.day, ctx.round))


@dataclass
class RankPlayConfig:
    plan_field: dict = field(default_factory=lambda: None)
    window_days: int = 15
    grid: tuple = (0.0, 0.3, 0.5, 0.75, 0.9, 1.0)
    shape: str = "risk_parity"
    n_paths: int = 300
    block: int = 5
    pool: int = 250
    hysteresis: float = 0.05
    seed: int = 0
    entry_only: bool = False  # plan the entry exposure, then hold whatever happens


class RankPlayer:
    """A `sim.Strategy`: at each round 1, the exposure with the best expected rank."""

    def __init__(self, config: Optional[RankPlayConfig] = None):
        from icaif import baselines

        self.cfg = config or RankPlayConfig()
        self.field_def = self.cfg.plan_field or baselines.FIELD
        self.decisions: dict = {}
        self.start = None
        self.day_no = 0
        self.shape: Optional[pd.Series] = None
        self.log: list[dict] = []

    def _field(self, market):
        """Full-window runs and prior-window profiles of the planning field, cached per
        market and window: each morning reads them only up to its own deadline."""
        cache = market.__dict__.setdefault("_rank_field", {})
        key = (id(self.field_def), self.start, self.cfg.window_days)
        if key not in cache:
            runs = {n: sim.run(f(), market, self.start, self.cfg.window_days)
                    for n, f in self.field_def.items()}
            i = market.days.index(self.start)
            prior = market.days[max(0, i - self.cfg.window_days)]
            profs = {n: profile(sim.run(f(), market, prior, self.cfg.window_days), market)
                     for n, f in self.field_def.items()} if i >= self.cfg.window_days else {}
            cache[key] = (runs, profs)
        return cache[key]

    def _own_state(self, ctx) -> Optional[EntrantState]:
        if not self.decisions:
            return None
        res = sim.run(_Replay(self.decisions), ctx.market, self.start, self.cfg.window_days)
        return state_at(res, ctx.deadline, ctx.market)

    def _start_state(self, ctx, px) -> EntrantState:
        return EntrantState(shares=np.zeros(len(px)), cash=sim.INITIAL_NAV,
                            nav_start=sim.INITIAL_NAV, ref_px=px, rets=np.array([]),
                            ratios=np.array([]), peak=sim.INITIAL_NAV, mdd=0.0, tau=0.0,
                            started=False)

    def __call__(self, ctx):
        if ctx.round != 1:
            return None
        if self.cfg.entry_only and self.decisions:
            return None
        m = ctx.market
        if self.start is None:
            self.start = ctx.day
        self.day_no += 1
        days_left = self.cfg.window_days - self.day_no + 1
        n_periods = sum(len(calendar.rounds_for(d)) for d in
                        m.days[m.days.index(self.start):][: self.cfg.window_days])
        units = day_units(m, ctx.deadline)
        if len(units) < self.cfg.pool // 2:
            # Holding cash here would read as a cautious choice in the ranks (cash
            # ranks respectably) when it is really a planner with nothing to plan on.
            raise ValueError(f"{len(units)} past days to bootstrap from on {ctx.day}; "
                             f"the planner needs at least {self.cfg.pool // 2}")
        # The last fill before the deadline prices every book (yesterday's 15:30).
        last_fill = m.exec_prices.index[m.exec_prices.index < ctx.deadline][-1]
        px = m.exec_prices.loc[last_fill].to_numpy(dtype=float)
        if self.shape is None:
            rets = np.log(qs.daily_closes(ctx, qs.SHAPE_DAYS + 1)).diff().iloc[1:]
            shape = qs.SHAPES[self.cfg.shape](rets)
            if shape is None:
                return None
            self.shape = shape.reindex(m.tickers).fillna(0.0).to_numpy()

        runs, profs = self._field(m)
        field_states = []
        for n in self.field_def:
            st = state_at(runs[n], ctx.deadline, m)
            if st is None:
                st = self._start_state(ctx, px)
                if n in profs:
                    st.target, st.tau = profs[n]
            field_states.append(st)
        own = self._own_state(ctx) or self._start_state(ctx, px)
        value = own.cash + own.ref_px @ own.shares
        gross = float((own.ref_px * own.shares).sum() / value) if value > 0 else 0.0
        cur_w = own.ref_px * own.shares / value if gross > 0 else self.shape

        rng = np.random.default_rng(self.cfg.seed * 7919 + ctx.day.toordinal())
        rel = bootstrap(units, days_left, self.cfg.n_paths, rng, self.cfg.block, self.cfg.pool)
        field_fin = [finish(s, rel, n_periods) for s in field_states]
        cands = {"hold": None} if gross > 0 else {}
        base = cur_w / cur_w.sum() if cur_w.sum() > 0 else self.shape
        for e in self.cfg.grid:
            cands[f"e{e:.2f}"] = base * e
        ours = {k: finish(own, rel, n_periods, target=v) if v is not None else
                finish(own, rel, n_periods) for k, v in cands.items()}
        exp = expected_scores(ours, field_fin)
        best = min(exp, key=exp.get)
        if "hold" in exp and exp["hold"] - exp[best] < self.cfg.hysteresis:
            best = "hold"
        self.log.append({"day": ctx.day, "day_no": self.day_no, "gross": gross,
                         "choice": best, **{f"exp_{k}": v for k, v in exp.items()}})
        if best == "hold" or (gross == 0 and best == "e0.00"):
            return None
        target = cands[best]
        w = W.safe({t: float(min(x, W.CAP)) for t, x in zip(m.tickers, target)}, m.tickers)
        self.decisions[(ctx.day, ctx.round)] = w
        return w


def rank_player(**kw):
    """A `windows.run_field` factory."""
    return lambda: RankPlayer(RankPlayConfig(**kw))
