"""Stage 1 of the cross-sectional attention model: learn to rank (design doc, Ensemble
models tab, "Stretch").

A GRU shared by every name reads its last 20 sessions of features. Two attention
layers then run across the names plus a market token, with each pair's trailing
return correlation added to the attention logits (a learned scale per head and
layer). With every node attending to every other, that is the GNN on a full graph.
About 50k parameters.

It trains on random 30-name subsets of the ~104-name daily universe, which gives many
more cross-sections than the 30 names alone, with a share of batches drawn as the
competition 30 themselves so the model sees the set it will score. The loss is
listwise (ListNet): the softmax of the scores against the softmax of the labels'
within-set z-scores. Only the order within a set matters, as with the IC it is judged on.

Two properties that fail silently when broken, each with a test:
- **Permutation equivariance.** Nothing may depend on a name's position in the set.
  A positional encoding, or a GRU run across names instead of across time, would let
  the model learn the (alphabetical) ticker order.
- **Padding is invisible.** A day with 27 of the 30 names pads to 30. A padded slot
  must not change any real name's score, or a missing name would shift everyone else.

The folds, purges and IC are pass 1's (`train.fold_split`, `train.per_decision_ic`),
so the numbers compare directly with the AutoGluon ensemble's.
"""

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from icaif import gnn_data
from icaif.gnn_data import CORR_MIN_OVERLAP, CORR_WINDOW, HISTORY, Tensors

SET_SIZE = 30
NEG = -1e4  # attention logit for a padded key; -inf turns a fully padded row into NaN


@dataclass
class Config:
    d_model: int = 48
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch: int = 64
    sets_per_day: int = 2
    competition_share: float = 0.25
    label_beta: float = 1.0      # ListNet target sharpness on within-set z-scores
    max_epochs: int = 40
    patience: int = 6
    train_days: int | None = None  # keep only the last n training days (smoke runs)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, S, d = z.shape
        q, k, v = self.qkv(self.ln1(z)).view(B, S, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.p if self.training else 0.0)
        z = z + self.drop(self.out(a.transpose(1, 2).reshape(B, S, d)))
        return z + self.drop(self.ff(self.ln2(z)))


class CrossSectionalRanker(nn.Module):
    def __init__(self, n_feat: int, n_ctx: int, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.inp = nn.Linear(n_feat, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.market = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)])
        # Starts at zero: the model begins as a plain set transformer and has to earn
        # the correlation bias.
        self.corr_scale = nn.Parameter(torch.zeros(cfg.layers, cfg.heads))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, corr: torch.Tensor,
                pad: torch.Tensor) -> torch.Tensor:
        """x [B, L, n, F], ctx [B, C], corr [B, n, n], pad [B, n] (True = padded) -> [B, n]."""
        B, L, n, _ = x.shape
        h = self.inp(x).permute(0, 2, 1, 3).reshape(B * n, L, -1)
        _, last = self.gru(h)
        h = last[0].view(B, n, -1)
        z = torch.cat([self.market(ctx)[:, None], h], dim=1)
        c = F.pad(corr, (1, 0, 1, 0))                      # the market token has no correlation
        keys = F.pad(pad, (1, 0), value=False)
        mask = torch.zeros_like(keys, dtype=z.dtype).masked_fill(keys, NEG)[:, None, None, :]
        for i, blk in enumerate(self.blocks):
            z = blk(z, self.corr_scale[i][None, :, None, None] * c[:, None] + mask)
        return self.head(z[:, 1:]).squeeze(-1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- batching on device

class DeviceData:
    """The tensors on the training device, with the fold's context standardisation.

    Context is z-scored with the training days' mean and std only. Statistics over the
    whole sample would carry the test years' VIX and rate levels into training.
    """

    def __init__(self, t: Tensors, train_days: np.ndarray, dev: torch.device):
        self.t, self.dev = t, dev
        c = t.ctx[train_days]
        mu, sd = np.nanmean(c, axis=0), np.nanstd(c, axis=0)
        ctx = np.clip((t.ctx - mu) / np.where(sd > 0, sd, 1.0), -5, 5)
        self.ctx = torch.tensor(np.nan_to_num(ctx), dtype=torch.float32, device=dev)
        self.x = torch.tensor(t.x, device=dev)
        self.ret = torch.tensor(t.ret, device=dev)
        self.y = torch.tensor(np.nan_to_num(t.y, nan=0.0), device=dev)
        self.hist_off = torch.arange(-HISTORY + 1, 1, device=dev)
        self.corr_off = torch.arange(-CORR_WINDOW, 0, device=dev)

    def batch(self, days: np.ndarray, names: np.ndarray, pad: np.ndarray):
        """days [B], names [B, n] (ticker positions, any value where padded), pad [B, n]."""
        d = torch.tensor(days, device=self.dev)
        nm = torch.tensor(names, device=self.dev)
        rows = (d[:, None] + self.hist_off).clamp(min=0)
        x = self.x[rows[:, :, None], nm[:, None, :]]                 # B L n F
        r = self.ret[(d[:, None] + self.corr_off).clamp(min=0)[:, :, None], nm[:, None, :]]
        corr = masked_corr_torch(r)
        y = self.y[d[:, None], nm]
        return x, self.ctx[d], corr, torch.tensor(pad, device=self.dev), y


def masked_corr_torch(r: torch.Tensor, min_overlap: int = CORR_MIN_OVERLAP) -> torch.Tensor:
    """Batched `gnn_data.masked_corr`: r [B, W, n] with NaN holes -> [B, n, n]."""
    m = (~torch.isnan(r)).float()
    v = torch.nan_to_num(r) * m
    mt = m.transpose(1, 2)
    n = mt @ m
    sx = v.transpose(1, 2) @ m
    sxx = (v * v).transpose(1, 2) @ m
    sxy = v.transpose(1, 2) @ v
    cov = sxy - sx * sx.transpose(1, 2) / n.clamp(min=1)
    var = (sxx - sx * sx / n.clamp(min=1)).clamp(min=0)
    c = cov / torch.sqrt(var * var.transpose(1, 2)).clamp(min=1e-12)
    ok = (n >= min_overlap) & (var > 0) & (var.transpose(1, 2) > 0)
    return torch.where(ok, c, torch.zeros_like(c)).clamp(-1, 1)


def pack(sets: list[np.ndarray], size: int = SET_SIZE) -> tuple[np.ndarray, np.ndarray]:
    names = np.zeros((len(sets), size), np.int64)
    pad = np.ones((len(sets), size), bool)
    for i, s in enumerate(sets):
        names[i, :len(s)] = s
        pad[i, :len(s)] = False
    return names, pad


def sample_sets(t: Tensors, days: np.ndarray, cfg: Config, rng: np.random.Generator):
    """Training sets for one epoch: (day, names) pairs, shuffled."""
    out = []
    for day in days:
        ok = gnn_data.eligible(t, day, need_label=True)
        if len(ok) < 10:
            continue
        comp = ok[t.is_competition[ok]]
        for _ in range(cfg.sets_per_day):
            if len(comp) >= 10 and rng.random() < cfg.competition_share:
                out.append((day, comp))
            else:
                out.append((day, rng.choice(ok, size=min(SET_SIZE, len(ok)), replace=False)))
    rng.shuffle(out)
    return out


def eval_sets(t: Tensors, days: np.ndarray, seed: int = 0):
    """Validation sets, fixed across epochs: each day's competition names, then the rest
    of the universe split at random into groups of 30 (a group under 10 is dropped)."""
    rng = np.random.default_rng(seed)
    out = []
    for day in days:
        ok = gnn_data.eligible(t, day, need_label=True)
        comp = ok[t.is_competition[ok]]
        rest = rng.permutation(ok[~t.is_competition[ok]])
        groups = [comp] + [rest[i:i + SET_SIZE] for i in range(0, len(rest), SET_SIZE)]
        out += [(day, g) for g in groups if len(g) >= 10]
    return out


def listnet_loss(scores: torch.Tensor, y: torch.Tensor, pad: torch.Tensor, beta: float) -> torch.Tensor:
    """Cross-entropy of softmax(scores) against softmax(beta * within-set z-score of y)."""
    live = (~pad).float()
    cnt = live.sum(1, keepdim=True)
    mu = (y * live).sum(1, keepdim=True) / cnt
    sd = torch.sqrt((((y - mu) * live) ** 2).sum(1, keepdim=True) / cnt).clamp(min=1e-6)
    target = torch.softmax(((y - mu) / sd * beta).masked_fill(pad, NEG), dim=1)
    logq = torch.log_softmax(scores.masked_fill(pad, NEG), dim=1)
    return -(target * logq * live).sum(1).mean()


def set_ic(pred: np.ndarray, y: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """Spearman within each set (row), padded slots excluded."""
    p = pd.DataFrame(np.where(pad, np.nan, pred)).rank(axis=1).to_numpy()
    q = pd.DataFrame(np.where(pad, np.nan, y)).rank(axis=1).to_numpy()
    p -= np.nanmean(p, axis=1, keepdims=True)
    q -= np.nanmean(q, axis=1, keepdims=True)
    return np.nansum(p * q, 1) / np.sqrt(np.nansum(p * p, 1) * np.nansum(q * q, 1))


@torch.no_grad()
def predict(model: nn.Module, dd: DeviceData, sets: list, batch: int = 256) -> tuple:
    model.eval()
    preds, ys, pads = [], [], []
    for i in range(0, len(sets), batch):
        chunk = sets[i:i + batch]
        names, pad = pack([s for _, s in chunk])
        x, ctx, corr, padt, y = dd.batch(np.array([d for d, _ in chunk]), names, pad)
        preds.append(model(x, ctx, corr, padt).float().cpu().numpy())
        ys.append(y.cpu().numpy())
        pads.append(pad)
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(pads)


@dataclass
class FitResult:
    model: nn.Module
    data: DeviceData
    best_epoch: int
    best_val_ic: float
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def fit(t: Tensors, train_days: np.ndarray, val_days: np.ndarray | None, cfg: Config, seed: int,
        dev: torch.device | None = None, epochs: int | None = None, log=print) -> FitResult:
    """Train with early stopping on the validation sets' mean IC, or for a fixed number
    of `epochs` with no validation (the refit on train + validation)."""
    dev = dev or device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if cfg.train_days:
        train_days = train_days[-cfg.train_days:]
    dd = DeviceData(t, train_days, dev)
    model = CrossSectionalRanker(t.x.shape[2], t.ctx.shape[1], cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    vsets = eval_sets(t, val_days) if val_days is not None and epochs is None else None
    best, best_state, best_epoch, hist = -math.inf, None, 0, []
    t0 = time.time()
    for epoch in range(1, (epochs or cfg.max_epochs) + 1):
        model.train()
        sets = sample_sets(t, train_days, cfg, rng)
        losses = []
        for i in range(0, len(sets), cfg.batch):
            chunk = sets[i:i + cfg.batch]
            names, pad = pack([s for _, s in chunk])
            x, ctx, corr, padt, y = dd.batch(np.array([d for d, _ in chunk]), names, pad)
            loss = listnet_loss(model(x, ctx, corr, padt), y, padt, cfg.label_beta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if vsets is not None:
            p, y, pad = predict(model, dd, vsets)
            ic = set_ic(p, y, pad)
            row["val_ic"] = float(np.nanmean(ic))
            row["val_ic_30"] = float(np.nanmean(ic[_first_of_day(vsets)]))
            if row["val_ic"] > best:
                best, best_epoch, best_state = row["val_ic"], epoch, copy.deepcopy(model.state_dict())
        hist.append(row)
        log(row)
        if vsets is not None and epoch - best_epoch >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(hist)
    return FitResult(model, dd, best_epoch, best, hist, time.time() - t0)


def _first_of_day(sets: list) -> np.ndarray:
    """Mask of each day's first set, which `eval_sets` makes the competition names."""
    days = np.array([d for d, _ in sets])
    return np.r_[True, days[1:] != days[:-1]]


def score_competition(fit_: FitResult, days: np.ndarray) -> pd.Series:
    """Scores for the competition names present on each day, as one set per day: the
    cross-section the competition trades. Index (date, ticker)."""
    t, sets = fit_.data.t, []
    for day in days:
        ok = np.flatnonzero(t.present[day] & t.is_competition)
        if len(ok) >= 2:
            sets.append((day, ok))
    p, _, pad = predict(fit_.model, fit_.data, sets)
    idx = [(t.dates[d], t.tickers[n]) for (d, s) in sets for n in s]
    vals = np.concatenate([p[i, ~pad[i]] for i in range(len(sets))])
    return pd.Series(vals, index=pd.MultiIndex.from_tuples(idx, names=["date", "ticker"]), name="pred")
