"""The organizers' own validator and scorer, imported from the vendored starter kit.

Imported rather than reimplemented: a local copy of the weight rules or the metric
formula would agree with the backend until the day it didn't, and a backtest scored
by our formula is a number the leaderboard never computes.
"""

import sys
from decimal import Decimal

from icaif.data import ROOT

_KIT = ROOT / "starter-kit"
if str(_KIT) not in sys.path:
    sys.path.insert(0, str(_KIT))

from kit import contracts, evaluation  # noqa: E402

SubmissionError = contracts.SubmissionError


def validate_weights(weights: dict) -> None:
    """Raise SubmissionError exactly where the backend's local contract would."""
    contracts._weights(weights)


def metrics(periods: list[dict], valuation_points: list[float], initial_nav: float) -> dict:
    """The official four metrics, via the kit's Decimal calculator, returned as floats."""
    out = evaluation.calculate_metrics(
        [{k: repr(float(v)) for k, v in p.items()} for p in periods],
        [repr(float(v)) for v in valuation_points],
        Decimal(repr(float(initial_nav))),
    )
    return {k: float(v) for k, v in out.items()}
