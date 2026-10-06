"""The evaluation suites: which 15-day windows a decisions file, an entry and a board are about.

    holdout    every rolling 15-session window that fits in Jan 2 - Jun 30 2026 (~109)
    official4  "Earnings season": four fixed 15-session windows, each opening as a quarter's
               reporting starts, chosen as the most like the Official phase (Oct 12-30 2026)

One definition, imported by the harness (`holdout`), the ranking (`leaderboard`) and both
pages, because each of them used to carry its own idea of "the board's windows". A suite
named in a file but defined differently in the scorer would score windows the agent was
never run on, and an entry ranked on the board under a third definition would sit beside
entries it shares no window with. Nothing would look wrong in any of the three.

A fixed suite states each window's first and last session, not just the first. The scorer
ships prices for those windows only, and a session missing from that file would otherwise
make "15 sessions from the start" run on into whatever the file holds next.

This module imports nothing from the repo: the public board ships it, and it may carry no
code beyond the ranking.
"""

from dataclasses import dataclass, replace
from datetime import date

WINDOW_DAYS = 15   # windows.WINDOW_DAYS; that module pulls in the simulator, so not imported
DEFAULT = "holdout"


@dataclass(frozen=True)
class Suite:
    name: str
    title: str
    span: tuple                 # (first, last) ISO day any window of the suite may touch
    windows: tuple = ()         # fixed suites only: ((first session, last session), ...)
    n_days: int = WINDOW_DAYS

    @property
    def fixed(self) -> bool:
        return bool(self.windows)

    def narrowed(self, start: str, end: str) -> "Suite":
        """A rolling suite over a shorter span: scoreable, never submittable (it is not
        the suite any more, and `is_canonical` says so)."""
        if self.fixed:
            raise ValueError(f"suite {self.name} has fixed windows; it has no span to narrow")
        return replace(self, span=(str(date.fromisoformat(str(start))),
                                   str(date.fromisoformat(str(end)))))

    def doc(self) -> dict:
        return {"name": self.name, "title": self.title, "span": list(self.span),
                "windows": [list(w) for w in self.windows], "n_days": self.n_days}


def _fixed(name: str, title: str, windows: tuple) -> Suite:
    return Suite(name, title, (windows[0][0], windows[-1][1]), windows)


SUITES = {s.name: s for s in (
    Suite("holdout", "Holdout, Jan 2 - Jun 30 2026",
          ("2026-01-02", "2026-06-30")),
    _fixed("official4", "Earnings season",
           (("2025-04-11", "2025-05-02"), ("2025-10-13", "2025-10-31"),
            ("2026-04-13", "2026-05-01"), ("2026-07-13", "2026-07-31"))),
)}


def get(suite) -> Suite:
    """A Suite from its name (or itself). An unknown name raises, naming the known ones."""
    if isinstance(suite, Suite):
        return suite
    if suite not in SUITES:
        raise KeyError(f"no suite {suite!r}; the suites are {', '.join(SUITES)}")
    return SUITES[suite]


def is_canonical(suite: Suite) -> bool:
    """True for a suite exactly as defined here: the only kind an entry may be scored on."""
    return SUITES.get(suite.name) == suite
