"""The public board's only Python entry point: rank entry texts, return JSON.

It imports `icaif.leaderboard` and, through it, `icaif.ranking`, and nothing else from
the repo. The board Space is public, so the build ships exactly these files and checks
that importing this module loaded nothing more. It has no prices, no kit and no
simulator.
"""

import json

from icaif import leaderboard


def boot() -> None:
    pass


def board(texts: list[str]) -> str:
    """The standings from entry JSON texts (references and submissions).

    Ranked on every load rather than stored ranked: a rank depends on the whole field,
    so a stored one would go stale the moment another entry arrived.
    """
    try:
        return json.dumps(leaderboard.standings([json.loads(t) for t in texts]))
    except leaderboard.EntryError as err:
        return json.dumps({"error": str(err)})
