"""Install (or reinstall) a macOS launchd job that runs tools/news_archive.py.

    .venv/bin/python tools/install_news_launchd.py          # write, load, print times
    .venv/bin/python tools/install_news_launchd.py --remove

Runs five minutes before every round's upload deadline (09:05, 10:20, ... 15:20 ET)
and once more at 08:00 ET, every day: weekend and holiday snapshots are harmless and
keep the archive continuous. launchd schedules in the Mac's *local* time, so the ET
times are converted for today's offsets. **Rerun it after either side changes
daylight saving** (US clocks move on 2026-11-01): the job would otherwise fire an
hour off. The Official phase (Oct 12-30) sits inside one offset.

A Mac that is asleep at a scheduled time runs the job once on wake, so the snapshot
is late but its timestamp is honest. Keep the machine awake and online through US
market hours for the archive to be before-the-deadline.
"""

import argparse
import plistlib
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icaif import calendar  # noqa: E402

LABEL = "com.icaif2026.news-archive"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LEAD = timedelta(minutes=5)
EXTRA_ET = [(8, 0)]


def local_times() -> list[tuple[int, int, str]]:
    """(hour, minute, what) in local time, for today's ET and local offsets."""
    et, local = ZoneInfo(calendar.TZ), datetime.now().astimezone().tzinfo
    today = datetime.now(et).date()
    slots = [(dl, f"r{n} deadline {dl:%H:%M} ET") for n, (dl, _) in calendar.ROUNDS.items()]
    slots += [(datetime.min.replace(hour=h, minute=m).time(), f"daily {h:02d}:{m:02d} ET")
              for h, m in EXTRA_ET]
    out = []
    for t, what in slots:
        run = datetime.combine(today, t, tzinfo=et) - (LEAD if what.startswith("r") else timedelta())
        loc = run.astimezone(local)
        out.append((loc.hour, loc.minute, what))
    return sorted(out)


def launchctl(*args, check=True):
    return subprocess.run(["launchctl", *args], check=check, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()
    domain = f"gui/{subprocess.check_output(['id', '-u'], text=True).strip()}"
    launchctl("bootout", domain, str(PLIST), check=False)
    if args.remove:
        PLIST.unlink(missing_ok=True)
        print(f"removed {PLIST}")
        return

    times = local_times()
    log = ROOT / "output" / "news_archive.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": LABEL,
        "ProgramArguments": [str(ROOT / ".venv" / "bin" / "python"),
                             str(ROOT / "tools" / "news_archive.py")],
        "WorkingDirectory": str(ROOT),
        "StartCalendarInterval": [{"Hour": h, "Minute": m} for h, m, _ in times],
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
        "RunAtLoad": False,
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    with PLIST.open("wb") as f:
        plistlib.dump(plist, f)
    launchctl("bootstrap", domain, str(PLIST))
    print(f"installed {PLIST}; log {log}")
    for h, m, what in times:
        print(f"  {h:02d}:{m:02d} local  ({what})")


if __name__ == "__main__":
    main()
