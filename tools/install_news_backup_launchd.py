"""A daily copy of the headline archive to S3, as a macOS launchd job (proposed; not installed).

    .venv/bin/python tools/install_news_backup_launchd.py            # print the job, change nothing
    .venv/bin/python tools/install_news_backup_launchd.py --install  # write and load it (owner)
    .venv/bin/python tools/install_news_backup_launchd.py --remove

`data/external/news/` is the only record of what the desk could have read before each
deadline: Yahoo cannot be asked later what it showed, so a lost disk loses the live
shadow's evidence for good. The job runs `aws s3 sync` of that folder to
s3://shaanil/icaif2026/data/external/news/, the repo's copy of record, once a day.

- **Never `--delete`.** A sync that mirrored deletions would let a wiped or half-restored
  folder empty the only other copy. Snapshots are never rewritten, so a plain sync
  uploads each new file once. The bucket also holds alphaBT production prefixes; the job
  writes only under icaif2026/.
- **It runs on the owner's AWS session.** With SSO credentials, the job fails whenever
  the session has lapsed, until `aws sso login`. Every run appends its outcome to
  output/news_backup.log, `ok` or `FAILED (exit n)`, so a lapse shows there rather than
  as a backup that quietly stopped. Credentials scoped to that one prefix would remove
  the lapse; that is the owner's call.
- **Local time.** launchd schedules in the Mac's local time, so the ET time is converted
  for today's offsets; rerun after either side changes daylight saving (US clocks move on
  2026-11-01), as with tools/install_news_launchd.py. A Mac asleep at the time runs the
  job once on wake.

Nothing is installed without `--install`: installing a job is the owner's decision.
"""

import argparse
import plistlib
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icaif import calendar  # noqa: E402

LABEL = "com.icaif2026.news-backup"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
SOURCE = ROOT / "data" / "external" / "news"
DEST = "s3://shaanil/icaif2026/data/external/news/"
# After the archiver's last snapshot of the US day (15:20 ET) and before the next 08:00.
AT_ET = time(16, 30)
LOG = ROOT / "output" / "news_backup.log"


def command(aws: str) -> str:
    sync = shlex.join([aws, "s3", "sync", str(SOURCE.resolve()), DEST, "--only-show-errors"])
    assert "--delete" not in sync
    stamp = 'date "+%Y-%m-%d %H:%M:%S %Z"'
    return (f'{sync}; s=$?; if [ $s -eq 0 ]; then echo "$({stamp}) ok"; '
            f'else echo "$({stamp}) FAILED (exit $s)"; fi')


def local_time() -> tuple[int, int]:
    et, local = ZoneInfo(calendar.TZ), datetime.now().astimezone().tzinfo
    run = datetime.combine(datetime.now(et).date(), AT_ET, tzinfo=et).astimezone(local)
    return run.hour, run.minute


def plist(aws: str) -> dict:
    h, m = local_time()
    return {"Label": LABEL, "ProgramArguments": ["/bin/sh", "-c", command(aws)],
            "WorkingDirectory": str(ROOT), "StartCalendarInterval": [{"Hour": h, "Minute": m}],
            "StandardOutPath": str(LOG), "StandardErrorPath": str(LOG), "RunAtLoad": False}


def launchctl(*args, check=True):
    return subprocess.run(["launchctl", *args], check=check, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--install", action="store_true", help="write and load the job (the owner's call)")
    g.add_argument("--remove", action="store_true")
    args = ap.parse_args()
    domain = f"gui/{subprocess.check_output(['id', '-u'], text=True).strip()}"
    if args.remove:
        launchctl("bootout", domain, str(PLIST), check=False)
        PLIST.unlink(missing_ok=True)
        print(f"removed {PLIST}")
        return
    aws = shutil.which("aws") or "/opt/homebrew/bin/aws"
    job = plist(aws)
    h, m = job["StartCalendarInterval"][0]["Hour"], job["StartCalendarInterval"][0]["Minute"]
    if not args.install:
        print(f"would write {PLIST}, daily at {h:02d}:{m:02d} local ({AT_ET:%H:%M} ET), logging to {LOG}:")
        print(plistlib.dumps(job).decode())
        print("nothing installed; --install writes and loads it")
        return
    if ".claude/worktrees/" in str(ROOT):
        # A session worktree is removed once merged, and the job's paths with it: it would
        # then fail every night into a log in a folder that no longer exists.
        raise SystemExit(f"{ROOT} is a session worktree; install from the main checkout")
    launchctl("bootout", domain, str(PLIST), check=False)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    with PLIST.open("wb") as f:
        plistlib.dump(job, f)
    launchctl("bootstrap", domain, str(PLIST))
    print(f"installed {PLIST}: daily at {h:02d}:{m:02d} local ({AT_ET:%H:%M} ET); log {LOG}")


if __name__ == "__main__":
    main()
