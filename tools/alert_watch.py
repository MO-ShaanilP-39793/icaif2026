"""Push an alert to the owner's phone when a live round goes wrong, or the runner goes quiet.

    .venv/bin/python tools/alert_watch.py --phase official      # cron, every minute
    .venv/bin/python tools/alert_watch.py --test                 # one test message

The runner records everything worth knowing in its phase directory (state.json,
rounds.jsonl, runner.log), but nobody reads them while a round runs: a refused entry,
a PM that fell back to the coded book, or a runner that died overnight would each be
found only afterwards. This reads those files, never writes them, and never imports the
runner, so it cannot slow or break an upload. A dead runner writes no file to alert on,
so silence during the phase is itself an alert.

Messages go to ntfy.sh (NTFY_TOPIC in the environment or .env; the topic name is the
only access control, so it is long and random). They carry status text only: round,
outcome, gross, names. Never weights, receipts or anything from decision.json, which
holds the team token.

Each alert is sent once (output/live/alerts/<phase>.json). A send that fails is not
marked sent, so the next minute tries again.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIVE = ROOT / "output" / "live"
SCHEDULE = ROOT / "starter-kit" / "schedule.json"
# A worker may run until 5 minutes past a deadline 12 minutes after its wake, and the
# scheduler logs at least every 10 minutes while it waits: 25 minutes of quiet is
# longer than any healthy gap.
QUIET = timedelta(minutes=25)
# The runner must be up before round 1 wakes (12 minutes before its deadline).
OPEN_LEAD = timedelta(minutes=45)
OK_UPLOADS = (None, "hold", "dry-run", "uploaded", "executed")


def setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    env = ROOT / ".env"
    if not value and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
    return value


def ts(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):   # absent, or caught mid-rename: try next minute
        return default


def read_lines(path: Path) -> list[dict]:
    rows = []
    try:
        text = path.read_text()
    except OSError:
        return rows
    for x in text.splitlines():
        try:
            rows.append(json.loads(x))
        except ValueError:
            continue
    return rows


def rounds_of(phase_dir: Path, phase: str) -> list[dict]:
    """The phase's rounds: the runner's latest read of the server schedule, else the kit's."""
    sched = read_json(phase_dir / "schedule.json", None) or read_json(SCHEDULE, {})
    rows = [r for r in sched.get("rounds", []) if r.get("phase") == phase
            and r.get("status") != "CANCELLED"]
    return sorted(rows, key=lambda r: ts(r["deadline"]))


def runner_alive(phase: str):
    """True or False from /proc on Linux; None where there is no /proc (unknown)."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in argv]
        if (any(a.endswith("live_runner.py") for a in args) and "run" in args
                and phase in args):
            return True
    return False


def events(phase: str, phase_dir: Path, now: datetime, alive) -> list[dict]:
    """Every alert the files call for now: {key, title, body, priority}. Sending decides
    which are new."""
    out = []
    state = read_json(phase_dir / "state.json", {})
    recs = {r.get("round_id"): r for r in read_lines(phase_dir / "rounds.jsonl")}
    for rid, s in (state.get("rounds") or {}).items():
        rec = recs.get(rid, {})
        outcome = str(s.get("outcome") or rec.get("outcome") or "")
        up = s.get("upload")
        upload = (rec.get("submitted") or {}).get("upload") or {}
        if up not in OK_UPLOADS:
            why = str(upload.get("why", ""))[:200]
            out.append({"key": f"upload:{rid}", "priority": "urgent", "tags": "rotating_light",
                        "title": f"{rid}: upload {up}", "body": f"{outcome}\n{why}".strip()})
        elif up in ("uploaded", "executed"):
            agent = rec.get("agent") or {}
            gross = agent.get("gross")
            out.append({"key": f"entry:{rid}", "priority": "default", "tags": "white_check_mark",
                        "title": f"{rid}: entry {up}",
                        "body": (f"submission {upload.get('submission_id')}"
                                 + (f", gross {gross:.3f}, {agent.get('names')} names"
                                    if isinstance(gross, (int, float)) else ""))})
        if outcome.startswith(("missed", "no submission")):
            out.append({"key": f"outcome:{rid}", "priority": "high", "tags": "warning",
                        "title": f"{rid}: {outcome[:80]}", "body": outcome[:300]})
        fell = [x.get("role") for x in (rec.get("agent") or {}).get("roles") or []
                if x.get("source") == "fallback"]
        if fell:
            out.append({"key": f"fallback:{rid}", "priority": "high", "tags": "warning",
                        "title": f"{rid}: {', '.join(map(str, fell))} fell back",
                        "body": "The model was not consulted; the coded book stood in. "
                                "See agent.roles in round.json."})
        if rec.get("errors"):
            out.append({"key": f"errors:{rid}", "priority": "high", "tags": "warning",
                        "title": f"{rid}: {len(rec['errors'])} error(s)",
                        "body": "\n".join(str(e)[:200] for e in rec["errors"][:3])})
    entry = state.get("entry") or {}
    if entry.get("status") in ("executed", "failed"):
        out.append({"key": f"receipt:{entry.get('round_id')}:{entry['status']}",
                    "priority": "urgent" if entry["status"] == "failed" else "default",
                    "tags": "receipt", "title": f"entry receipt: {entry['status']}",
                    "body": f"{entry.get('round_id')}, submission {entry.get('submission_id')}"})

    rows = rounds_of(phase_dir, phase)
    if rows and ts(rows[0]["deadline"]) - OPEN_LEAD <= now <= ts(rows[-1]["deadline"]) + timedelta(minutes=15):
        beats = [p.stat().st_mtime for p in (phase_dir / "runner.log", phase_dir / "schedule.json",
                                              phase_dir / "state.json") if p.exists()]
        last = datetime.fromtimestamp(max(beats), timezone.utc) if beats else None
        quiet = last is None or now - last > QUIET
        if alive is False or quiet:
            why = ("no runner process" if alive is False else
                   "never started" if last is None else f"silent since {last:%H:%M} UTC")
            out.append({"key": "silent", "priority": "urgent", "tags": "skull",
                        "title": f"{phase} runner down: {why}",
                        "body": "No round will run until it is restarted (see HANDOFF_v3.md)."})
    return out


def send(msg: dict, topic: str, server: str = "https://ntfy.sh", email: str | None = None) -> None:
    """Post to the topic; with `email`, ntfy.sh also mails the message to that address."""
    from icaif import net

    headers = {"Title": msg["title"].encode("ascii", "replace").decode(),
               "Priority": msg["priority"], "Tags": msg.get("tags", "")}
    if email:
        headers["Email"] = email
    req = urllib.request.Request(f"{server}/{topic}", data=msg["body"].encode() or b" ", method="POST",
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=10, context=net.ssl_context()) as r:
        if r.status != 200:
            raise OSError(f"ntfy answered HTTP {r.status}")


def watch(phase: str, phase_dir: Path, seen_path: Path, now: datetime, alive, post) -> list[str]:
    """Send the new alerts; return their keys. 'silent' is forgotten once the runner is
    heard again, so a second outage alerts too."""
    seen = read_json(seen_path, {"sent": []})
    sent = set(seen["sent"])
    found = events(phase, phase_dir, now, alive)
    keys = {e["key"] for e in found}
    new = []
    if "silent" in sent and "silent" not in keys:
        sent.discard("silent")
        try:
            post({"key": "back", "title": f"{phase} runner heard again", "body": "Heartbeat is fresh.",
                  "priority": "low", "tags": "heartbeat"})
        except Exception as err:  # noqa: BLE001 - logged; nothing depends on it
            print(f"{now:%F %T} send failed (back): {type(err).__name__}: {err}", flush=True)
    for e in found:
        if e["key"] in sent:
            continue
        try:
            post(e)
        except Exception as err:  # noqa: BLE001 - not marked sent: retried next minute
            print(f"{now:%F %T} send failed ({e['key']}): {type(err).__name__}: {err}", flush=True)
            continue
        sent.add(e["key"])
        new.append(e["key"])
    seen_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = seen_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sent": sorted(sent)}))
    os.replace(tmp, seen_path)
    return new


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phase", default="official")
    p.add_argument("--out", default=None, help="phase directory (default output/live/<phase>)")
    p.add_argument("--test", action="store_true", help="send one test message and stop")
    args = p.parse_args(argv)
    topic = setting("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC is not set (environment or .env)", file=sys.stderr)
        return 2
    post = lambda m: send(m, topic)  # noqa: E731
    now = datetime.now(timezone.utc)
    if args.test:
        post({"title": "icaif alert test", "body": f"from {os.uname().nodename} at {now:%F %T} UTC",
              "priority": "default", "tags": "bell"})
        print("sent")
        return 0
    phase_dir = Path(args.out) if args.out else LIVE / args.phase
    new = watch(args.phase, phase_dir, LIVE / "alerts" / f"{args.phase}.json", now,
                runner_alive(args.phase), post)
    for k in new:
        print(f"{now:%F %T} sent {k}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
