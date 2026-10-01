"""The watchdog kills what hangs: a deadlocked child is a missed round, not a frozen runner."""

import sys
import time

from icaif import watchdog

PY = sys.executable


def test_a_child_that_hangs_without_raising_is_killed_at_its_timeout():
    """The LightGBM/torch deadlock sits at 0% CPU with no error. A timeout inside the same
    process returns, and the deadlocked thread keeps the round waiting past its deadline."""
    t0 = time.monotonic()
    res = watchdog.run([PY, "-c", "import threading; threading.Event().wait()"], timeout=1.0)
    assert res.timed_out and not res.ok and res.returncode is None
    assert time.monotonic() - t0 < 10


def test_a_grandchild_holding_the_hang_dies_with_its_parent(tmp_path):
    """A data loader's worker pool can outlive its parent; killing only the child would
    leave the deadlock running beside the next round."""
    pidfile = tmp_path / "grandchild.pid"
    code = (f"import subprocess, sys, time; "
            f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
            f"open({str(pidfile)!r}, 'w').write(str(p.pid)); time.sleep(600)")
    res = watchdog.run([PY, "-c", code], timeout=2.0)
    assert res.timed_out
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    import os
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        # A zombie (killed, not yet reaped by init) still answers kill(0); check its state.
        state = os.popen(f"ps -o stat= -p {pid}").read().strip()
        if not state or state.startswith("Z"):
            return
        time.sleep(0.1)
    raise AssertionError("the grandchild survived the watchdog")


def test_a_failing_child_is_reported_with_its_error_not_as_a_timeout():
    res = watchdog.run([PY, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"], timeout=10)
    assert not res.ok and not res.timed_out and res.returncode == 3 and "boom" in res.stderr


def test_a_child_that_finishes_keeps_its_output():
    res = watchdog.run([PY, "-c", "print('scored')"], timeout=10, env={"OMP_NUM_THREADS": "1"})
    assert res.ok and "scored" in res.stdout
