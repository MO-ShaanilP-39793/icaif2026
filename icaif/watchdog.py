"""Run a step in a child process and kill it at a deadline.

The step this exists for is scoring. On the owner's Mac, predicting LightGBM and then
the FastAI net in one process deadlocked: torch's batch_norm waited forever in an
OpenMP barrier, two OpenMP runtimes, 0% CPU and no error. `live.load_predictor` holds
torch to one thread, which cured it there, but a hang in C code cannot be interrupted
from Python in the same process: a thread with a timeout returns, and the deadlocked
thread keeps the process (and the round) waiting. A child process can be killed.

So anything that can hang without raising runs here: the scorer, and each round's
worker under the scheduler. The child is started in its own process group and the
whole group is killed, so a grandchild (a data loader's worker pool) cannot keep the
deadlock alive after its parent is gone.
"""

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

TAIL = 4000  # characters of output kept for the record


@dataclass
class Outcome:
    ok: bool
    returncode: Optional[int]
    timed_out: bool
    elapsed_s: float
    stdout: str
    stderr: str

    def summary(self) -> dict:
        return {"ok": self.ok, "returncode": self.returncode, "timed_out": self.timed_out,
                "elapsed_s": round(self.elapsed_s, 2), "stderr_tail": self.stderr[-600:]}


def run(argv: list[str], timeout: float, env: Optional[dict] = None,
        cwd: Optional[str] = None) -> Outcome:
    """Run `argv`; at `timeout` seconds kill its process group and report a timeout.

    `env` is added to this process's environment, never replacing it: a child
    without PATH or HOME fails in ways that look like the step's own bug.
    """
    full_env = {**os.environ, **(env or {})}
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=full_env, cwd=cwd, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=max(timeout, 0.0))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return Outcome(False, None, True, time.monotonic() - t0, (out or "")[-TAIL:],
                       (err or "")[-TAIL:])
    return Outcome(proc.returncode == 0, proc.returncode, False, time.monotonic() - t0,
                   (out or "")[-TAIL:], (err or "")[-TAIL:])
