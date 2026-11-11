"""Run-environment facts and per-job resource measurements, attached to every result row (M4 WP0).

A timing or memory number is only interpretable together with the machine it came from, so each job records:
  static  : python / numpy / scipy / scikit-learn versions, CPU model, logical CPUs, host name, BLAS thread variables, code hash
  per job : peak resident memory (``peak_rss_mb``), RSS at the job start, CPU seconds, wall seconds, load average at start and end

Peak memory is read from ``VmHWM`` after resetting it with ``/proc/self/clear_refs`` (Linux), so a long-lived worker process reports the
peak of THIS job rather than of its whole life; elsewhere it falls back to ``ru_maxrss`` (a lifetime maximum, flagged by ``rss_scope``).
"""
import hashlib
import os
import platform
import resource
import socket
import sys
import time
from typing import Dict, Optional

_STATIC: Optional[Dict] = None
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def code_hash(root: Optional[str] = None) -> str:
    """Short hash of the source files that decide the results (pgm, skeleton, scripts), so a result line names its code version."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1()
    for sub in ("pgm", "skeleton", "scripts"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                h.update(name.encode())
                with open(os.path.join(d, name), "rb") as f:
                    h.update(f.read())
    return h.hexdigest()[:12]


def static_env() -> Dict:
    global _STATIC
    if _STATIC is None:
        import numpy
        import scipy
        try:
            import sklearn
            skv = sklearn.__version__
        except Exception:                                    # noqa: BLE001
            skv = None
        _STATIC = dict(py_version=platform.python_version(), numpy_version=numpy.__version__, scipy_version=scipy.__version__,
                       sklearn_version=skv, cpu_model=_cpu_model(), logical_cpus=os.cpu_count(), host=socket.gethostname(),
                       blas_threads=",".join(f"{v.split('_')[0]}={os.environ.get(v, '?')}" for v in THREAD_VARS), code_hash=code_hash())
    return dict(_STATIC)


def rss_mb() -> float:
    """Current resident memory of this process in MB (0.0 if unreadable)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _hwm_mb() -> Optional[float]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def _reset_hwm() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")                                       # 5 = reset the peak-RSS watermark
        return True
    except OSError:
        return False


def _load():
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return float("nan")


class JobMeter:
    """``with JobMeter() as m: ...`` then ``m.fields()``: resource use of the block (one job in one worker process)."""

    def __enter__(self):
        self.reset_ok = _reset_hwm()
        self.rss0 = rss_mb()
        self.load0 = _load()
        self.t0, self.c0 = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        self.cpu = time.process_time() - self.c0
        self.load1 = _load()
        hwm = _hwm_mb()
        if self.reset_ok and hwm is not None:
            self.peak, self.scope = hwm, "job"
        else:
            self.peak, self.scope = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, "process_lifetime"
        return False

    def fields(self) -> Dict:
        out = static_env()
        out.update(peak_rss_mb=round(self.peak, 1), rss_start_mb=round(self.rss0, 1), rss_scope=self.scope,
                   job_cpu_s=round(self.cpu, 2), job_wall_s=round(self.wall, 2),
                   load1_start=round(self.load0, 2), load1_end=round(self.load1, 2))
        return out


def timing_clean(load1_end: float, workers: int, margin: float = 1.0) -> bool:
    """True when the 1-minute load average at the end of a job did not exceed the number of our own workers by more than ``margin``:
    nothing else was competing.  (A busy start is invisible in a 1-minute average for jobs shorter than a minute; treat it as a
    screen, not a proof.)"""
    return bool(load1_end == load1_end and load1_end <= workers + margin)
