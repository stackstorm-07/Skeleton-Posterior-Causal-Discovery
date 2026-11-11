import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from pgm import envinfo as E


def test_static_env_has_the_fields_a_timing_claim_needs():
    s = E.static_env()
    for k in ("py_version", "numpy_version", "sklearn_version", "cpu_model", "logical_cpus", "host", "blas_threads", "code_hash"):
        assert k in s and s[k] not in (None, "")
    assert len(s["code_hash"]) == 12 and s["code_hash"] == E.code_hash()


def test_job_meter_sees_a_big_allocation_and_resets_between_jobs():
    with E.JobMeter() as small:
        time.sleep(0.01)
    with E.JobMeter() as big:
        x = np.ones(30_000_000)                       # 240 MB
        x.sum()
    f_small, f_big = small.fields(), big.fields()
    assert f_big["job_wall_s"] >= 0 and f_big["job_cpu_s"] >= 0
    if f_big["rss_scope"] == "job":                   # Linux: the peak is per job, so the small job after a big one stays small
        assert f_big["peak_rss_mb"] - f_big["rss_start_mb"] > 200
        del x
        with E.JobMeter() as after:
            pass
        assert after.fields()["peak_rss_mb"] < f_big["peak_rss_mb"] - 100


def test_timing_clean_rule():
    assert E.timing_clean(5.2, workers=5) and not E.timing_clean(9.0, workers=5) and not E.timing_clean(float("nan"), workers=5)
