"""Stage 2e: the report generator runs on the stored raw results and produces all tables (skipped if no results are present)."""
import glob
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_make_report_runs():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    paths = sorted(glob.glob(os.path.join(ROOT, "results", "ps_2d_main_*.jsonl")))
    if not paths:
        import pytest  # noqa
        raise __import__("unittest").SkipTest("no stage-2d results")
    import make_report
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rep")
        make_report.main(["--jsonl", *paths, "--out", out])
        txt = open(out + ".md").read()
        assert "Table 1" in txt and "Table 5" in txt and "PARTIAL" in txt
        assert os.path.getsize(out + ".tex") > 500
