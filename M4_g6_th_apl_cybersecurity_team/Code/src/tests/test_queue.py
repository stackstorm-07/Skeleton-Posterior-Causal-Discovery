"""M3 item 8: the parallel queue (scripts/run_queue_parallel.py): plan counts, safety rules, end-to-end run on FAKE PetShop data with
two worker processes, resume, failure logging and retry."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import run_queue_parallel as rq                                                      # noqa: E402


def _lines(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def test_thread_env_is_forced_to_one():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[v] == "1"


def test_plan_job_counts_match_the_plan():
    with tempfile.TemporaryDirectory() as tmp:
        jobs = rq.main(["--group", "all", "--list", "--results-dir", tmp, "--workers", "4"])
        n = {e: sum(j["exp"] == e for j in jobs) for e in rq.ALL_EXPS}
        assert n["E3"] == 4 * 5 * 3                    # scenarios x seeds x {abic, spot_cal, spot}
        assert n["E4"] == 4 * 3 * 4                    # x {nofloor, floor_only, random, oracle}
        assert n["E5"] == 20 * 3 * 2                   # B x scenarios x {abic, spot_cal}
        assert n["E1"] == 4 * 10                       # n values x graphs
        assert n["E2"] == 10 * 8                       # cells x graphs
        assert n["E6"] == 4 * 3 * 2 * 2                # x variants x budgets
        assert n["E10"] == 4 * 3 * 2 and n["E13"] == 4 * 3 * 2
        assert n["E12"] == 4 * 3 and n["E8"] == 4 and n["E7"] == 5
        assert n["E9"] == 4 + 40
        assert len({j["id"] for j in jobs}) == len(jobs)                                  # ids are unique
        cut = rq.main(["--group", "all", "--list", "--cut", "--results-dir", tmp, "--workers", "4"])
        assert sum(j["exp"] == "E5" for j in cut) == 10 * 3 * 2 and sum(j["exp"] == "E1" for j in cut) == 4 * 6
        sub = rq.main(["--experiments", "E2", "--e2-factors", "density", "snr", "--list", "--results-dir", tmp])
        assert len(sub) == 6 * 8
        assert not os.listdir(tmp)                                                         # --list writes nothing


def test_e6_safety_rules():
    for argv in (["--experiments", "E6", "E3", "--workers", "1"], ["--group", "all", "--workers", "1"],
                 ["--experiments", "E6", "--workers", str((os.cpu_count() or 1) + 5)]):
        try:
            rq.main(argv + ["--results-dir", tempfile.gettempdir() + "/_unused_queue"])
            raise AssertionError(f"expected the safety rule to refuse {argv}")
        except SystemExit as e:
            assert e.code == 2


def _need_tetrad():
    from pgm.graphs import find_tetrad_jar
    try:
        find_tetrad_jar()
    except Exception as e:
        raise __import__("unittest").SkipTest(str(e))


def _model():
    import sklearn
    from skeleton.posterior import SkeletonPosterior
    main = os.path.join(ROOT, "models", "skeleton_posterior.pkl")
    try:
        SkeletonPosterior.load(main)
        return main
    except Exception:
        if os.path.exists(os.path.join(ROOT, "models", f"skeleton_posterior_sk{sklearn.__version__}.pkl")):
            return main
        raise __import__("unittest").SkipTest("no loadable posterior")


def test_queue_end_to_end_on_fake_petshop_with_two_workers():
    _need_tetrad()
    model = _model()
    from fake_petshop import make_fake_issues, make_fake_petshop
    with tempfile.TemporaryDirectory() as tmp:
        data, res = os.path.join(tmp, "data"), os.path.join(tmp, "res")
        make_fake_petshop(data, "low_traffic", n=300)
        make_fake_issues(data, "low_traffic", n=300)
        base = ["--root", data, "--model", model, "--results-dir", res, "--workers", "2", "--scenarios", "low_traffic",
                "--seeds", "0", "--B", "2", "--d", "8", "--syn-graphs", "1", "--e1-ns", "200", "--graph-seeds", "0",
                "--pq-n-sim", "3", "--pq-n-cal", "3", "--lbfgs-maxiter", "60", "--max-outer", "4"]
        exps = ["E3", "E5", "E9", "E10", "E7", "E12", "E1"]
        rq.main(base + ["--experiments", *exps])
        f = lambda name: _lines(os.path.join(res, name))
        e3 = f("ps_m3_e3.jsonl")
        assert {r["variant"] for r in e3} == {"abic", "spot_cal", "spot"} and len(e3) == 3
        assert all(r["job_id"].startswith("E3|") and r["queue_workers"] == 2 and r["timing_quiet"] is False for r in e3)
        assert all("c_shd_pag" in r and r["scenario"] == "low_traffic" and r["prep"] == "gauss_rank" for r in e3)
        # M4 WP0: every row carries its run environment, its job's peak memory and the load average it ran under
        for r in e3:
            for k in ("peak_rss_mb", "rss_scope", "job_cpu_s", "load1_start", "load1_end", "cpu_model", "sklearn_version", "code_hash", "timing_clean"):
                assert k in r, k
            assert r["peak_rss_mb"] > 0 and r["timing_clean"] is False        # not a quiet-group run
        assert len(f("boot_main.jsonl")) == 2 * 2                                          # B = 2 x {abic, spot_cal}
        assert len(f("fci_petshop.jsonl")) == 4 and len(f("fci_syn_e1.jsonl")) == 4         # 2 algorithms x 2 alphas
        assert {r["variant"] for r in f("ps_m3_e10.jsonl")} == {"abic_dir", "spot_cal_dir"}
        assert all(r["n_est_bidirected"] == 0 for r in f("ps_m3_e10.jsonl"))
        pq = f("post_quality_m3.jsonl")
        assert {r["calibrator"] for r in pq} == {"none", "isotonic", "platt"} and {r["dataset"] for r in pq} >= {"simulated", "low_traffic"}
        assert len(f("rca_m3.jsonl")) == 4 * (2 + 3 * 8)                                    # 4 cases; sources abic, spot_cal, truth
        e1 = f("syn_m3_e1.jsonl")
        assert {r["variant"] for r in e1} == {"abic", "spot_cal"} and all(r["n"] == 200 and r["d"] == 8 for r in e1)
        assert not os.path.exists(os.path.join(res, "queue_failures.jsonl"))
        counts = {n: len(open(os.path.join(res, n)).readlines()) for n in os.listdir(res) if n.endswith(".jsonl")}
        rq.main(base + ["--experiments", *exps])                                           # resume: nothing is re-run
        assert counts == {n: len(open(os.path.join(res, n)).readlines()) for n in counts}


def test_quiet_group_end_to_end_flags_timing_and_writes_inference_rows():
    _need_tetrad()
    model = _model()
    from fake_petshop import make_fake_petshop
    with tempfile.TemporaryDirectory() as tmp:
        data, res = os.path.join(tmp, "data"), os.path.join(tmp, "res")
        make_fake_petshop(data, "low_traffic", n=300)
        base = ["--root", data, "--model", model, "--results-dir", res, "--workers", "1", "--scenarios", "low_traffic",
                "--seeds", "0", "--graph-seeds", "0", "--lbfgs-maxiter", "60", "--max-outer", "4"]
        rq.main(base + ["--experiments", "E8"])
        rows = _lines(os.path.join(res, "inference_m3.jsonl"))
        models = {r["model"] for r in rows}
        assert {"marginal", "saturated", "truth/admg", "truth/dag", "abic/admg", "abic/dag", "spot_cal/admg", "spot_cal/dag"} == models
        assert {r["split"] for r in rows} == {0, 1, 2, 3, 4} and {r["query"] for r in rows} >= {"B"}
        assert all(r["t_fit"] >= 0 and "logpdf" in r and r["n_test"] > 0 for r in rows)
        # time-ordered, no leakage: training windows grow, test windows start right after them
        tr = sorted({(r["split"], r["n_train"]) for r in rows})
        assert [t for _, t in tr] == sorted(t for _, t in tr) and tr[0][1] == 150 and tr[-1][1] == 270
        assert os.path.isdir(os.path.join(res, "rca_graphs"))
        e6 = rq.main(base + ["--experiments", "E6", "--max-jobs", "1"])                    # 1 worker <= cores / 2 only if cores >= 2
        if (os.cpu_count() or 1) >= 2:
            r6 = _lines(os.path.join(res, "ps_m3_e6.jsonl"))
            assert len(r6) == 1 and r6[0]["timing_quiet"] is True and r6[0]["budget"] in (60.0, 120.0)


def test_failed_job_is_logged_not_recorded_and_retried():
    _need_tetrad()
    model = _model()
    from fake_petshop import make_fake_petshop
    with tempfile.TemporaryDirectory() as tmp:
        good, bad, res = os.path.join(tmp, "good"), os.path.join(tmp, "missing"), os.path.join(tmp, "res")
        make_fake_petshop(good, "low_traffic", n=300)
        args = ["--model", model, "--results-dir", res, "--workers", "1", "--scenarios", "low_traffic", "--experiments", "E9",
                "--e1-ns", "200", "--syn-graphs", "1", "--d", "8", "--max-jobs", "1"]
        rq.main(args + ["--root", bad])                                                    # first job = the petshop one: no data -> fails
        fails = _lines(os.path.join(res, "queue_failures.jsonl"))
        assert len(fails) == 1 and fails[0]["job_id"].startswith("E9|petshop") and fails[0]["error"]
        assert not os.path.exists(os.path.join(res, "fci_petshop.jsonl"))                   # nothing recorded as done
        rq.main(args + ["--root", good])                                                   # retried (still the first job) and now succeeds
        assert len(_lines(os.path.join(res, "fci_petshop.jsonl"))) == 4


def test_execute_stamps_every_row_with_environment_and_resources_without_needing_java():
    """The stamping wrapper around a job, tested with a stand-in job so it runs where the Tetrad jar is absent."""
    import numpy as np
    orig = rq._execute_job
    try:
        def fake(job):
            x = np.ones(20_000_000)                    # 160 MB, so the peak is visible
            return [dict(variant="a", v=float(x.sum())), dict(variant="b")]
        rq._execute_job = fake
        rows = rq.execute({"kind": "ps", "spec": {}})
    finally:
        rq._execute_job = orig
    assert len(rows) == 2
    for r in rows:
        for k in ("peak_rss_mb", "rss_scope", "job_cpu_s", "job_wall_s", "load1_start", "load1_end", "cpu_model", "sklearn_version", "code_hash", "blas_threads"):
            assert k in r, k
        assert r["peak_rss_mb"] > 100 and r["blas_threads"].count("=1") == 4          # the queue forces one BLAS thread
    assert rows[0]["variant"] == "a" and rows[1]["variant"] == "b"
