"""M3 item 8: one resumable process pool that runs the job lists of the plan's experiments E1-E13.

    export PETSHOP_ROOT=<dataset dir>  CAUSAL_CMD_JAR=<causal-cmd jar>
    python scripts/run_queue_parallel.py --list                                   # what would run, how much is done, core-hours: runs nothing
    python scripts/run_queue_parallel.py --group main --workers 7                 # the overnight pass (E3 E4 E5 E1 E9 E10 E2 E12 E7 E13)
    python scripts/run_queue_parallel.py --group quiet --workers 3                # afterwards, on a quiet machine (E6, E8)
    python scripts/run_queue_parallel.py --experiments E3 E4 --workers 7          # any subset
    wc -l results/*.jsonl                                                         # progress

Rules this script enforces (plan section 5)
* One thread per job: OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS are forced to 1 before numpy is imported (and are inherited by the workers).
* The PARENT is the only writer.  A worker returns the rows of its job; the parent appends all of them with a single ``write`` call to the
  experiment's file, so there are no write races and a killed run never leaves half a job.  Every row carries ``job_id``, ``exp`` and
  ``queue_workers`` / ``queue_cores`` / ``timing_quiet`` (was the machine quiet enough for timings to be compared?).
* Resumable: a job whose ``job_id`` already appears in its file is skipped.  A job that raised is NOT recorded as done; it is logged in
  ``results/queue_failures.jsonl`` with its traceback and is retried at the next start.  Ctrl-C keeps everything finished so far.
* E6 (equal wall-clock budget) must run alone with at most half of the cores busy; the script refuses otherwise (override: --force).
  E8 is also best run in that quiet pass; its rows are flagged with ``timing_quiet`` either way.
* Order: the plan's order inside a group; inside an experiment the most expensive jobs first (shorter tail).
* Pool start method is 'spawn' (forking after OpenMP has run in the parent can deadlock).
* ``--workers`` defaults to cpu_count // 2 - 1, which assumes hyper-threading; pass physical cores - 1 yourself if you know it.

What each experiment runs (job = unit of work; rows go to results/<file>)
  E1  synthetic sample-size sweep (spot_vs_abic_synthetic --design e1)         job = (n, graph)         syn_m3_e1.jsonl
  E2  synthetic robustness (--design e2)                                         job = (cell, graph)      syn_m3_e2.jsonl
  E3  PetShop main: abic spot_cal spot x scenarios x 5 seeds                     job = (scenario, seed, variant)   ps_m3_e3.jsonl
  E4  PetShop ablations: spot_nofloor spot_floor_only spot_random spot_oracle    same                     ps_m3_e4.jsonl
  E5  block bootstrap, abic vs spot_cal, B resamples x 3 scenarios               job = (scenario, b, variant)      boot_main.jsonl
  E6  equal budget 60 s / 120 s, abic vs spot_cal                                job = (scenario, seed, variant, budget)   ps_m3_e6.jsonl
  E7  skeleton-posterior probability quality (+ isotonic / Platt)                job = dataset            post_quality_m3.jsonl
  E8  inference validation, 5 rolling time-ordered splits                        job = (scenario, graph seed)      inference_m3.jsonl
  E9  FCI / RFCI, alpha 0.01 / 0.05: PetShop and the E1 synthetic graphs         job = scenario / graph   fci_petshop.jsonl, fci_syn_e1.jsonl
  E10 directed-only learner: abic_dir spot_cal_dir                               job = (scenario, seed, variant)   ps_m3_e10.jsonl
  E12 root-cause ranking: anomaly / corr / dag / admg x ancestor modes           job = (scenario, graph seed)      rca_m3.jsonl
  E13 preprocessing sensitivity (raw_centered), abic vs spot_cal                 job = (scenario, seed, variant)   ps_m3_e13.jsonl
  E11 (error analysis) runs nothing: it reads the files above.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # forced, not setdefault: parallel jobs must not share BLAS threads, or timings mean nothing

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

GROUPS = {"main": ["E3", "E4", "E5", "E1", "E9", "E10", "E2", "E12", "E7", "E13"], "quiet": ["E6", "E8"]}
GROUPS["all"] = GROUPS["main"] + GROUPS["quiet"]
ALL_EXPS = GROUPS["all"]
QUIET = set(GROUPS["quiet"])
FILES = dict(E1="syn_m3_e1.jsonl", E2="syn_m3_e2.jsonl", E3="ps_m3_e3.jsonl", E4="ps_m3_e4.jsonl", E5="boot_main.jsonl",
             E6="ps_m3_e6.jsonl", E7="post_quality_m3.jsonl", E8="inference_m3.jsonl", E10="ps_m3_e10.jsonl",
             E12="rca_m3.jsonl", E13="ps_m3_e13.jsonl", E9_ps="fci_petshop.jsonl", E9_syn="fci_syn_e1.jsonl")
PLAN_CORE_HOURS = dict(E1=1.5, E2=2.0, E3=0.8, E4=1.0, E5=2.7, E6=0.5, E7=0.3, E8=0.5, E9=0.5, E10=0.5, E12=0.7, E13=1.5)
SCENARIOS = ["low_traffic", "high_traffic", "temporal_traffic1", "temporal_traffic2"]


# ================================================================================================ job lists (parent side)
def _hint(variant, scenario=None):
    """Rough single-core seconds, only used to order jobs and for the first ETA (M2 PetShop timings)."""
    base = 110.0 if variant.startswith("abic") else 50.0
    if variant.endswith("_dir"):
        base *= 0.4
    return base * (1.6 if scenario == "high_traffic" else 1.0)


def _common(a):
    return dict(root=a.root, model=a.model, lbfgs_maxiter=a.lbfgs_maxiter, max_outer=a.max_outer, max_rows=a.max_rows, restarts=1,
                results_dir=a.results_dir)


def _job(exp, jid, kind, spec, hint, file=None):
    return dict(id=jid, exp=exp, kind=kind, spec=spec, hint=float(hint), file=file or FILES[exp])


def _ps_jobs(a, exp, variants, seeds, prep="gauss_rank", budgets=(None,)):
    out = []
    for scn in a.scenarios:
        for seed in seeds:
            for v in variants:
                for b in budgets:
                    jid = f"{exp}|{scn}|{prep}|seed{seed}|{v}|budget{b}"
                    spec = dict(_common(a), scenario=scn, prep=prep, seed=seed, variant=v, budget=b, tag=exp)
                    out.append(_job(exp, jid, "ps", spec, (b if b else _hint(v, scn))))
    return out


def jobs_E1(a):
    from spot_vs_abic_synthetic import design_conditions
    conds = design_conditions("e1", d=a.d, graphs=a.syn_graphs or (6 if a.cut else None))
    if a.e1_ns:
        conds = [c for c in conds if c["n"] in a.e1_ns]
    return [_job("E1", f"E1|d{c['d']}|n{c['n']}|seed{s}|{'+'.join(c['variants'])}", "syn",
                 dict(_common(a), cond=c, seed=s, variants=c["variants"], tag="E1", design="e1"),
                 sum(_hint(v) for v in c["variants"]) * (c["n"] / 1000.0) ** 0.5)
            for c in conds for s in c["seeds"]]


def jobs_E2(a):
    from spot_vs_abic_synthetic import design_conditions
    conds = design_conditions("e2", d=a.d, graphs=a.syn_graphs)
    if a.e2_factors:
        keep = set(a.e2_factors)
        conds = [c for c in conds if ("density" in keep and c["density"] is not None) or ("snr" in keep and c["snr"] is not None)
                 or ("ar" in keep and c["ar"]) or ("missing" in keep and c["missing"])]
    out = []
    for c in conds:
        cell = f"dens{c['density']}|snr{c['snr']}|ar{c['ar']}|mis{c['missing']}"
        for s in c["seeds"]:
            out.append(_job("E2", f"E2|d{c['d']}|n{c['n']}|{cell}|seed{s}|{'+'.join(c['variants'])}", "syn",
                            dict(_common(a), cond=c, seed=s, variants=c["variants"], tag="E2", design="e2"),
                            sum(_hint(v) for v in c["variants"]) * (1.8 if c["density"] and c["density"] >= 3 else 1.0)))
    return out


def jobs_E3(a):
    return _ps_jobs(a, "E3", ["abic", "spot_cal", "spot"], a.seeds or range(5))


def jobs_E4(a):
    return _ps_jobs(a, "E4", ["spot_nofloor", "spot_floor_only", "spot_random", "spot_oracle"], a.seeds or range(3))


def jobs_E6(a):
    return _ps_jobs(a, "E6", ["abic", "spot_cal"], a.seeds or range(3), budgets=(60.0, 120.0))


def jobs_E10(a):
    return _ps_jobs(a, "E10", ["abic_dir", "spot_cal_dir"], a.seeds or range(3))


def jobs_E13(a):
    return _ps_jobs(a, "E13", ["abic", "spot_cal"], a.seeds or range(3), prep="raw_centered")


def jobs_E5(a):
    scns = [s for s in a.scenarios if s in ("low_traffic", "high_traffic", "temporal_traffic1")] or list(a.scenarios)
    B = a.B if a.B else (10 if a.cut else 20)
    return [_job("E5", f"E5|{s}|b{b}|{v}", "boot",
                 dict(_common(a), scenario=s, b=b, variant=v, prep="gauss_rank", budget=None, block_len=None, boot_seed=2026),
                 _hint(v, s))
            for b in range(B) for s in scns for v in ("abic", "spot_cal")]


def jobs_E7(a):
    spec = dict(_common(a), n_sim=a.pq_n_sim, n_cal=a.pq_n_cal)
    out = [_job("E7", "E7|simulated", "pq", dict(spec, dataset="simulated"), 60)]
    out += [_job("E7", f"E7|{s}", "pq", dict(spec, dataset=s), 30) for s in a.scenarios]
    return out


def jobs_E8(a):
    return [_job("E8", f"E8|{s}|graphseed{g}", "inf", dict(_common(a), scenario=s, graph_seed=g, sources=["truth", "abic", "spot_cal"]), 300)
            for s in a.scenarios for g in (a.graph_seeds or [0])]


def jobs_E9(a):
    from spot_vs_abic_synthetic import design_conditions
    algs = [(alg, al) for alg in ("fci", "rfci") for al in (0.01, 0.05)]
    out = [_job("E9", f"E9|petshop|{s}|gauss_rank", "fci_ps", dict(_common(a), scenario=s, prep="gauss_rank", algs=algs, timeout=900.0),
                30, file=FILES["E9_ps"]) for s in a.scenarios]
    conds = design_conditions("e1", d=a.d, graphs=a.syn_graphs or (6 if a.cut else None))
    if a.e1_ns:
        conds = [c for c in conds if c["n"] in a.e1_ns]
    out += [_job("E9", f"E9|syn|d{c['d']}|n{c['n']}|seed{s}", "fci_syn", dict(_common(a), cond=c, seed=s, algs=algs, timeout=900.0), 10,
                 file=FILES["E9_syn"]) for c in conds for s in c["seeds"]]
    return out


def jobs_E12(a):
    out = []
    for s in a.scenarios:
        for g in (a.graph_seeds or [0, 1, 2]):
            sources = ["abic", "spot_cal"] + (["truth"] if g == (a.graph_seeds or [0, 1, 2])[0] else [])    # the truth graph is seed-free: once
            out.append(_job("E12", f"E12|{s}|graphseed{g}", "rca", dict(_common(a), scenario=s, graph_seed=g, sources=sources), 400))
    return out


BUILDERS = dict(E1=jobs_E1, E2=jobs_E2, E3=jobs_E3, E4=jobs_E4, E5=jobs_E5, E6=jobs_E6, E7=jobs_E7, E8=jobs_E8, E9=jobs_E9,
                E10=jobs_E10, E12=jobs_E12, E13=jobs_E13)


# ================================================================================================ workers
_CACHE = {}


def _posterior(model):
    if "post" not in _CACHE:
        from skeleton.posterior import SkeletonPosterior
        _CACHE["post"] = SkeletonPosterior.load_or_retrain(model, verbose=False)
    return _CACHE["post"]


def _petshop(root, scn):
    if ("ps", scn) not in _CACHE:
        from pgm.petshop import default_root, load_petshop
        _CACHE[("ps", scn)] = load_petshop(scn, root or default_root())
    return _CACHE[("ps", scn)]


def _dcd(spec, seed):
    from pgm.dcd import DCDConfig
    cfg = DCDConfig(num_restarts=spec["restarts"], seed=seed)
    over = {k: v for k, v in (("lbfgs_maxiter", spec["lbfgs_maxiter"]), ("max_outer", spec["max_outer"])) if v}
    return DCDConfig(**{**cfg.__dict__, **over}) if over else cfg


def _ps_context(spec):
    key = ("ctx", spec["scenario"], spec["prep"], spec["max_rows"])
    if key not in _CACHE:
        from run_petshop import petshop_context
        ps = _petshop(spec["root"], spec["scenario"])
        _CACHE[key] = (ps,) + petshop_context(ps, spec["prep"], _posterior(spec["model"]), spec["max_rows"])
    return _CACHE[key]


def execute(job):
    """Runs in a worker process.  Returns the list of result rows of the job."""
    k, spec = job["kind"], job["spec"]
    if k == "ps":
        from run_petshop import petshop_row
        ps, X, names, skel, post_res, pm, pinned, = _ps_context(spec)
        row, _ = petshop_row(ps, X, names, skel, post_res, pm, pinned, spec["scenario"], spec["prep"], spec["seed"], spec["variant"],
                             _dcd(spec, spec["seed"]), spec["budget"], spec["restarts"], spec["tag"])
        return [row]
    if k == "syn":
        from spot_vs_abic_synthetic import run_graph_job
        return run_graph_job(spec["cond"], spec["seed"], spec["variants"], _posterior(spec["model"]), _dcd(spec, 0), None,
                             spec["restarts"], spec["tag"], spec["design"])
    if k == "boot":
        import run_bootstrap
        _CACHE["post"] = _posterior(spec["model"])
        run_bootstrap._CACHE["post"] = _CACHE["post"]
        s = dict(spec, lbfgs_maxiter=spec["lbfgs_maxiter"], max_outer=spec["max_outer"])
        return [run_bootstrap.run_job(s)]
    if k == "fci_ps":
        import run_fci
        from pgm.petshop import preprocess
        ps = _petshop(spec["root"], spec["scenario"])
        data = ps.data[:spec["max_rows"]] if spec["max_rows"] else ps.data
        return run_fci.fci_petshop_rows(ps, preprocess(data, spec["prep"]), spec["scenario"], spec["prep"], [tuple(x) for x in spec["algs"]],
                                        spec["timeout"], -1, "E9")
    if k == "fci_syn":
        import run_fci
        return run_fci.fci_synthetic_rows(spec["cond"], spec["seed"], [tuple(x) for x in spec["algs"]], spec["timeout"], -1, "e1", "E9")
    if k == "pq":
        import posterior_quality as pq
        post = _posterior(spec["model"])
        cals = pq.fit_calibrators(post, spec["n_cal"])
        if spec["dataset"] == "simulated":
            return pq.simulated_rows(post, cals, spec["n_sim"])
        return pq.petshop_rows(post, cals, spec["dataset"], spec["root"], max_rows=spec["max_rows"])
    if k == "rca":
        import run_rca
        tmpdir = os.path.join(spec["results_dir"], "_queue_tmp")
        os.makedirs(tmpdir, exist_ok=True)
        tmp = os.path.join(tmpdir, f"rca_{os.getpid()}_{int(time.time() * 1000)}.jsonl")
        args = ["--scenarios", spec["scenario"], "--sources", *spec["sources"], "--graph-seeds", str(spec["graph_seed"]),
                "--model", spec["model"], "--out", tmp, "--graphs-dir", os.path.join(spec["results_dir"], "rca_graphs")]
        for flag, key in (("--root", "root"), ("--max-rows", "max_rows"), ("--lbfgs-maxiter", "lbfgs_maxiter"), ("--max-outer", "max_outer")):
            if spec[key]:
                args += [flag, str(spec[key])]
        try:
            rows = run_rca.main(args)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return rows
    if k == "inf":
        return _execute_inference(spec)
    raise ValueError(f"unknown job kind {k}")


def _execute_inference(spec):
    """E8: make sure the graphs exist (learn them the way run_rca does, same cache), then the rolling time-ordered evaluation."""
    import argparse as ap
    import run_rca
    import validate_inference as vi
    from pgm.rca import masks_of
    scn = spec["scenario"]
    ps, X, names, skel, post_res, pm, pinned = _ps_context(dict(spec, prep="gauss_rank"))
    graphs_dir = os.path.join(spec["results_dir"], "rca_graphs")          # the same cache run_rca.py writes (E12 and E8 share it)
    graphs, t_opt = {}, {}
    for s in spec["sources"]:
        if s == "truth":
            graphs["truth"] = masks_of(ps.mag, names)
            continue
        a = ap.Namespace(graphs_dir=graphs_dir, graph_seed=spec["graph_seed"], relearn=False, restarts=1, prep="gauss_rank",
                         lbfgs_maxiter=spec["lbfgs_maxiter"], max_outer=spec["max_outer"])
        D, B, meta = run_rca.get_graph(s, scn, ps, X, names, _posterior(spec["model"]), post_res, skel, a)
        graphs[s] = (D, B)
        t_opt[s] = float(meta.get("t_optimize", 0.0))
    sink = names.index("PetSite") if "PetSite" in names else 0
    raw = ps.data[:spec["max_rows"]] if spec["max_rows"] else ps.data
    rows = vi.rolling_rows(raw, names, graphs, t_opt, sink)
    for r in rows:
        r.update(scenario=scn, graph_seed=spec["graph_seed"], sink=names[sink], n=int(len(raw)), d=len(names))
    return rows


# ================================================================================================ parent side
def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def append_rows(path, rows):
    """All rows of one job in ONE write() on an O_APPEND descriptor: a kill can never leave half a job in the file."""
    blob = "".join(json.dumps(r, default=_jsonable) + "\n" for r in rows).encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(blob)
        while view:                                       # a regular-file write is complete in practice; the loop is a safeguard
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                try:
                    jid = json.loads(line).get("job_id")
                except ValueError:
                    continue                                 # a torn last line (power loss): ignored, the job is retried
                if jid:
                    ids.add(jid)
    return ids


def build_jobs(a, results):
    jobs = []
    for exp in a.experiments:
        for j in BUILDERS[exp](a):
            j["path"] = os.path.join(results, j["file"])
            jobs.append(j)
    return jobs


def fmt(sec):
    return f"{sec / 60:5.1f} min" if sec < 5400 else f"{sec / 3600:5.2f} h"


def summarize_plan(jobs, done_by_file, a):
    print(f"{'exp':5s} {'jobs':>6s} {'done':>6s} {'todo':>6s} {'est core-h todo':>16s}   {'plan core-h':>11s}")
    tot = 0.0
    for exp in a.experiments:
        js = [j for j in jobs if j["exp"] == exp]
        todo = [j for j in js if j["id"] not in done_by_file[j["path"]]]
        h = sum(j["hint"] for j in todo) / 3600
        tot += h
        print(f"{exp:5s} {len(js):6d} {len(js) - len(todo):6d} {len(todo):6d} {h:16.2f}   {PLAN_CORE_HOURS.get(exp, 0):11.1f}")
    print(f"total estimated core-hours still to do: {tot:.1f}  ->  about {tot * 1.2 / max(a.workers, 1):.1f} h wall clock with {a.workers} workers (+20 % overhead)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--group", choices=list(GROUPS), default=None, help="main = overnight pass, quiet = E6 + E8, all")
    g.add_argument("--experiments", nargs="+", choices=ALL_EXPS, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2 - 1))
    ap.add_argument("--root", default=None, help="PetShop `dataset` dir (else $PETSHOP_ROOT)")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "skeleton_posterior.pkl"))
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--list", action="store_true", help="print the plan (jobs, done, estimated core-hours) and exit")
    ap.add_argument("--max-jobs", type=int, default=None, help="run at most this many jobs, then stop (smoke test)")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--force", action="store_true", help="override the E6 safety rules (alone, <= half the cores)")
    ap.add_argument("--cut", action="store_true", help="the plan's cut rule: E5 B = 10, E1 and E9 6 graphs per n")
    # sizes (defaults = the plan)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="optimizer seeds of the PetShop experiments (default E3 0-4, E4/E6/E10/E13 0-2)")
    ap.add_argument("--graph-seeds", type=int, nargs="+", default=None, help="E12 default 0 1 2, E8 default 0")
    ap.add_argument("--B", type=int, default=None, help="E5 resamples (default 20, 10 with --cut)")
    ap.add_argument("--d", type=int, default=25, help="synthetic graph size")
    ap.add_argument("--syn-graphs", type=int, default=None, help="graphs per synthetic cell (default E1 10, E2 8)")
    ap.add_argument("--e1-ns", type=int, nargs="+", default=None, help="restrict E1/E9-synthetic to these sample sizes")
    ap.add_argument("--e2-factors", nargs="+", choices=["density", "snr", "ar", "missing"], default=None, help="E2 factors to keep (cut rule: density snr)")
    ap.add_argument("--pq-n-sim", type=int, default=60)
    ap.add_argument("--pq-n-cal", type=int, default=40)
    # tests only
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--lbfgs-maxiter", type=int, default=None)
    ap.add_argument("--max-outer", type=int, default=None)
    a = ap.parse_args(argv)

    a.experiments = a.experiments or GROUPS[a.group or "main"]
    cores = os.cpu_count() or 1
    if "E6" in a.experiments and not a.force and not a.list:
        if set(a.experiments) != {"E6"}:
            ap.error("E6 (equal wall-clock budget) must run alone, after the main pass: use --group quiet or --experiments E6 "
                     "(or --force). --group all is for --list only.")
        if a.workers > max(1, cores // 2):
            ap.error(f"E6 must run with at most half the cores busy: --workers <= {max(1, cores // 2)} (cpu_count = {cores}; or --force)")
    os.makedirs(a.results_dir, exist_ok=True)
    jobs = build_jobs(a, a.results_dir)
    done_by_file = {p: done_ids(p) for p in {j["path"] for j in jobs}}
    summarize_plan(jobs, done_by_file, a)
    if a.list:
        return jobs
    order = {e: i for i, e in enumerate(a.experiments)}
    todo = sorted([j for j in jobs if j["id"] not in done_by_file[j["path"]]], key=lambda j: (order[j["exp"]], -j["hint"]))
    if a.max_jobs is not None:
        todo = todo[:a.max_jobs]
    if not todo:
        print("nothing to do.")
        return jobs
    if any(j["kind"] in ("ps", "boot", "pq", "inf", "rca", "fci_ps") for j in todo) or any(j["kind"] in ("syn",) for j in todo):
        from skeleton.posterior import SkeletonPosterior
        SkeletonPosterior.load_or_retrain(a.model)             # once in the parent: workers never retrain concurrently
    quiet = set(a.experiments) <= QUIET and a.workers <= max(1, cores // 2)
    print(f"running {len(todo)} jobs with {a.workers} workers (cpu_count {cores}); timing_quiet = {quiet}", flush=True)
    meta = dict(queue_workers=a.workers, queue_cores=cores, timing_quiet=bool(quiet))

    failures = os.path.join(a.results_dir, "queue_failures.jsonl")
    t0 = time.time()
    obs = {}                                                   # kind -> list of observed job wall seconds
    remaining = {j["id"]: j for j in todo}
    n_ok = n_fail = 0

    def eta():
        est = 0.0
        for j in remaining.values():
            o = obs.get(j["kind"])
            est += (sum(o) / len(o)) if o else j["hint"]
        return est / max(1, min(a.workers, len(remaining)))

    queue = list(todo)
    inflight = {}
    ex = ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"))
    try:
        while queue or inflight:
            while queue and len(inflight) < a.workers * 2:
                j = queue.pop(0)
                inflight[ex.submit(execute, j)] = (j, time.time())
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                j, ts = inflight.pop(fut)
                remaining.pop(j["id"], None)
                try:
                    rows = fut.result()
                except BaseException as e:                     # noqa: BLE001  (a job failure must not stop the queue)
                    n_fail += 1
                    with open(failures, "a") as f:
                        f.write(json.dumps(dict(job_id=j["id"], exp=j["exp"], t=time.strftime("%F %T"),
                                                error="".join(traceback.format_exception(e))[-3000:])) + "\n")
                    print(f"[FAILED] {j['id']}: {type(e).__name__}: {str(e)[:160]}  (logged in {failures}; retried at the next start)", flush=True)
                    if a.stop_on_error:
                        raise
                    continue
                wall = time.time() - ts
                obs.setdefault(j["kind"], []).append(wall)
                for r in rows:
                    r.update(job_id=j["id"], exp=j["exp"], job_wall=round(wall, 2), job_rows=len(rows), **meta)
                append_rows(j["path"], rows)
                n_ok += 1
                lead = rows[0] if rows else {}
                hint = f"skel {lead['c_skeleton_f1']:.3f}" if "c_skeleton_f1" in lead else f"{len(rows)} rows"
                print(f"[{n_ok + n_fail}/{len(todo)}] {j['id']}  {hint}  {wall:6.1f}s | elapsed {fmt(time.time() - t0)}, ETA {fmt(eta())}"
                      f"{'' if not n_fail else f', {n_fail} failed'}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted: finished jobs are saved, running ones are lost; restart the same command to continue.", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        return jobs
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print(f"done: {n_ok} ok, {n_fail} failed in {fmt(time.time() - t0)}" + ("  -> see results/queue_failures.jsonl" if n_fail else ""))
    return jobs


if __name__ == "__main__":
    main()
