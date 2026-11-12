"""M4 fixes: PetShop-root resolution with diagnostic errors, the E6/E8 rule of the queue, and the run_all.sh preflight."""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from fake_petshop import make_fake_petshop                                           # noqa: E402
from pgm.petshop import load_petshop, resolve_root                                   # noqa: E402


def _env(**kw):
    class Ctx:
        def __enter__(s):
            s.old = {k: os.environ.get(k) for k in kw}
            for k, v in kw.items():
                (os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v))
        def __exit__(s, *a):
            for k, v in s.old.items():
                (os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v))
    return Ctx()


def test_resolve_root_accepts_dataset_repo_root_tilde_and_relative_paths():
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "repo", "dataset")
        make_fake_petshop(ds, "low_traffic", n=200)
        with _env(PETSHOP_ROOT=ds):
            assert resolve_root() == ds
        with _env(PETSHOP_ROOT=os.path.join(tmp, "repo")):                       # the cloned repo instead of its dataset folder
            assert resolve_root() == ds
        with _env(PETSHOP_ROOT="~/x_repo/dataset", HOME=tmp):                      # ~ is expanded (an unexpanded "~" was a real failure mode)
            os.makedirs(os.path.join(tmp, "x_repo"))
            os.symlink(ds, os.path.join(tmp, "x_repo", "dataset"))
            assert os.path.realpath(resolve_root()) == os.path.realpath(ds) or resolve_root().endswith("x_repo/dataset")
        cwd = os.getcwd()
        try:
            os.chdir(os.path.join(tmp, "repo"))
            with _env(PETSHOP_ROOT="./dataset"):
                assert resolve_root() == ds
        finally:
            os.chdir(cwd)
        assert resolve_root(ds) == ds                                              # explicit argument too
        from pgm.graphs import find_tetrad_jar
        try:
            find_tetrad_jar()
        except Exception:
            return                                                                 # loading the data needs the jar (PAG step)
        assert load_petshop("low_traffic", os.path.join(tmp, "repo")).data.shape[0] == 200


def test_resolve_root_error_says_what_was_tried_and_why():
    with tempfile.TemporaryDirectory() as tmp:
        with _env(PETSHOP_ROOT=None):
            try:
                resolve_root()
                raise AssertionError("expected FileNotFoundError")   # (fallback directories do not exist in CI)
            except FileNotFoundError as e:
                assert "not set" in str(e) and "export" in str(e)
        with _env(PETSHOP_ROOT=os.path.join(tmp, "missing")):
            try:
                resolve_root()
                raise AssertionError("expected FileNotFoundError")
            except FileNotFoundError as e:
                assert "does not exist" in str(e) and "missing" in str(e)
        os.makedirs(os.path.join(tmp, "empty"))
        with _env(PETSHOP_ROOT=os.path.join(tmp, "empty")):
            try:
                resolve_root()
                raise AssertionError("expected FileNotFoundError")
            except FileNotFoundError as e:
                assert "no low_traffic/graph.csv" in str(e)


def test_queue_accepts_the_quiet_group_and_e6_with_e8_but_not_e6_with_main_experiments():
    import run_queue_parallel as rq
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset")
        make_fake_petshop(ds, "low_traffic", n=200)
        base = ["--workers", "1", "--results-dir", os.path.join(tmp, "res"), "--root", ds, "--max-jobs", "0", "--scenarios", "low_traffic"]
        rq.main(["--group", "quiet"] + base)                                       # was rejected by the old "E6 must run alone" rule
        rq.main(["--experiments", "E6", "E8"] + base)
        rq.main(["--experiments", "E6"] + base)
        for bad in (["--experiments", "E3", "E6"], ["--group", "all"]):
            try:
                rq.main(bad + base)
                raise AssertionError("expected the E6 rule to refuse " + str(bad))
            except SystemExit as e:
                assert e.code == 2
        rq.main(["--group", "all", "--list"] + base)                               # listing everything stays allowed


def test_run_all_preflight_fails_early_with_a_reason_and_accepts_a_good_root():
    from pgm.graphs import find_tetrad_jar
    try:
        jar = find_tetrad_jar()
    except Exception as e:
        raise __import__("unittest").SkipTest(str(e))
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset")
        make_fake_petshop(ds, "low_traffic", n=200)
        env = {k: v for k, v in os.environ.items() if k != "PETSHOP_ROOT"}
        env["CAUSAL_CMD_JAR"] = jar
        sh = os.path.join(ROOT, "run_all.sh")
        r = subprocess.run(["bash", sh, "bogus"], env=env, capture_output=True, text=True, cwd=tmp)
        assert r.returncode == 3 and "PETSHOP_ROOT" in r.stderr and "not set" in r.stderr
        env["PETSHOP_ROOT"] = os.path.join(tmp, "nope")
        r = subprocess.run(["bash", sh, "bogus"], env=env, capture_output=True, text=True, cwd=tmp)
        assert r.returncode == 3 and "does not exist" in r.stderr
        env["PETSHOP_ROOT"] = "./dataset"                                          # relative to the caller's directory, caller is not Code/src
        r = subprocess.run(["bash", sh, "bogus"], env=env, capture_output=True, text=True, cwd=tmp)
        assert r.returncode == 2 and ds in r.stdout and "usage" in r.stdout
