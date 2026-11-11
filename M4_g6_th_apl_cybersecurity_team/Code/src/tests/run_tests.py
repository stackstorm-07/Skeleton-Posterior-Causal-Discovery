"""Minimal test runner for environments without pytest:  python tests/run_tests.py [-k substring] [--fast]

Collects every ``test_*`` function in tests/test_*.py (they are plain functions, so pytest runs them too).
``--fast`` skips tests marked slow (a ``SLOW = True`` attribute on the function).
"""
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main(argv):
    pat = argv[argv.index("-k") + 1] if "-k" in argv else ""
    fast = "--fast" in argv
    passed = failed = skipped = 0
    for fn in sorted(os.listdir(HERE)):
        if not (fn.startswith("test_") and fn.endswith(".py")):
            continue
        spec = importlib.util.spec_from_file_location(fn[:-3], os.path.join(HERE, fn))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(dir(mod)):
            f = getattr(mod, name)
            if not (name.startswith("test_") and callable(f)) or pat not in f"{fn}::{name}":
                continue
            if fast and getattr(f, "SLOW", False):
                skipped += 1
                continue
            t = time.time()
            try:
                f()
                print(f"PASS {fn}::{name} ({time.time() - t:.1f}s)")
                passed += 1
            except Exception as e:
                if type(e).__name__ == "SkipTest":
                    print(f"SKIP {fn}::{name}: {str(e)[:70]}")
                    skipped += 1
                    continue
                print(f"FAIL {fn}::{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
