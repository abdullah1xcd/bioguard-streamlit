import sys, tempfile, pathlib, inspect, traceback, importlib, types, contextlib
sys.path.insert(0, ".")
class _Raises:
    def __init__(s, exc): s.exc = exc
    def __enter__(s): return s
    def __exit__(s, t, v, tb):
        if t is None: raise AssertionError(f"DID NOT RAISE {s.exc}")
        return issubclass(t, s.exc)
pt = types.ModuleType("pytest")
pt.raises = lambda e: _Raises(e)
pt.fixture = lambda f: f
sys.modules["pytest"] = pt
from src.platform import Platform
mod = importlib.import_module("tests.test_core_security_automation")
ok = bad = 0
for name, fn in inspect.getmembers(mod, inspect.isfunction):
    if not name.startswith("test_"): continue
    tmp = pathlib.Path(tempfile.mkdtemp())
    args = {}
    for p in inspect.signature(fn).parameters:
        if p == "tmp_path": args[p] = tmp
        elif p == "plat": args[p] = Platform(db_path=tmp/"t.db", bundle_path=tmp/"none.joblib", dataset_dir=tmp/"ds", backup_dir=tmp/"bk", reports_dir=tmp/"rep")
    try:
        fn(**args); ok += 1; print("PASS", name)
    except Exception:
        bad += 1; print("FAIL", name); traceback.print_exc(limit=4)
print(f"\n{ok} passed, {bad} failed")
