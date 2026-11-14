"""P1 archives must retain their dependencies and exclude runtime/private evidence."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "cadathlon/common"
PROBLEM = ROOT / "cadathlon/non_agent_pset/p1_benchmark"


def fixture(tmp_path):
    root = tmp_path / "cadathlon"
    problem = root / "non_agent_pset/p1_benchmark"
    (problem / "src/suite").mkdir(parents=True)
    (root / "common").mkdir()
    for name in ("package.py", "support.py", "Makefile.problem"):
        shutil.copy2(COMMON / name, root / "common" / name)
    for name in ("REPORT.md", "cross.patch", "src/Makefile", "src/runner.py"):
        (problem / name).write_text("")
    (problem / "src/suite/suite_config.json").write_text("{}")
    shutil.copy2(PROBLEM / "EXCLUDE", problem / "EXCLUDE")
    (root / "validation/p1_student_v6e8").mkdir(parents=True)
    for name in ("p1_student_suite.json", "p1_student_v6e8/report.json", "p1_legacy_suite.json"):
        (root / "validation" / name).write_text("{}")
    return root, problem


def test_p1_submission_has_complete_common_dependencies(tmp_path):
    root, problem = fixture(tmp_path)
    scratch = problem / "src/suite/run"
    scratch.mkdir()
    (scratch / "secret.py").write_text("private = 1")
    (problem / "src/suite/secret_keys.json").write_text("{}")
    subprocess.run([sys.executable, str(root / "common/package.py"),
                    "submit", str(problem)], check=True)
    with tarfile.open(problem / "submission.tar.gz") as archive:
        names = set(archive.getnames())
    assert {"common/support.py", "common/Makefile.problem", "common/package.py",
            "non_agent_pset/p1_benchmark/src/suite/suite_config.json"} <= names
    assert not any("secret" in name or "/run/" in name for name in names)
    assert not any("transformer" in name or "qwen" in name for name in names)


def test_p1_release_retains_public_calibration_only(tmp_path):
    root, problem = fixture(tmp_path)
    public = problem / "src/suite/calibration/student-baseline/example/report.json"
    public.parent.mkdir(parents=True)
    public.write_text('{"aggregate_only": true}')
    for name in ("src/bench_opt.py.golden", "src/suite/run/private.py",
                 "src/suite/results/C3/evidence/labels.npy"):
        p = problem / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"excluded")
    # The contestant release packages every problem and the complete shared
    # runtime, so give the other problems their real EXCLUDE lists.
    for other in ("agent_pset/p2_bert", "non_agent_pset/p3_cnn",
                  "agent_pset/p4_qwen", "agent_pset/p5_wise"):
        (root / other).mkdir(parents=True)
        shutil.copy2(ROOT / "cadathlon" / other / "EXCLUDE", root / other / "EXCLUDE")
    for path in COMMON.iterdir():
        if path.is_file() and not (root / "common" / path.name).exists():
            shutil.copy2(path, root / "common" / path.name)
    (root / "validation/p3_compact_baseline.json").write_text("{}")
    subprocess.run([sys.executable, str(root / "common/package.py"),
                    "release", str(root)], check=True)
    with tarfile.open(root / "contestant-release.tar.gz") as archive:
        names = set(archive.getnames())
    assert public.relative_to(root).as_posix() in names
    assert {"validation/p1_student_suite.json", "validation/p1_student_v6e8/report.json",
            "validation/p1_legacy_suite.json"} <= names
    assert not any(name.endswith(".golden") or "/run/" in name or "/evidence/" in name
                   for name in names)


def test_default_cross_root_uses_this_checkout():
    env = dict(os.environ, PYTHONPATH="", JAX_PLATFORMS="standalone_no_backend")
    env.pop("CROSS_ROOT", None)
    probe = ("import json,sys; sys.path.insert(0,sys.argv[1]); "
             "from support import cross_root; "
             "print(json.dumps({'root':str(cross_root()),'jax_loaded':'jax' in sys.modules}))")
    result = subprocess.run([sys.executable, "-I", "-c", probe, str(COMMON)],
                            env=env, cwd=PROBLEM / "src", check=True,
                            capture_output=True, text=True)
    assert json.loads(result.stdout) == {"root": str(ROOT), "jax_loaded": False}
