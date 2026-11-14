"""Submission archives retain the paths needed by shared checks and suite stages."""
from pathlib import Path
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile

import pytest


TRANSFORMER_RUNTIME = (
    "transformer_runner.py", "encrypted_polynomial.py", "encrypted_transformer.py",
    "encrypted_reference.py", "encrypted_refresh.py", "encrypted_rotation.py",
    "encrypted_compilation.py", "encrypted_primitives.py", "encrypted_ntt.py",
    "encrypted_packed_qwen.py", "encrypted_packed_transformer.py",
    "encrypted_bounded_arithmetic.py", "encrypted_bounded_transformer.py",
    "encrypted_streaming_linear.py", "encrypted_chained_streaming_linear.py",
    "encrypted_integer_affine.py", "encrypted_scaled_newton.py",
)


# P1 imports only the security gate, so its archive carries just the shared
# tooling; transformer problems such as P2 carry the complete shared runtime.
@pytest.mark.parametrize("problem_path, full_runtime", [
    ("non_agent_pset/p1_benchmark", False),
    ("agent_pset/p2_bert", True),
])
def test_submit_includes_shared_security_gate_and_suite_configuration(tmp_path, problem_path, full_runtime):
    root = tmp_path / "cadathlon"
    problem = root / problem_path
    suite = problem / "src" / "suite"
    suite.mkdir(parents=True)
    common = root / "common"
    common.mkdir()
    for name in ("REPORT.md", "cross.patch"):
        (problem / name).write_text("")
    (problem / "src" / "Makefile").write_text("")
    (problem / "src" / "runner.py").write_text("")
    (suite / "suite_config.json").write_text("{}")
    (suite / "run").mkdir()
    (suite / "run" / "active.json").write_text("{}")
    (suite / "run" / "secrets.py").write_text("private = 1")
    (suite / "secret_keys.json").write_text("{}")
    (suite / "private.npy").write_bytes(b"private")
    for name in ("support.py", "Makefile.problem", "package.py") + TRANSFORMER_RUNTIME:
        (common / name).write_text("")
    script = Path(__file__).parents[1] / "common" / "package.py"
    subprocess.run([sys.executable, str(script), "submit", str(problem)], check=True)
    with tarfile.open(problem / "submission.tar.gz") as archive:
        names = set(archive.getnames())
    assert {"common/support.py", "common/Makefile.problem", "common/package.py"} <= names
    transformer = {f"common/{name}" for name in TRANSFORMER_RUNTIME}
    if full_runtime:
        assert transformer <= names
    else:
        assert not transformer & names
    assert f"{problem_path}/src/runner.py" in names
    assert f"{problem_path}/src/suite/suite_config.json" in names
    assert not any("/run/" in name for name in names)
    assert not any("secret" in name or name.endswith(".npy") for name in names)


def test_missing_shared_runtime_fails_before_creating_an_archive(tmp_path):
    root = tmp_path / "cadathlon"
    problem = root / "agent_pset" / "p4_qwen"
    (problem / "src").mkdir(parents=True)
    for name in ("REPORT.md", "cross.patch"):
        (problem / name).write_text("")
    script = Path(__file__).parents[1] / "common" / "package.py"
    result = subprocess.run([sys.executable, str(script), "submit", str(problem)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "Missing declared archive dependency" in result.stderr
    assert not (problem / "submission.tar.gz").exists()


def test_shared_runtime_manifest_contains_local_import_dependencies():
    """A helper added inside a lazy method must also reach the archive."""
    common = Path(__file__).parents[1] / "common"
    spec = importlib.util.spec_from_file_location("package_manifest_test", common / "package.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    shipped = set(package.COMMON_RUNTIME_FILES)
    for filename in shipped:
        if not filename.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((common / filename).read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom)
                       else [name.name for name in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                dependency = module.split(".")[0] + ".py" if module else None
                if dependency and (common / dependency).is_file():
                    assert dependency in shipped, f"{filename} imports unshipped {dependency}"


def test_partial_submission_runtimes_contain_their_import_closure():
    """P1's and P5's declared runtimes already hold every shared module they import."""
    cadathlon = Path(__file__).parents[1]
    common = cadathlon / "common"
    spec = importlib.util.spec_from_file_location("package_subset_test", common / "package.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    problems = {path.name: path for path in cadathlon.glob("*_pset/*") if path.is_dir()}
    assert set(package.SUBMISSION_RUNTIME) <= set(problems)
    for name, (runtime, _kind) in package.SUBMISSION_RUNTIME.items():
        # The same files `package.py submit` scans: its sources and the declared modules.
        scanned = package.submission_sources(problems[name] / "src")
        scanned += [common / module for module in runtime if module.endswith(".py")]
        missing = package.import_closure(scanned, common) - set(runtime)
        assert not missing, f"{name} submission imports unshipped {sorted(missing)}"


def test_release_runtime_contains_every_problems_shared_imports():
    """The release ships COMMON_RUNTIME_FILES only, so it must cover every problem's imports."""
    cadathlon = Path(__file__).parents[1]
    common = cadathlon / "common"
    spec = importlib.util.spec_from_file_location("package_release_test", common / "package.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    for problem in sorted(cadathlon.glob("*_pset/*/src")):
        sources = package.submission_sources(problem)
        missing = package.import_closure(sources, common) - set(package.COMMON_RUNTIME_FILES)
        assert not missing, f"{problem.parent.name} imports {sorted(missing)}, absent from the release"


def test_submit_ships_shared_modules_that_contestant_code_adds(tmp_path):
    """A new shared import in a narrowed submission reaches its archive."""
    root = tmp_path / "cadathlon"
    problem = root / "non_agent_pset" / "p1_benchmark"
    (problem / "src" / "suite").mkdir(parents=True)
    common = root / "common"
    common.mkdir()
    for name in ("REPORT.md", "cross.patch"):
        (problem / name).write_text("")
    (problem / "src" / "Makefile").write_text("")
    (problem / "src" / "suite" / "suite_config.json").write_text("{}")
    # Saved with a UTF-8 byte-order mark, which Python accepts.
    (problem / "src" / "bench_opt.py").write_text("\ufeffimport os\nfrom encrypted_polynomial import X\n",
                                                  encoding="utf-8")
    for name in ("support.py", "Makefile.problem", "package.py") + TRANSFORMER_RUNTIME:
        (common / name).write_text("")
    (common / "encrypted_polynomial.py").write_text("import numpy\nfrom encrypted_ntt import Y\n")
    # support.py is declared but not imported by the sources: only the scan of
    # declared modules finds its import.
    (common / "support.py").write_text("def lazy():\n    import encrypted_scaled_newton\n")
    script = Path(__file__).parents[1] / "common" / "package.py"
    subprocess.run([sys.executable, str(script), "submit", str(problem)], check=True)
    with tarfile.open(problem / "submission.tar.gz") as archive:
        names = set(archive.getnames())
    assert {"common/support.py", "common/encrypted_polynomial.py", "common/encrypted_ntt.py",
            "common/encrypted_scaled_newton.py"} <= names
    assert "common/transformer_runner.py" not in names


def test_submit_skips_editor_lock_links_and_rejects_linked_shared_modules(tmp_path):
    """Broken links in src/ are not shipped or scanned; an imported shared link fails loudly."""
    root = tmp_path / "cadathlon"
    problem = root / "non_agent_pset/p1_benchmark"
    (problem / "src").mkdir(parents=True)
    common = root / "common"
    common.mkdir()
    for name in ("REPORT.md", "cross.patch"):
        (problem / name).write_text("")
    (problem / "src" / "Makefile").write_text("")
    (problem / "src" / ".#runner.py").symlink_to("user@host.4242:1700000000")  # an Emacs lock file
    original = Path(__file__).parents[1] / "common"
    for path in original.iterdir():
        if path.is_file():
            shutil.copy2(path, common / path.name)
    script = common / "package.py"
    subprocess.run([sys.executable, str(script), "submit", str(problem)], check=True)
    with tarfile.open(problem / "submission.tar.gz") as archive:
        assert not any(name.endswith(".#runner.py") for name in archive.getnames())
    (common / "linked_helper_impl.py").write_text("")
    (common / "linked_helper.py").symlink_to(common / "linked_helper_impl.py")
    (problem / "src" / "uses_link.py").write_text("import linked_helper\n")
    result = subprocess.run([sys.executable, str(script), "submit", str(problem)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing imported shared module" in result.stderr
    (common / "linked_helper.py").unlink()
    (common / "linked_helper.py").symlink_to(common / "no_such_module.py")  # a broken link
    result = subprocess.run([sys.executable, str(script), "submit", str(problem)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing imported shared module" in result.stderr


def test_submit_archives_unparsable_sources_without_following_them(tmp_path):
    """A scratch file that does not parse is archived as before, with a warning."""
    root = tmp_path / "cadathlon"
    problem = root / "non_agent_pset/p1_benchmark"
    (problem / "src").mkdir(parents=True)
    common = root / "common"
    common.mkdir()
    for name in ("REPORT.md", "cross.patch"):
        (problem / name).write_text("")
    (problem / "src" / "Makefile").write_text("")
    (problem / "src" / "notes.py").write_text("def unfinished(:\n")
    (problem / "src" / "nul.py").write_bytes(b"x = 1\0\n")
    (problem / "src" / "deep.py").write_text("x = " + "-" * 100000 + "1\n")  # parser limit
    for path in (Path(__file__).parents[1] / "common").iterdir():
        if path.is_file():
            shutil.copy2(path, common / path.name)
    (problem / "src" / "uses_broken.py").write_text("import broken_shared\n")
    (common / "broken_shared.py").write_text("def unfinished(:\n")
    result = subprocess.run([sys.executable, str(common / "package.py"), "submit", str(problem)],
                            capture_output=True, text=True, check=True)
    assert "notes.py does not parse" in result.stderr and "nul.py does not parse" in result.stderr
    assert "deep.py does not parse" in result.stderr
    assert result.stderr.count("broken_shared.py does not parse") == 1
    with tarfile.open(problem / "submission.tar.gz") as archive:
        names = set(archive.getnames())
    assert {"non_agent_pset/p1_benchmark/src/notes.py", "non_agent_pset/p1_benchmark/src/nul.py",
            "non_agent_pset/p1_benchmark/src/deep.py", "common/broken_shared.py"} <= names


def test_extracted_release_runs_fixtures_and_experimental_preflight(tmp_path):
    """Test shipped paths, without relying on the source tree's import paths."""
    source = Path(__file__).parents[1]
    root = tmp_path / "cadathlon"
    problems = ("non_agent_pset/p1_benchmark", "agent_pset/p2_bert",
                "non_agent_pset/p3_cnn", "agent_pset/p4_qwen")
    # P5 is packaged too, but its plaintext reference needs the external WISE
    # planner, so the runner checks below cover the four other problems.
    for problem in problems + ("agent_pset/p5_wise",):
        shutil.copytree(source / problem, root / problem,
                        ignore=shutil.ignore_patterns("__pycache__", "run", "results"))
    shutil.copytree(source / "common", root / "common",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for filename in ("p1_student_suite.json", "p1_student_v6e8/report.json",
                     "p1_legacy_suite.json", "p3_compact_baseline.json"):
        (root / "validation" / filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "validation" / filename, root / "validation" / filename)
    evidence = root / problems[0] / "src/suite/results/C3/evidence/labels.npy"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"organizer-only labels")
    scratch = root / problems[2] / "src/suite/run/active.json"
    scratch.parent.mkdir(parents=True)
    scratch.write_text('{"private": true}')
    # New student calibration dependencies live outside ephemeral run/.
    public_calibration = root / problems[0] / 'src/suite/calibration/student-baseline/test-id'
    public_calibration.mkdir(parents=True)
    (public_calibration/'report.json').write_text('{"aggregate_only": true}')
    (public_calibration/'archives/C3').mkdir(parents=True)
    (public_calibration/'archives/C3/case.json').write_text('{"images": 100}')
    subprocess.run([sys.executable, str(root / "common/package.py"), "release", str(root)], check=True)
    extracted = tmp_path / "extracted"
    with tarfile.open(root / "contestant-release.tar.gz") as archive:
        names = set(archive.getnames())
        assert "common/encrypted_rotation.py" in names
        assert "common/encrypted_compilation.py" in names
        assert "common/encrypted_primitives.py" in names
        assert "common/encrypted_ntt.py" in names
        assert "common/encrypted_packed_qwen.py" in names
        assert "common/encrypted_packed_transformer.py" in names
        assert "common/encrypted_bounded_arithmetic.py" in names
        assert "common/encrypted_bounded_transformer.py" in names
        assert "common/encrypted_streaming_linear.py" in names
        assert "common/encrypted_chained_streaming_linear.py" in names
        assert "common/encrypted_integer_affine.py" in names
        assert "common/encrypted_scaled_newton.py" in names
        assert "validation/p1_student_suite.json" in names
        assert "validation/p3_compact_baseline.json" in names
        assert str((public_calibration/'report.json').relative_to(root)) in names
        assert str((public_calibration/'archives/C3/case.json').relative_to(root)) in names
        assert 'non_agent_pset/p1_benchmark/src/suite/p1_evidence.py' in names
        assert not any(name.endswith(".golden") or "/run/" in name or "/evidence/" in name
                       for name in names)
        archive.extractall(extracted, filter="data")
    # An invalid backend makes accidental JAX initialization fail immediately.
    # Neither plaintext fixture checks nor interface preflight needs a backend.
    env = dict(os.environ, PYTHONPATH="", CROSS_ROOT=str(source.parent),
               JAX_PLATFORMS="packaging_test_no_backend")
    # New helpers must import and execute public scale planning from the
    # extracted closure without a source checkout or valid JAX backend.
    public_probe = "import sys; sys.path.insert(0, 'common'); import encrypted_chained_streaming_linear as s; assert s.ChainedStreamingLinear.__name__ == 'ChainedStreamingLinear'; import encrypted_scaled_newton as n; from encrypted_integer_affine import backward_scale_plan; p=backward_scale_plan(range(101,109),input_scale=11556.,first_output_level=2,updates=2,final_scale=11556.); assert len(p['rows'])==2; assert 'jax' not in sys.modules; assert n.__file__.startswith(str(__import__('pathlib').Path.cwd()))"
    probe=subprocess.run([sys.executable,'-c',public_probe],cwd=extracted,env=env,text=True,capture_output=True)
    assert probe.returncode==0,probe.stdout+probe.stderr
    for problem in problems:
        src = extracted / problem / "src"
        result = subprocess.run([sys.executable, "runner.py", "--reference", "--case", "../test/test1.in"],
                                cwd=src, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["mode"] == "plaintext-reference"
    for problem, reference in ((problems[1], "bert_reference.py"), (problems[1], "bert_packed_reference.py"),
                               (problems[1], "bert_bounded_reference.py"),
                               (problems[1], "bert_optimized_reference.py"),
                               (problems[3], "qwen_reference.py"),
                               (problems[3], "qwen_packed_reference.py"),
                               (problems[3], "qwen_packed_pair_reference.py"),
                               (problems[3], "qwen_bounded_reference.py"),
                               (problems[3], "qwen_optimized_reference.py")):
        src = extracted / problem / "src"
        result = subprocess.run([sys.executable, "runner.py", "--preflight", "--implementation", reference],
                                cwd=src, env=env, text=True, capture_output=True)
        report = json.loads(result.stdout)
        assert report["implementation_status"] == "experimental-unvalidated", result.stderr
        assert report["encryption_validated"] is False
        assert "Interface inspection only" in report["note"]
        if problem == problems[1]:
            # The toy session is present; the full-model file harness is not.
            assert "create_session" not in report["missing_interfaces"]
            assert "client_key_generation" in report["missing_interfaces"]
            assert result.returncode == 1
        else:
            assert not report["missing_interfaces"]
            assert result.returncode == 0
